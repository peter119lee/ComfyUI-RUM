from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline, StableDiffusionXLPipeline
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.utils import apply_lora_scale
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class SDXLTextEncoderOnly:
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        device: str,
        dtype: torch.dtype,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_prompt(self, prompt: str | list[str], device: torch.device | str | None = None):
        target_device = torch.device(device or self.device)
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds_list = []
        for tokenizer, text_encoder in (
            (self.tokenizer, self.text_encoder),
            (self.tokenizer_2, self.text_encoder_2),
        ):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(target_device)
            output = text_encoder(text_input_ids, output_hidden_states=True)
            prompt_embeds = output.hidden_states[-2].to(dtype=self.dtype, device=target_device)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, None, None, None


class RUMFlux2KleinTransformer(Flux2Transformer2DModel):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=guidance_embeds,
            **kwargs,
        )
        self.context_embedder_2 = nn.Linear(2048, self.inner_dim, bias=False)
        self.rum_base_text_tokens = 200

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        hidden_states = self.x_embedder(hidden_states)

        base_text_tokens = int(getattr(self, "rum_base_text_tokens", 200))
        if encoder_hidden_states.shape[1] <= base_text_tokens:
            raise ValueError(
                "RUM transformer 需要拼接 FLUX.2-klein 文本特征和 SDXL CLIP 特征，"
                "但当前 encoder_hidden_states 没有额外 SDXL token。"
            )
        encoder_hidden_states_1 = encoder_hidden_states[:, :base_text_tokens]
        encoder_hidden_states_2 = encoder_hidden_states[:, base_text_tokens:, :2048]
        encoder_hidden_states = torch.cat(
            [self.context_embedder(encoder_hidden_states_1), self.context_embedder_2(encoder_hidden_states_2)],
            dim=1,
        )

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class RUMFlux2KleinPipeline(Flux2KleinPipeline):
    rum_teacher_pipeline: StableDiffusionXLPipeline | SDXLTextEncoderOnly | None = None
    rum_teacher_device: str = "cpu"

    @torch.no_grad()
    def __call__(
        self,
        *,
        prompt: str | list[str],
        max_sequence_length: int = 200,
        text_encoder_out_layers: tuple[int, ...] = (10, 20, 30),
        **kwargs: Any,
    ):
        if self.rum_teacher_pipeline is None:
            raise RuntimeError("RUMFlux2KleinPipeline 尚未 attach SDXL teacher pipeline。")

        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=list(text_encoder_out_layers),
        )

        try:
            sdxl_prompt_embeds = self.rum_teacher_pipeline.encode_prompt(
                prompt=prompt,
                device=torch.device(self.rum_teacher_device),
            )[0]
        except TypeError:
            sdxl_prompt_embeds = self.rum_teacher_pipeline.encode_prompt(prompt)[0]

        feature_dim = prompt_embeds.shape[-1]
        if sdxl_prompt_embeds.shape[-1] > feature_dim:
            raise ValueError(
                f"SDXL prompt embedding 维度 {sdxl_prompt_embeds.shape[-1]} 大于 FLUX embedding 维度 {feature_dim}。"
            )

        sdxl_prompt_embeds = sdxl_prompt_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        sdxl_prompt_embeds = F.pad(sdxl_prompt_embeds, (0, feature_dim - sdxl_prompt_embeds.shape[-1]))

        self.transformer.rum_base_text_tokens = prompt_embeds.shape[1]
        rum_prompt_embeds = torch.cat([prompt_embeds, sdxl_prompt_embeds], dim=1)
        return super().__call__(prompt_embeds=rum_prompt_embeds, **kwargs)


@dataclass
class RUMPipelineHandle:
    pipeline: RUMFlux2KleinPipeline
    device: str
    dtype_name: str
    base_model_path: str
    rum_checkpoint_path: str
    sdxl_model_path: str

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        num_images: int,
        max_sequence_length: int,
        text_encoder_out_layers: tuple[int, ...],
    ):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        output = self.pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            num_images_per_prompt=int(num_images),
            max_sequence_length=int(max_sequence_length),
            text_encoder_out_layers=text_encoder_out_layers,
        )
        return output.images


_PIPELINE_CACHE: dict[tuple[str, ...], RUMPipelineHandle] = {}


def load_rum_pipeline(
    *,
    base_model_path: str,
    rum_checkpoint_path: str,
    sdxl_model_path: str,
    dtype_name: str,
    device: str,
    sdxl_text_device: str,
    unload_sdxl_unet_vae: bool,
    force_reload: bool,
) -> RUMPipelineHandle:
    resolved_device = _resolve_device(device)
    resolved_sdxl_device = _resolve_sdxl_device(sdxl_text_device, resolved_device)
    cache_key = (
        _normalize_path_or_id(base_model_path),
        _normalize_path_or_id(rum_checkpoint_path),
        _normalize_path_or_id(sdxl_model_path),
        dtype_name,
        resolved_device,
        resolved_sdxl_device,
        str(bool(unload_sdxl_unet_vae)),
    )

    if force_reload and cache_key in _PIPELINE_CACHE:
        del _PIPELINE_CACHE[cache_key]
        _clean_memory()

    if cache_key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[cache_key] = _build_rum_pipeline(
            base_model_path=base_model_path,
            rum_checkpoint_path=rum_checkpoint_path,
            sdxl_model_path=sdxl_model_path,
            dtype_name=dtype_name,
            device=resolved_device,
            sdxl_text_device=resolved_sdxl_device,
            unload_sdxl_unet_vae=unload_sdxl_unet_vae,
        )

    return _PIPELINE_CACHE[cache_key]


def clear_pipeline_cache() -> int:
    count = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    _clean_memory()
    return count


def _build_rum_pipeline(
    *,
    base_model_path: str,
    rum_checkpoint_path: str,
    sdxl_model_path: str,
    dtype_name: str,
    device: str,
    sdxl_text_device: str,
    unload_sdxl_unet_vae: bool,
) -> RUMPipelineHandle:
    dtype = _resolve_dtype(dtype_name)

    transformer = RUMFlux2KleinTransformer.from_pretrained(base_model_path, subfolder="transformer")
    checkpoint_path = _resolve_safetensors_path(rum_checkpoint_path)
    state_dict = load_file(str(checkpoint_path), device="cpu")
    try:
        transformer.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        transformer.load_state_dict(state_dict, strict=True)
    transformer.eval()

    pipeline = RUMFlux2KleinPipeline.from_pretrained(
        base_model_path,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipeline.to(device)

    teacher = _load_sdxl_teacher(sdxl_model_path, sdxl_text_device)
    if unload_sdxl_unet_vae:
        _strip_sdxl_generation_modules(teacher)

    pipeline.rum_teacher_pipeline = teacher
    pipeline.rum_teacher_device = sdxl_text_device
    pipeline.set_progress_bar_config(disable=False)
    _clean_memory()

    return RUMPipelineHandle(
        pipeline=pipeline,
        device=device,
        dtype_name=dtype_name,
        base_model_path=base_model_path,
        rum_checkpoint_path=str(checkpoint_path),
        sdxl_model_path=sdxl_model_path,
    )


def _load_sdxl_teacher(path_or_id: str, text_device: str) -> StableDiffusionXLPipeline | SDXLTextEncoderOnly:
    path = Path(path_or_id).expanduser()
    load_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if path.is_file():
        teacher = StableDiffusionXLPipeline.from_single_file(str(path), torch_dtype=load_dtype)
    elif _looks_like_sdxl_text_only(path):
        teacher = _load_sdxl_text_only(str(path), text_device)
    else:
        try:
            teacher = _load_sdxl_text_only(path_or_id, text_device)
        except Exception:
            teacher = StableDiffusionXLPipeline.from_pretrained(path_or_id, torch_dtype=load_dtype)

    text_dtype = torch.float16 if text_device == "cuda" else torch.float32
    for module_name in ("text_encoder", "text_encoder_2"):
        module = getattr(teacher, module_name, None)
        if module is not None:
            module.to(text_device, dtype=text_dtype)
            module.requires_grad_(False)
            module.eval()
    return teacher


def _looks_like_sdxl_text_only(path: Path) -> bool:
    return all(
        (path / relative).exists()
        for relative in (
            "tokenizer/tokenizer_config.json",
            "tokenizer_2/tokenizer_config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
        )
    )


def _load_sdxl_text_only(path_or_id: str, text_device: str) -> SDXLTextEncoderOnly:
    dtype = torch.float16 if text_device == "cuda" else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(path_or_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(path_or_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(path_or_id, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        path_or_id,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
    )
    for module in (text_encoder, text_encoder_2):
        module.to(text_device, dtype=dtype)
        module.requires_grad_(False)
        module.eval()
    return SDXLTextEncoderOnly(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        device=text_device,
        dtype=dtype,
    )


def _strip_sdxl_generation_modules(teacher: StableDiffusionXLPipeline) -> None:
    for module_name in ("unet", "vae"):
        if hasattr(teacher, module_name):
            setattr(teacher, module_name, None)
    _clean_memory()


def _resolve_safetensors_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到 RUM checkpoint：{raw_path}")

    candidates = [
        path / "model.safetensors",
        path / "diffusion_pytorch_model.safetensors",
        path / "transformer" / "diffusion_pytorch_model.safetensors",
        path / "transformer" / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    safetensors_files = sorted(path.rglob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]
    raise FileNotFoundError(
        f"目录 {raw_path} 里没有明确的 RUM safetensors 文件，请直接填 model.safetensors 路径。"
    )


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持 dtype：{dtype_name}")
    return mapping[dtype_name]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你选择了 cuda，但当前 PyTorch 看不到 CUDA。")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"不支持 device：{device}")
    return device


def _resolve_sdxl_device(sdxl_text_device: str, pipeline_device: str) -> str:
    if sdxl_text_device == "same_as_pipeline":
        return pipeline_device
    if sdxl_text_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你选择了 sdxl_text_device=cuda，但当前 PyTorch 看不到 CUDA。")
    if sdxl_text_device not in {"cuda", "cpu"}:
        raise ValueError(f"不支持 sdxl_text_device：{sdxl_text_device}")
    return sdxl_text_device


def _normalize_path_or_id(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def _clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
