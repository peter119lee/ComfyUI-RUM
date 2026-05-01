from __future__ import annotations

import re
import gc
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


_DIFFUSION_PREFIX = "diffusion_model."
_TEXT_ENCODER_CACHE: dict[tuple[str, ...], "RUMDiffusersExactTextEncoder"] = {}
_DIFFUSERS_FLUX2_VAE_CACHE: dict[tuple[str, str], nn.Module] = {}


class RUMDiffusersExactTextEncoder:
    def __init__(
        self,
        *,
        flux2_model_path: str | Path,
        sdxl_text_model_path: str | Path,
        device: str,
        qwen_dtype: torch.dtype,
        sdxl_dtype: torch.dtype,
    ):
        from diffusers import Flux2KleinPipeline
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self.device = torch.device(device)
        self.qwen_dtype = qwen_dtype
        self.sdxl_dtype = sdxl_dtype
        flux2_path = Path(flux2_model_path).expanduser()
        sdxl_path = Path(sdxl_text_model_path).expanduser()
        self.flux2_model_path = str(flux2_path)
        self.sdxl_text_model_path = str(sdxl_path)

        qwen_pipe = Flux2KleinPipeline.from_pretrained(
            str(flux2_path),
            transformer=None,
            vae=None,
            torch_dtype=qwen_dtype,
        )
        self.qwen_tokenizer = qwen_pipe.tokenizer
        self.qwen_text_encoder = qwen_pipe.text_encoder
        self.qwen_text_encoder.to(self.device)
        self.qwen_text_encoder.requires_grad_(False)
        self.qwen_text_encoder.eval()
        qwen_pipe.tokenizer = None
        qwen_pipe.text_encoder = None
        del qwen_pipe

        self.sdxl_tokenizer = CLIPTokenizer.from_pretrained(str(sdxl_path), subfolder="tokenizer")
        self.sdxl_tokenizer_2 = CLIPTokenizer.from_pretrained(str(sdxl_path), subfolder="tokenizer_2")
        self.sdxl_text_encoder = CLIPTextModel.from_pretrained(
            str(sdxl_path),
            subfolder="text_encoder",
            torch_dtype=sdxl_dtype,
        )
        self.sdxl_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            str(sdxl_path),
            subfolder="text_encoder_2",
            torch_dtype=sdxl_dtype,
        )
        for module in (self.sdxl_text_encoder, self.sdxl_text_encoder_2):
            module.to(self.device, dtype=sdxl_dtype)
            module.requires_grad_(False)
            module.eval()

    @torch.inference_mode()
    def encode(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        guidance: float | None,
        base_text_tokens: int,
        negative_text_tokens: int,
        extra_text_tokens: int,
        positive_qwen_layers: Iterable[int],
        negative_qwen_layers: Iterable[int],
    ):
        positive_qwen_layers = tuple(int(layer) for layer in positive_qwen_layers)
        negative_qwen_layers = tuple(int(layer) for layer in negative_qwen_layers)
        if len(positive_qwen_layers) != 3:
            raise ValueError("positive_qwen_layers 必须正好包含 3 个层号，例如 10,20,30。")
        if len(negative_qwen_layers) != 3:
            raise ValueError("negative_qwen_layers 必须正好包含 3 个层号，例如 9,18,27。")

        positive_qwen = self._encode_qwen(
            prompt=prompt,
            max_sequence_length=base_text_tokens,
            hidden_states_layers=positive_qwen_layers,
        )
        positive_sdxl = self._encode_sdxl(prompt)[:, :extra_text_tokens]
        positive_sdxl = positive_sdxl.to(device=positive_qwen.device, dtype=positive_qwen.dtype)
        positive_sdxl = _fit_last_dim(positive_sdxl, positive_qwen.shape[-1])

        positive = torch.cat([positive_qwen, positive_sdxl], dim=1).to(device="cpu", dtype=torch.float32)
        positive_meta = {
            "rum_base_text_tokens": int(base_text_tokens),
            "rum_extra_text_tokens": int(extra_text_tokens),
        }
        if guidance is not None:
            positive_meta["guidance"] = float(guidance)

        negative = self._encode_qwen(
            prompt=negative_prompt,
            max_sequence_length=negative_text_tokens,
            hidden_states_layers=negative_qwen_layers,
        ).to(device="cpu", dtype=torch.float32)

        return [[positive, positive_meta]], [[negative, {}]]

    def _encode_qwen(
        self,
        *,
        prompt: str,
        max_sequence_length: int,
        hidden_states_layers: Iterable[int],
    ) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.qwen_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
        )
        output = self.qwen_text_encoder(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[int(layer)] for layer in hidden_states_layers], dim=1)
        stacked = stacked.to(dtype=self.qwen_dtype, device=self.device)
        batch_size, layer_count, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, layer_count * hidden_dim)

    def _encode_sdxl(self, prompt: str) -> torch.Tensor:
        prompt_embeds = []
        for tokenizer, text_encoder in (
            (self.sdxl_tokenizer, self.sdxl_text_encoder),
            (self.sdxl_tokenizer_2, self.sdxl_text_encoder_2),
        ):
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            output = text_encoder(
                text_inputs.input_ids.to(self.device),
                output_hidden_states=True,
            )
            prompt_embeds.append(output.hidden_states[-2].to(dtype=self.sdxl_dtype, device=self.device))
        return torch.cat(prompt_embeds, dim=-1)


def encode_diffusers_exact_rum_text(
    *,
    prompt: str,
    negative_prompt: str = "",
    flux2_model_path: str | Path,
    sdxl_text_model_path: str | Path,
    device: str = "cuda",
    qwen_dtype: str = "bfloat16",
    sdxl_dtype: str = "float16",
    unload_after_encode: bool = False,
    guidance: float | None = None,
    base_text_tokens: int = 200,
    negative_text_tokens: int = 512,
    extra_text_tokens: int = 77,
    positive_qwen_layers: str | Iterable[int] = "10,20,30",
    negative_qwen_layers: str | Iterable[int] = "9,18,27",
):
    resolved_device = _resolve_text_device(device)
    qwen_torch_dtype = _resolve_dtype(qwen_dtype)
    sdxl_torch_dtype = _resolve_dtype(sdxl_dtype)
    positive_layers = _parse_qwen_layers(positive_qwen_layers)
    negative_layers = _parse_qwen_layers(negative_qwen_layers)
    cache_key = (
        str(Path(flux2_model_path).expanduser().resolve()),
        str(Path(sdxl_text_model_path).expanduser().resolve()),
        resolved_device,
        str(qwen_torch_dtype),
        str(sdxl_torch_dtype),
    )
    encoder = _TEXT_ENCODER_CACHE.get(cache_key)
    if encoder is None:
        encoder = RUMDiffusersExactTextEncoder(
            flux2_model_path=flux2_model_path,
            sdxl_text_model_path=sdxl_text_model_path,
            device=resolved_device,
            qwen_dtype=qwen_torch_dtype,
            sdxl_dtype=sdxl_torch_dtype,
        )
        _TEXT_ENCODER_CACHE[cache_key] = encoder

    positive, negative = encoder.encode(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance=guidance,
        base_text_tokens=base_text_tokens,
        negative_text_tokens=negative_text_tokens,
        extra_text_tokens=extra_text_tokens,
        positive_qwen_layers=positive_layers,
        negative_qwen_layers=negative_layers,
    )
    if unload_after_encode:
        _TEXT_ENCODER_CACHE.pop(cache_key, None)
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return positive, negative



def _parse_qwen_layers(value: str | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        layers = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        layers = tuple(int(layer) for layer in value)
    if len(layers) != 3:
        raise ValueError("Qwen 层号必须正好是 3 个整数，例如 10,20,30。")
    return layers


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持 dtype：{dtype_name}")
    return mapping[dtype_name]


def _resolve_text_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("你选择了 cuda，但当前 PyTorch 看不到 CUDA。")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"不支持 text encoder device：{device}")
    return device


class RUMDualTextProjection(nn.Module):
    def __init__(self, base_weight: torch.Tensor, extra_weight: torch.Tensor, base_text_tokens: int):
        super().__init__()
        if base_weight.ndim != 2:
            raise ValueError(f"context_embedder.weight 必须是 2D tensor，实际是 {tuple(base_weight.shape)}。")
        if extra_weight.ndim != 2:
            raise ValueError(f"context_embedder_2.weight 必须是 2D tensor，实际是 {tuple(extra_weight.shape)}。")
        if base_weight.shape[0] != extra_weight.shape[0]:
            raise ValueError(
                "context_embedder.weight 和 context_embedder_2.weight 输出维度不同："
                f"{base_weight.shape[0]} != {extra_weight.shape[0]}。"
            )
        if base_text_tokens <= 0:
            raise ValueError("base_text_tokens 必须大于 0。")

        self.base_text_tokens = int(base_text_tokens)
        self.base_in_features = int(base_weight.shape[1])
        self.extra_in_features = int(extra_weight.shape[1])
        self.out_features = int(base_weight.shape[0])
        self.base_weight = nn.Parameter(base_weight.detach().contiguous(), requires_grad=False)
        self.extra_weight = nn.Parameter(extra_weight.detach().contiguous(), requires_grad=False)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        base_tokens = min(self.base_text_tokens, text.shape[1])
        base_text = _fit_last_dim(text[:, :base_tokens], self.base_in_features)
        base_weight = self.base_weight.to(device=base_text.device, dtype=base_text.dtype)
        base_out = F.linear(base_text, base_weight)

        extra_text = text[:, base_tokens:]
        if extra_text.shape[1] == 0:
            return base_out

        extra_text = _fit_last_dim(extra_text, self.extra_in_features)
        extra_weight = self.extra_weight.to(device=extra_text.device, dtype=extra_text.dtype)
        extra_out = F.linear(extra_text, extra_weight)
        return torch.cat([base_out, extra_out], dim=1)


def combine_rum_conditioning(
    flux2_conditioning,
    sdxl_conditioning,
    *,
    guidance: float | None,
    base_text_tokens: int,
    extra_text_tokens: int,
    sdxl_clip_width: int,
    use_sdxl_extra: bool = True,
):
    if not flux2_conditioning:
        raise ValueError("flux2_conditioning 不能为空。")
    if not sdxl_conditioning:
        raise ValueError("sdxl_conditioning 不能为空。")
    if base_text_tokens <= 0:
        raise ValueError("base_text_tokens 必须大于 0；FLUX.2-Klein 原生 ComfyUI 通常是 512。")
    if extra_text_tokens <= 0:
        raise ValueError("extra_text_tokens 必须大于 0；SDXL CLIP 通常是 77。")
    if sdxl_clip_width <= 0:
        raise ValueError("sdxl_clip_width 必须大于 0；SDXL CLIP-L + CLIP-G 通常是 2048。")

    output = []
    for index, flux_item in enumerate(flux2_conditioning):
        sdxl_item = sdxl_conditioning[min(index, len(sdxl_conditioning) - 1)]
        flux_embeds, flux_meta = flux_item
        sdxl_embeds, _ = sdxl_item

        if flux_embeds.ndim != 3 or sdxl_embeds.ndim != 3:
            raise ValueError(
                "RUM conditioning 需要 3D tensor：[batch, tokens, channels]。"
                f"当前 FLUX={tuple(flux_embeds.shape)}，SDXL={tuple(sdxl_embeds.shape)}。"
            )

        flux_base = flux_embeds[:, :base_text_tokens]
        if use_sdxl_extra:
            sdxl_extra = sdxl_embeds.to(device=flux_embeds.device, dtype=flux_embeds.dtype)
            sdxl_extra = sdxl_extra[:, :extra_text_tokens]
            sdxl_extra = _fit_last_dim(sdxl_extra, sdxl_clip_width)
            sdxl_extra = _fit_last_dim(sdxl_extra, flux_base.shape[-1])
        else:
            sdxl_extra = flux_base.new_zeros((flux_base.shape[0], extra_text_tokens, flux_base.shape[-1]))

        combined = torch.cat([flux_base, sdxl_extra], dim=1)
        meta = flux_meta.copy()
        if guidance is None:
            meta.pop("guidance", None)
        else:
            meta["guidance"] = float(guidance)
        meta["rum_base_text_tokens"] = int(flux_base.shape[1])
        meta["rum_extra_text_tokens"] = int(sdxl_extra.shape[1])
        meta.pop("attention_mask", None)
        output.append([combined, meta])
    return output


class RUMDiffusersMatchTokenPolicy:
    def __init__(self, base_text_tokens: int, extra_text_tokens: int):
        if base_text_tokens <= 0:
            raise ValueError("diffusers match base_text_tokens 必须大于 0。")
        if extra_text_tokens <= 0:
            raise ValueError("diffusers match extra_text_tokens 必须大于 0。")
        self.base_text_tokens = int(base_text_tokens)
        self.extra_text_tokens = int(extra_text_tokens)

    @property
    def total_tokens(self) -> int:
        return self.base_text_tokens + self.extra_text_tokens

    def select(self, cross_attn: torch.Tensor, branch: str | None) -> torch.Tensor:
        if branch == "negative":
            return self.negative(cross_attn)
        if branch == "positive":
            return self.positive(cross_attn)
        return self.legacy(cross_attn)

    def positive(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] == self.total_tokens:
            return cross_attn
        if cross_attn.shape[1] <= 512:
            return cross_attn[:, -self.total_tokens :]
        base = cross_attn[:, : self.base_text_tokens]
        extra = cross_attn[:, -self.extra_text_tokens :]
        return torch.cat([base, extra], dim=1)

    def negative(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] <= 512:
            return cross_attn
        return cross_attn[:, :512]

    def legacy(self, cross_attn: torch.Tensor) -> torch.Tensor:
        if cross_attn.shape[1] <= self.total_tokens:
            return cross_attn
        if cross_attn.shape[1] == 512:
            return cross_attn
        if cross_attn.shape[1] < 512:
            return cross_attn[:, -self.total_tokens :]
        base = cross_attn[:, : self.base_text_tokens]
        extra = cross_attn[:, -self.extra_text_tokens :]
        return torch.cat([base, extra], dim=1)


class RUMDiffusersMatchExtraConds:
    def __init__(
        self,
        original_extra_conds,
        token_policy: RUMDiffusersMatchTokenPolicy,
        *,
        disable_guidance: bool,
    ):
        self.original_extra_conds = original_extra_conds
        self.token_policy = token_policy
        self.disable_guidance = bool(disable_guidance)

    def __call__(self, **kwargs):
        output = self.original_extra_conds(**kwargs)
        if self.disable_guidance:
            output = output.copy()
            output.pop("guidance", None)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is None:
            return output

        branch = kwargs.get("rum_diffusers_cfg_branch")
        if branch is None:
            branch = kwargs.get("prompt_type")
        if branch is None:
            branch = kwargs.get("transformer_options", {}).get("rum_diffusers_cfg_branch")
        selected = self.token_policy.select(cross_attn, branch)

        try:
            import comfy.conds
        except Exception:
            return output

        output = output.copy()
        output["c_crossattn"] = comfy.conds.CONDRegular(selected)
        return output


def apply_diffusers_match_model_wrapper(
    model,
    *,
    base_text_tokens: int,
    extra_text_tokens: int,
    disable_guidance: bool = True,
):
    import comfy.patcher_extension

    patched = model.clone()
    patched.rum_diffusers_match_config = {
        "base_text_tokens": int(base_text_tokens),
        "extra_text_tokens": int(extra_text_tokens),
        "disable_guidance": bool(disable_guidance),
    }
    patched.add_wrapper(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        RUMDiffusersTimestepWrapper(),
    )
    return patched


def _linear_split(module: nn.Module, tensor: torch.Tensor, start: int, width: int) -> torch.Tensor:
    weight = module.weight.narrow(0, start, width)
    bias = getattr(module, "bias", None)
    if bias is not None:
        bias = bias.narrow(0, start, width)
    if weight.device != tensor.device or weight.dtype != tensor.dtype:
        weight = weight.to(device=tensor.device, dtype=tensor.dtype)
    if bias is not None and (bias.device != tensor.device or bias.dtype != tensor.dtype):
        bias = bias.to(device=tensor.device, dtype=tensor.dtype)
    return F.linear(tensor, weight, bias)


def _diffusers_rope_from_comfy_pe(pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pe.ndim != 6 or pe.shape[-2:] != (2, 2):
        raise ValueError(f"RUM diffusers-match RoPE 需要 ComfyUI FLUX pe tensor，当前形状是 {tuple(pe.shape)}。")
    matrix = pe[0, 0]
    cos = matrix[..., 0, 0]
    sin = matrix[..., 1, 0]
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def _diffusers_apply_rotary_emb(tensor: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rope
    cos = cos[None, :, None, :].to(device=tensor.device)
    sin = sin[None, :, None, :].to(device=tensor.device)
    real, imag = tensor.reshape(*tensor.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(3)
    return (tensor.float() * cos + rotated.float() * sin).to(tensor.dtype)


def _diffusers_rms_norm(tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if weight.device != tensor.device or weight.dtype != tensor.dtype:
        weight = weight.to(device=tensor.device, dtype=tensor.dtype)
    return F.rms_norm(tensor, (tensor.shape[-1],), weight, eps)


def _diffusers_native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    try:
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=False,
        )
    except TypeError:
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    return output.permute(0, 2, 1, 3)


def _rum_diffusers_double_block_forward(
    self,
    img: torch.Tensor,
    txt: torch.Tensor,
    vec,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    transformer_options=None,
):
    if modulation_dims_img is not None or modulation_dims_txt is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    transformer_options = transformer_options or {}
    if transformer_options.get("patches"):
        raise ValueError("RUM diffusers-match exact forward 暂不支持 attention patches / LoRA patches。")

    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    norm_img = self.img_norm1(img)
    norm_img = (1 + img_mod1.scale) * norm_img + img_mod1.shift
    norm_txt = self.txt_norm1(txt)
    norm_txt = (1 + txt_mod1.scale) * norm_txt + txt_mod1.shift

    hidden_size = self.hidden_size
    img_query = _linear_split(self.img_attn.qkv, norm_img, 0, hidden_size)
    img_key = _linear_split(self.img_attn.qkv, norm_img, hidden_size, hidden_size)
    img_value = _linear_split(self.img_attn.qkv, norm_img, hidden_size * 2, hidden_size)
    txt_query = _linear_split(self.txt_attn.qkv, norm_txt, 0, hidden_size)
    txt_key = _linear_split(self.txt_attn.qkv, norm_txt, hidden_size, hidden_size)
    txt_value = _linear_split(self.txt_attn.qkv, norm_txt, hidden_size * 2, hidden_size)

    img_query = img_query.unflatten(-1, (self.num_heads, -1))
    img_key = img_key.unflatten(-1, (self.num_heads, -1))
    img_value = img_value.unflatten(-1, (self.num_heads, -1))
    txt_query = txt_query.unflatten(-1, (self.num_heads, -1))
    txt_key = txt_key.unflatten(-1, (self.num_heads, -1))
    txt_value = txt_value.unflatten(-1, (self.num_heads, -1))

    img_query = _diffusers_rms_norm(img_query, self.img_attn.norm.query_norm.weight)
    img_key = _diffusers_rms_norm(img_key, self.img_attn.norm.key_norm.weight)
    txt_query = _diffusers_rms_norm(txt_query, self.txt_attn.norm.query_norm.weight)
    txt_key = _diffusers_rms_norm(txt_key, self.txt_attn.norm.key_norm.weight)

    query = torch.cat((txt_query, img_query), dim=1)
    key = torch.cat((txt_key, img_key), dim=1)
    value = torch.cat((txt_value, img_value), dim=1)
    rope = _diffusers_rope_from_comfy_pe(pe)
    query = _diffusers_apply_rotary_emb(query, rope)
    key = _diffusers_apply_rotary_emb(key, rope)

    attn = _diffusers_native_attention(query, key, value, attn_mask=attn_mask).flatten(2, 3).to(query.dtype)
    txt_attn, img_attn = attn.split_with_sizes((txt.shape[1], img.shape[1]), dim=1)
    txt_attn = self.txt_attn.proj(txt_attn)
    img_attn = self.img_attn.proj(img_attn)

    img = img + img_mod1.gate * img_attn
    norm_img = self.img_norm2(img)
    norm_img = norm_img * (1 + img_mod2.scale) + img_mod2.shift
    img = img + img_mod2.gate * self.img_mlp(norm_img)

    txt = txt + txt_mod1.gate * txt_attn
    norm_txt = self.txt_norm2(txt)
    norm_txt = norm_txt * (1 + txt_mod2.scale) + txt_mod2.shift
    txt = txt + txt_mod2.gate * self.txt_mlp(norm_txt)
    return img, txt


def _rum_diffusers_single_block_forward(
    self,
    x: torch.Tensor,
    vec,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims=None,
    transformer_options=None,
) -> torch.Tensor:
    if modulation_dims is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    transformer_options = transformer_options or {}
    if transformer_options.get("patches"):
        raise ValueError("RUM diffusers-match exact forward 暂不支持 attention patches / LoRA patches。")

    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec

    hidden_states = self.pre_norm(x)
    hidden_states = (1 + mod.scale) * hidden_states + mod.shift
    projected = self.linear1(hidden_states)
    qkv, mlp = torch.split(projected, [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
    query, key, value = qkv.chunk(3, dim=-1)

    query = query.unflatten(-1, (self.num_heads, -1))
    key = key.unflatten(-1, (self.num_heads, -1))
    value = value.unflatten(-1, (self.num_heads, -1))
    query = _diffusers_rms_norm(query, self.norm.query_norm.weight)
    key = _diffusers_rms_norm(key, self.norm.key_norm.weight)

    rope = _diffusers_rope_from_comfy_pe(pe)
    query = _diffusers_apply_rotary_emb(query, rope)
    key = _diffusers_apply_rotary_emb(key, rope)
    hidden_states = _diffusers_native_attention(query, key, value, attn_mask=attn_mask).flatten(2, 3).to(query.dtype)

    mlp = self.mlp_act(mlp)
    output = self.linear2(torch.cat((hidden_states, mlp), dim=-1))
    return x + mod.gate * output


def _rum_diffusers_final_layer_forward(self, x: torch.Tensor, vec: torch.Tensor, modulation_dims=None) -> torch.Tensor:
    if modulation_dims is not None:
        raise ValueError("RUM diffusers-match exact forward 暂不支持 reference/timestep-zero modulation。")
    if vec.ndim != 2:
        vec = vec.reshape(vec.shape[0], -1)

    modulation = self.adaLN_modulation[1](self.adaLN_modulation[0](vec).to(x.dtype))
    shift, scale = modulation.chunk(2, dim=1)
    x = self.norm_final(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    return self.linear(x)


@contextmanager
def _diffusers_match_forward_patch(diffusion_model):
    originals = []
    try:
        for block in getattr(diffusion_model, "double_blocks", []):
            originals.append((block, block.forward))
            block.forward = MethodType(_rum_diffusers_double_block_forward, block)
        for block in getattr(diffusion_model, "single_blocks", []):
            originals.append((block, block.forward))
            block.forward = MethodType(_rum_diffusers_single_block_forward, block)
        final_layer = getattr(diffusion_model, "final_layer", None)
        if final_layer is not None:
            originals.append((final_layer, final_layer.forward))
            final_layer.forward = MethodType(_rum_diffusers_final_layer_forward, final_layer)
        yield
    finally:
        for block, original_forward in reversed(originals):
            block.forward = original_forward


class RUMDiffusersTimestepWrapper:
    def __call__(self, apply_model, x, timestep, context, *args, **kwargs):
        diffusion_model = apply_model.class_obj
        original_time_in = diffusion_model.time_in
        diffusion_dtype = diffusion_model.dtype

        class DiffusersTimeIn(nn.Module):
            def forward(self, ignored_embedding):
                from comfy.ldm.flux.layers import timestep_embedding

                diffusers_timestep = (timestep.to(torch.float32) * 1000.0).to(dtype=diffusion_dtype)
                embedding = timestep_embedding(diffusers_timestep, 256, time_factor=1.0).to(
                    device=diffusers_timestep.device,
                    dtype=diffusion_dtype,
                )
                return original_time_in(embedding)

        diffusion_model.time_in = DiffusersTimeIn()
        try:
            with _diffusers_match_forward_patch(diffusion_model):
                return apply_model(x, timestep, context, *args, **kwargs)
        finally:
            diffusion_model.time_in = original_time_in



def _condition_tensor(model_cond):
    value = getattr(model_cond, "cond", model_cond)
    return value


def _predict_raw_noise(model, x, timestep, cond, model_options, branch: str):
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.samplers

    area_cond = comfy.samplers.get_area_and_mult(cond, x, timestep)
    if area_cond is None:
        raise ValueError(f"RUM diffusers-match {branch} conditioning 当前 timestep 没有可用区域。")
    if area_cond.area is not None:
        raise ValueError("RUM diffusers-match raw-noise 路径暂不支持 area/mask conditioning；请用全图 conditioning 做对齐验证。")

    diffusion_model = model.diffusion_model
    diffusion_dtype = model.get_dtype_inference()
    x_in = model.model_sampling.calculate_input(timestep, area_cond.input_x).to(diffusion_dtype)
    device = x_in.device

    conditioning = area_cond.conditioning
    context = _condition_tensor(conditioning["c_crossattn"])
    context = comfy.model_management.cast_to_device(context, device, diffusion_dtype)

    transformer_options = model.current_patcher.apply_hooks(hooks=area_cond.hooks)
    if "transformer_options" in model_options:
        transformer_options = comfy.patcher_extension.merge_nested_dicts(
            transformer_options,
            model_options["transformer_options"],
            copy_dict1=False,
        )
    if area_cond.patches is not None:
        transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
            transformer_options.get("patches", {}),
            area_cond.patches,
        )
    transformer_options["cond_or_uncond"] = [0]
    transformer_options["uuids"] = [area_cond.uuid]
    transformer_options["sigmas"] = timestep
    transformer_options["rum_diffusers_cfg_branch"] = branch

    guidance = _condition_tensor(conditioning["guidance"]) if "guidance" in conditioning else None
    timestep_in = ((timestep.to(torch.float32) * 1000.0).to(diffusion_dtype) / 1000.0).to(diffusion_dtype)

    with torch.inference_mode(), _diffusers_match_forward_patch(diffusion_model):
        model_output = diffusion_model(
            x_in,
            timestep_in,
            context=context,
            control=area_cond.control,
            transformer_options=transformer_options,
            guidance=guidance,
        )

    if len(model_output) > 1 and not torch.is_tensor(model_output):
        from comfy import utils

        model_output, _ = utils.pack_latents(model_output)
    return model_output



def create_diffusers_cfg_guider(model, positive, negative, cfg: float):
    import comfy.model_patcher
    import comfy.samplers

    def branch_options(model_options, branch: str):
        options = comfy.model_patcher.create_model_options_clone(model_options)
        transformer_options = options.setdefault("transformer_options", {})
        transformer_options["rum_diffusers_cfg_branch"] = branch
        options["disable_cfg1_optimization"] = True
        return options

    class RUMDiffusersCFGGuider(comfy.samplers.CFGGuider):
        def inner_sample(
            self,
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=None,
        ):
            match_config = getattr(self.model_patcher, "rum_diffusers_match_config", None)
            if match_config is None:
                return super().inner_sample(
                    noise,
                    latent_image,
                    device,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )

            original_extra_conds = self.inner_model.extra_conds
            token_policy = RUMDiffusersMatchTokenPolicy(
                match_config["base_text_tokens"],
                match_config["extra_text_tokens"],
            )
            self.inner_model.extra_conds = RUMDiffusersMatchExtraConds(
                original_extra_conds,
                token_policy,
                disable_guidance=match_config.get("disable_guidance", True),
            )
            try:
                return super().inner_sample(
                    noise,
                    latent_image,
                    device,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )
            finally:
                self.inner_model.extra_conds = original_extra_conds

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            positive_cond = self.conds.get("positive", None)
            negative_cond = self.conds.get("negative", None)

            if model_options.get("rum_diffusers_return_raw_noise", False):
                positive_out = _predict_raw_noise(
                    self.inner_model,
                    x,
                    timestep,
                    positive_cond[0],
                    model_options,
                    "positive",
                )
                if negative_cond is None or self.cfg <= 1.0:
                    return positive_out
                negative_out = _predict_raw_noise(
                    self.inner_model,
                    x,
                    timestep,
                    negative_cond[0],
                    model_options,
                    "negative",
                )
                return negative_out + self.cfg * (positive_out - negative_out)

            positive_out = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [positive_cond],
                x,
                timestep,
                branch_options(model_options, "positive"),
            )[0]
            if negative_cond is None or self.cfg <= 1.0:
                return positive_out

            negative_out = comfy.samplers.calc_cond_batch(
                self.inner_model,
                [negative_cond],
                x,
                timestep,
                branch_options(model_options, "negative"),
            )[0]
            return comfy.samplers.cfg_function(
                self.inner_model,
                positive_out,
                negative_out,
                self.cfg,
                x,
                timestep,
                model_options=model_options,
                cond=positive_cond,
                uncond=negative_cond,
            )

    guider = RUMDiffusersCFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(float(cfg))
    return guider


class RUMDiffusersEulerSampler:
    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        import math

        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        import torch
        from tqdm.auto import trange

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if denoise_mask is not None:
            raise ValueError("RUM diffusers-match sampler 暂不支持 denoise_mask；请用全图采样来做对齐验证。")

        model_dtype = getattr(model_wrap.inner_model, "get_dtype_inference", lambda: noise.dtype)()
        x = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0],
            noise,
            latent_image,
            self.max_denoise(model_wrap, sigmas),
        ).to(dtype=model_dtype)

        model_options = extra_args.get("model_options", {})
        seed = extra_args.get("seed", None)
        total_steps = len(sigmas) - 1
        step_iter = trange(total_steps, disable=disable_pbar)
        for i in step_iter:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_in = sigma.expand(x.shape[0]).to(device=x.device)
            step_model_options = model_options.copy()
            step_model_options["rum_diffusers_return_raw_noise"] = True
            noise_pred = model_wrap(x, sigma_in, model_options=step_model_options, seed=seed)

            dt = (sigma_next - sigma).to(device=x.device, dtype=torch.float32)
            x = (x.float() + dt * noise_pred).to(dtype=noise_pred.dtype)

            if callback is not None:
                denoised = x.float() - noise_pred.float() * sigma.to(device=x.device, dtype=torch.float32)
                callback(i, denoised, x, total_steps)

        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], x.float())


def create_diffusers_euler_sampler():
    return RUMDiffusersEulerSampler()


def diffusers_flux2_sigmas(*, steps: int, width: int, height: int) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps 必须大于 0。")
    if width <= 0 or height <= 0:
        raise ValueError("width/height 必须大于 0。")

    import numpy as np

    image_seq_len = round(width * height / (16 * 16))
    mu = _diffusers_compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
    sigmas = np.linspace(1.0, 1.0 / int(steps), int(steps), dtype=np.float32)
    sigmas = np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1))
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _diffusers_compute_empirical_mu(*, image_seq_len: int, num_steps: int) -> float:
    # Matches diffusers Flux2KleinPipeline.compute_empirical_mu, not ComfyUI's Flux2Scheduler approximation.
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _diffusers_time_shift_exponential(mu: float, sigma: float, timesteps: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.tensor(mu, dtype=timesteps.dtype, device=timesteps.device)) / (
        torch.exp(torch.tensor(mu, dtype=timesteps.dtype, device=timesteps.device))
        + (1 / timesteps - 1) ** sigma
    )



def _unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def _resolve_vae_first_stage(vae):
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is None:
        raise ValueError("当前 VAE 对象没有 first_stage_model，不能执行 RUM diffusers-exact decode。")
    return first_stage


def _load_diffusers_flux2_vae(flux2_model_path: str | Path, dtype: torch.dtype) -> nn.Module:
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

    model_path = str(Path(flux2_model_path).expanduser())
    cache_key = (model_path, str(dtype))
    diffusers_vae = _DIFFUSERS_FLUX2_VAE_CACHE.get(cache_key)
    if diffusers_vae is None:
        diffusers_vae = AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        diffusers_vae.requires_grad_(False)
        diffusers_vae.eval()
        _DIFFUSERS_FLUX2_VAE_CACHE[cache_key] = diffusers_vae
    return diffusers_vae


@torch.inference_mode()
def _decode_with_diffusers_flux2_vae(latent: torch.Tensor, flux2_model_path: str | Path, dtype: torch.dtype) -> torch.Tensor:
    import comfy.model_management
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2ImageProcessor

    device = comfy.model_management.vae_device()
    offload_device = comfy.model_management.vae_offload_device()
    diffusers_vae = _load_diffusers_flux2_vae(flux2_model_path, dtype)
    diffusers_vae.to(device)

    latents = latent.to(device=device, dtype=dtype)
    bn_mean = diffusers_vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    bn_std = torch.sqrt(
        diffusers_vae.bn.running_var.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        + float(diffusers_vae.config.batch_norm_eps)
    )
    latents = _unpatchify_flux2_latents(latents * bn_std + bn_mean)
    decoded = diffusers_vae.decode(latents, return_dict=False)[0]

    processor = Flux2ImageProcessor(vae_scale_factor=16)
    images_np = processor.postprocess(decoded, output_type="np")
    images_uint8 = (images_np * 255).round().astype("uint8")
    images = torch.from_numpy(images_uint8).to(dtype=torch.float32).div_(255.0)
    diffusers_vae.to(offload_device)
    return images


@torch.inference_mode()
def decode_diffusers_exact_flux2_latent(vae, samples, flux2_model_path: str | Path | None = None):
    import comfy.model_management

    latent = samples["samples"] if isinstance(samples, dict) else samples
    if latent.is_nested:
        latent = latent.unbind()[0]

    dtype = getattr(vae, "vae_dtype", torch.bfloat16) if vae is not None else torch.bfloat16
    if flux2_model_path is not None and str(flux2_model_path).strip():
        return _decode_with_diffusers_flux2_vae(latent, str(flux2_model_path).strip(), dtype)

    first_stage = _resolve_vae_first_stage(vae)
    device = comfy.model_management.vae_device()
    offload_device = comfy.model_management.vae_offload_device()
    memory_used = vae.memory_used_decode(latent.shape, dtype=dtype)
    comfy.model_management.load_models_gpu([vae.patcher], memory_required=memory_used)

    latents = latent.to(device=device, dtype=dtype)
    if not hasattr(first_stage, "bn"):
        raise ValueError("当前 VAE 不是 FLUX.2 diffusers VAE：找不到 batch norm 参数 bn。")

    first_stage = first_stage.to(device)
    decoded = first_stage.decode(latents)
    images = vae.process_output(decoded.to(torch.float32)).movedim(1, -1)
    first_stage.to(offload_device)
    return images

def load_rum_native_model(checkpoint_path: str | Path, *, base_text_tokens: int = 200):
    import comfy.sd

    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    state_dict = load_file(str(checkpoint), device="cpu")
    base_weight = _require_key(state_dict, "context_embedder.weight")
    extra_weight = _require_key(state_dict, "context_embedder_2.weight")
    converted = convert_rum_diffusers_to_comfy(state_dict, output_prefix="")
    converted_count = len(converted)
    model = comfy.sd.load_diffusion_model_state_dict(converted)
    if model is None:
        raise ValueError("无法把 RUM checkpoint 加载为 ComfyUI native MODEL。")

    dual_projection = RUMDualTextProjection(base_weight, extra_weight, base_text_tokens=base_text_tokens)
    model.add_object_patch(f"{_DIFFUSION_PREFIX}txt_in", dual_projection)
    return model, converted_count, str(checkpoint)


def apply_rum_model_patch(model, checkpoint_path: str | Path, *, base_text_tokens: int, strict: bool):
    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    state_dict = load_file(str(checkpoint), device="cpu")
    base_weight = _require_key(state_dict, "context_embedder.weight")
    extra_weight = _require_key(state_dict, "context_embedder_2.weight")
    converted = convert_rum_diffusers_to_comfy(state_dict, output_prefix=_DIFFUSION_PREFIX)
    converted.pop(f"{_DIFFUSION_PREFIX}txt_in.weight", None)

    patched = model.clone()
    dual_projection = RUMDualTextProjection(base_weight, extra_weight, base_text_tokens=base_text_tokens)
    patched.add_object_patch(f"{_DIFFUSION_PREFIX}txt_in", dual_projection)

    model_keys = set(patched.model_state_dict().keys())
    converted = _adapt_converted_keys_to_model(converted, model_keys)
    missing = sorted(key for key in converted if key not in model_keys)
    shape_errors = []
    for key, tensor in converted.items():
        if key not in model_keys:
            continue
        current = patched.get_model_object(key)
        current_shape = tuple(current.shape)
        if current_shape != tuple(tensor.shape):
            shape_errors.append(f"{key}: checkpoint {tuple(tensor.shape)} != model {current_shape}")

    if strict and (missing or shape_errors):
        details = []
        if missing:
            details.append("缺少目标权重：" + ", ".join(missing[:20]))
        if shape_errors:
            details.append("形状不匹配：" + "; ".join(shape_errors[:20]))
        raise ValueError("RUM checkpoint 无法严格套用到当前 MODEL。" + "\n".join(details))

    patches = {
        key: ("set", (tensor,))
        for key, tensor in converted.items()
        if key in model_keys and tuple(patched.get_model_object(key).shape) == tuple(tensor.shape)
    }
    if not patches:
        raise ValueError("没有任何 RUM 权重能套用到当前 MODEL；请确认底模是 FLUX.2-Klein 4B。")
    patched.add_patches(patches, strength_patch=1.0, strength_model=1.0)
    return patched, len(patches), str(checkpoint)


def _adapt_converted_keys_to_model(
    converted: dict[str, torch.Tensor],
    model_keys: set[str],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, tensor in converted.items():
        if key in model_keys:
            output[key] = tensor
            continue

        alternative = _alternate_norm_key(key)
        if alternative is not None and alternative in model_keys:
            output[alternative] = tensor
            continue

        output[key] = tensor
    return output


def _alternate_norm_key(key: str) -> str | None:
    suffixes = (
        ".img_attn.norm.query_norm.",
        ".img_attn.norm.key_norm.",
        ".txt_attn.norm.query_norm.",
        ".txt_attn.norm.key_norm.",
        ".norm.query_norm.",
        ".norm.key_norm.",
    )
    if not any(suffix in key for suffix in suffixes):
        return None
    if key.endswith(".weight"):
        return f"{key[:-len('.weight')]}.scale"
    if key.endswith(".scale"):
        return f"{key[:-len('.scale')]}.weight"
    return None


def convert_rum_diffusers_to_comfy(state_dict: dict[str, torch.Tensor], output_prefix: str = "") -> dict[str, torch.Tensor]:
    keys = set(state_dict)
    hidden_size = _infer_hidden_size(state_dict)
    double_blocks = _count_blocks(keys, "transformer_blocks")
    single_blocks = _count_blocks(keys, "single_transformer_blocks")
    if double_blocks == 0 or single_blocks == 0:
        raise ValueError("RUM checkpoint 里找不到 transformer_blocks / single_transformer_blocks。")

    key_map = _flux2_diffusers_to_comfy_key_map(
        double_blocks=double_blocks,
        single_blocks=single_blocks,
        hidden_size=hidden_size,
        output_prefix=output_prefix,
    )
    output: dict[str, torch.Tensor] = {}
    for source_key, target in key_map.items():
        if source_key not in state_dict:
            continue
        weight = state_dict[source_key]
        if isinstance(target, str):
            output[target] = weight
            continue

        target_key, offset, transform = target
        weight = transform(weight)
        if offset is None:
            output[target_key] = weight
            continue

        old_weight = output.get(target_key)
        if old_weight is None:
            shape = list(weight.shape)
            shape[offset[0]] = offset[1] + offset[2]
            old_weight = torch.zeros(shape, dtype=weight.dtype, device=weight.device)
        elif old_weight.shape[offset[0]] < offset[1] + offset[2]:
            shape = list(old_weight.shape)
            shape[offset[0]] = offset[1] + offset[2]
            expanded = torch.zeros(shape, dtype=old_weight.dtype, device=old_weight.device)
            slices = [slice(None)] * old_weight.ndim
            slices[offset[0]] = slice(0, old_weight.shape[offset[0]])
            expanded[tuple(slices)] = old_weight
            old_weight = expanded

        old_weight.narrow(offset[0], offset[1], offset[2]).copy_(weight)
        output[target_key] = old_weight
    return output


def _flux2_diffusers_to_comfy_key_map(
    *,
    double_blocks: int,
    single_blocks: int,
    hidden_size: int,
    output_prefix: str,
):
    identity = lambda tensor: tensor
    key_map = {}

    for index in range(double_blocks):
        source = f"transformer_blocks.{index}"
        target = f"{output_prefix}double_blocks.{index}"
        for suffix in ("weight",):
            img_qkv = f"{target}.img_attn.qkv.{suffix}"
            key_map[f"{source}.attn.to_q.{suffix}"] = (img_qkv, (0, 0, hidden_size), identity)
            key_map[f"{source}.attn.to_k.{suffix}"] = (img_qkv, (0, hidden_size, hidden_size), identity)
            key_map[f"{source}.attn.to_v.{suffix}"] = (img_qkv, (0, hidden_size * 2, hidden_size), identity)

            txt_qkv = f"{target}.txt_attn.qkv.{suffix}"
            key_map[f"{source}.attn.add_q_proj.{suffix}"] = (txt_qkv, (0, 0, hidden_size), identity)
            key_map[f"{source}.attn.add_k_proj.{suffix}"] = (txt_qkv, (0, hidden_size, hidden_size), identity)
            key_map[f"{source}.attn.add_v_proj.{suffix}"] = (txt_qkv, (0, hidden_size * 2, hidden_size), identity)

        block_map = {
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "ff.linear_in.weight": "img_mlp.0.weight",
            "ff.linear_out.weight": "img_mlp.2.weight",
            "ff_context.linear_in.weight": "txt_mlp.0.weight",
            "ff_context.linear_out.weight": "txt_mlp.2.weight",
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        }
        for source_suffix, target_suffix in block_map.items():
            key_map[f"{source}.{source_suffix}"] = f"{target}.{target_suffix}"

    for index in range(single_blocks):
        source = f"single_transformer_blocks.{index}"
        target = f"{output_prefix}single_blocks.{index}"
        block_map = {
            "attn.to_qkv_mlp_proj.weight": "linear1.weight",
            "attn.to_out.weight": "linear2.weight",
            "attn.norm_q.weight": "norm.query_norm.scale",
            "attn.norm_k.weight": "norm.key_norm.scale",
        }
        for source_suffix, target_suffix in block_map.items():
            key_map[f"{source}.{source_suffix}"] = f"{target}.{target_suffix}"

    basic_map = {
        "x_embedder.weight": "img_in.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "context_embedder.weight": "txt_in.weight",
        "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
        "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
        "proj_out.weight": "final_layer.linear.weight",
    }
    for source_key, target_key in basic_map.items():
        key_map[source_key] = f"{output_prefix}{target_key}"
    key_map["norm_out.linear.weight"] = (
        f"{output_prefix}final_layer.adaLN_modulation.1.weight",
        None,
        _swap_scale_shift,
    )
    return key_map


def _fit_last_dim(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.shape[-1] == width:
        return tensor
    if tensor.shape[-1] > width:
        return tensor[..., :width]
    return F.pad(tensor, (0, width - tensor.shape[-1]))


def _swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    return torch.cat([scale, shift], dim=0)


def _count_blocks(keys: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indexes = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return max(indexes) + 1 if indexes else 0


def _infer_hidden_size(state_dict: dict[str, torch.Tensor]) -> int:
    if "x_embedder.weight" in state_dict:
        return int(state_dict["x_embedder.weight"].shape[0])
    if "context_embedder.weight" in state_dict:
        return int(state_dict["context_embedder.weight"].shape[0])
    raise ValueError("无法从 RUM checkpoint 推断 hidden_size。")


def _require_key(state_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise ValueError(f"RUM checkpoint 缺少必要权重：{key}")
    return state_dict[key]


def _resolve_checkpoint_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"找不到 RUM checkpoint：{raw_path}")
    candidates = sorted(path.rglob("*.safetensors"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"{raw_path} 不是明确的 safetensors 文件，请直接选择/填写 RUM checkpoint 文件。")
