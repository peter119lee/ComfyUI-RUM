from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


_DIFFUSION_PREFIX = "diffusion_model."


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
    guidance: float,
    base_text_tokens: int,
    extra_text_tokens: int,
    sdxl_clip_width: int,
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
        sdxl_extra = sdxl_embeds.to(device=flux_embeds.device, dtype=flux_embeds.dtype)
        sdxl_extra = sdxl_extra[:, :extra_text_tokens]
        sdxl_extra = _fit_last_dim(sdxl_extra, sdxl_clip_width)
        sdxl_extra = _fit_last_dim(sdxl_extra, flux_base.shape[-1])

        combined = torch.cat([flux_base, sdxl_extra], dim=1)
        meta = flux_meta.copy()
        meta["guidance"] = float(guidance)
        meta["rum_base_text_tokens"] = int(flux_base.shape[1])
        meta["rum_extra_text_tokens"] = int(sdxl_extra.shape[1])
        meta.pop("attention_mask", None)
        output.append([combined, meta])
    return output


class RUMDiffusersMatchWrapper:
    def __init__(self, base_model, base_text_tokens: int, extra_text_tokens: int):
        if base_text_tokens <= 0:
            raise ValueError("diffusers match base_text_tokens 必须大于 0。")
        if extra_text_tokens <= 0:
            raise ValueError("diffusers match extra_text_tokens 必须大于 0。")
        self.base_model = base_model
        self.base_text_tokens = int(base_text_tokens)
        self.extra_text_tokens = int(extra_text_tokens)

    def __call__(self, apply_model, params):
        conditions = params["c"].copy()
        cross_attn = conditions.get("c_crossattn")
        if cross_attn is not None:
            total_tokens = self.base_text_tokens + self.extra_text_tokens
            if cross_attn.shape[1] > total_tokens:
                base = cross_attn[:, : self.base_text_tokens]
                extra = cross_attn[:, -self.extra_text_tokens :]
                conditions["c_crossattn"] = torch.cat([base, extra], dim=1)
        return apply_model(params["input"], params["timestep"], **conditions)

    def clone(self):
        return RUMDiffusersMatchWrapper(
            self.base_model,
            base_text_tokens=self.base_text_tokens,
            extra_text_tokens=self.extra_text_tokens,
        )


def apply_diffusers_match_model_wrapper(model, *, base_text_tokens: int, extra_text_tokens: int):
    patched = model.clone()
    patched.set_model_unet_function_wrapper(
        RUMDiffusersMatchWrapper(
            patched,
            base_text_tokens=base_text_tokens,
            extra_text_tokens=extra_text_tokens,
        )
    )
    return patched


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
            "attn.norm_q.weight": "img_attn.norm.query_norm.weight",
            "attn.norm_k.weight": "img_attn.norm.key_norm.weight",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.weight",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.weight",
        }
        for source_suffix, target_suffix in block_map.items():
            key_map[f"{source}.{source_suffix}"] = f"{target}.{target_suffix}"

    for index in range(single_blocks):
        source = f"single_transformer_blocks.{index}"
        target = f"{output_prefix}single_blocks.{index}"
        block_map = {
            "attn.to_qkv_mlp_proj.weight": "linear1.weight",
            "attn.to_out.weight": "linear2.weight",
            "attn.norm_q.weight": "norm.query_norm.weight",
            "attn.norm_k.weight": "norm.key_norm.weight",
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
