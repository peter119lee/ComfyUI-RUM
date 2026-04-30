from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


NODE_DIR = Path(__file__).resolve().parent


def _load_local_config() -> dict[str, str]:
    config_path = NODE_DIR / "local_config.json"
    if not config_path.exists():
        return {}
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}") from exc
    return {str(key): str(value) for key, value in raw_config.items() if value}


LOCAL_CONFIG = _load_local_config()


def _default_path(config_key: str, env_key: str, fallback: str) -> str:
    for value in (LOCAL_CONFIG.get(config_key), os.getenv(env_key)):
        if value:
            return value
    return str(NODE_DIR / fallback)


DEFAULT_BASE_MODEL_PATH = _default_path(
    "base_model_path",
    "RUM_BASE_MODEL_PATH",
    "models/FLUX.2-klein-base-4B",
)
DEFAULT_RUM_CHECKPOINT_PATH = _default_path(
    "rum_checkpoint_path",
    "RUM_CHECKPOINT_PATH",
    "models/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors",
)
DEFAULT_SDXL_MODEL_PATH = _default_path(
    "sdxl_model_path",
    "RUM_SDXL_MODEL_PATH",
    "models/waiIllustriousSDXL_v160_text",
)


def _parse_layers(raw: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("text_encoder_out_layers 必须是逗号分隔的整数，例如 10,20,30。") from exc
    if not layers:
        raise ValueError("text_encoder_out_layers 不能为空。")
    return layers


def _pil_images_to_comfy(images: Iterable[Image.Image]) -> torch.Tensor:
    tensors = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array))
    if not tensors:
        raise ValueError("RUM pipeline 没有返回图片。")
    return torch.stack(tensors, dim=0)


class RUMFlux2KleinLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_MODEL_PATH,
                        "multiline": False,
                    },
                ),
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": DEFAULT_RUM_CHECKPOINT_PATH,
                        "multiline": False,
                    },
                ),
                "sdxl_model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_SDXL_MODEL_PATH,
                        "multiline": False,
                    },
                ),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "sdxl_text_device": (["cpu", "cuda", "same_as_pipeline"], {"default": "cpu"}),
                "unload_sdxl_unet_vae": ("BOOLEAN", {"default": True}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("RUM_FLUX2_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "RUM/loaders"

    def load(
        self,
        base_model_path: str,
        rum_checkpoint_path: str,
        sdxl_model_path: str,
        dtype: str,
        device: str,
        sdxl_text_device: str,
        unload_sdxl_unet_vae: bool,
        force_reload: bool,
    ):
        from .rum_diffusers import load_rum_pipeline

        handle = load_rum_pipeline(
            base_model_path=base_model_path,
            rum_checkpoint_path=rum_checkpoint_path,
            sdxl_model_path=sdxl_model_path,
            dtype_name=dtype,
            device=device,
            sdxl_text_device=sdxl_text_device,
            unload_sdxl_unet_vae=unload_sdxl_unet_vae,
            force_reload=force_reload,
        )
        return (handle,)


class RUMFlux2KleinSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("RUM_FLUX2_PIPELINE", {"forceInput": True}),
                "prompt": (
                    "STRING",
                    {
                        "default": "1girl, kisaki (blue archive), eating baozi, sitting, indoors",
                        "multiline": True,
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "max_sequence_length": ("INT", {"default": 200, "min": 16, "max": 512, "step": 8}),
                "text_encoder_out_layers": (
                    "STRING",
                    {"default": "10,20,30", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "RUM/sampling"

    def sample(
        self,
        pipeline,
        prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
        num_images: int,
        max_sequence_length: int,
        text_encoder_out_layers: str,
    ):
        layers = _parse_layers(text_encoder_out_layers)
        images = pipeline.generate(
            prompt=prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images=num_images,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=layers,
        )
        return (_pil_images_to_comfy(images),)


class RUMUnloadModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "RUM/utils"

    def unload(self):
        from .rum_diffusers import clear_pipeline_cache

        count = clear_pipeline_cache()
        return (f"已卸载 {count} 个 RUM pipeline 缓存。",)


NODE_CLASS_MAPPINGS = {
    "RUMFlux2KleinLoader": RUMFlux2KleinLoader,
    "RUMFlux2KleinSampler": RUMFlux2KleinSampler,
    "RUMUnloadModels": RUMUnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RUMFlux2KleinLoader": "RUM FLUX.2-klein Loader",
    "RUMFlux2KleinSampler": "RUM FLUX.2-klein Sampler",
    "RUMUnloadModels": "RUM Unload Models",
}
