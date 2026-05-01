from __future__ import annotations

from pathlib import Path


NODE_DIR = Path(__file__).resolve().parent


def _diffusion_model_list():
    try:
        import folder_paths

        names = folder_paths.get_filename_list("diffusion_models")
        return names if names else ["put_rum_checkpoint_in_models_diffusion_models.safetensors"]
    except Exception:
        return ["put_rum_checkpoint_in_models_diffusion_models.safetensors"]


class RUMFlux2LoadNativeModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rum_checkpoint_name": (_diffusion_model_list(),),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写绝对路径时会覆盖 rum_checkpoint_name。推荐直接选择 BF16 RUM checkpoint。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load_model"
    CATEGORY = "RUM/native"

    def load_model(self, rum_checkpoint_name: str, base_text_tokens: int, rum_checkpoint_path: str = ""):
        import folder_paths

        from .rum_native import load_rum_native_model

        checkpoint = rum_checkpoint_path.strip()
        if not checkpoint:
            checkpoint = folder_paths.get_full_path_or_raise("diffusion_models", rum_checkpoint_name)

        model, count, resolved = load_rum_native_model(checkpoint, base_text_tokens=base_text_tokens)
        return (model, f"RUM native BF16 MODEL 已加载：{count} 个权重；base={base_text_tokens}；checkpoint={resolved}")


class RUMFlux2ApplyModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rum_checkpoint_name": (_diffusion_model_list(),),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "strict": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "rum_checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写绝对路径时会覆盖 rum_checkpoint_name。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply_patch"
    CATEGORY = "RUM/native"

    def apply_patch(self, model, rum_checkpoint_name: str, base_text_tokens: int, strict: bool, rum_checkpoint_path: str = ""):
        import folder_paths

        from .rum_native import apply_rum_model_patch

        checkpoint = rum_checkpoint_path.strip()
        if not checkpoint:
            checkpoint = folder_paths.get_full_path_or_raise("diffusion_models", rum_checkpoint_name)

        patched, count, resolved = apply_rum_model_patch(
            model,
            checkpoint,
            base_text_tokens=base_text_tokens,
            strict=strict,
        )
        return (patched, f"RUM native patch 已套用：{count} 个权重；checkpoint={resolved}")


class RUMFlux2CombineConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux2_conditioning": ("CONDITIONING",),
                "sdxl_conditioning": ("CONDITIONING",),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "use_guidance_embedding": ("BOOLEAN", {"default": False}),
                "base_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "sdxl_clip_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "use_sdxl_extra": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "RUM/native"

    def combine(
        self,
        flux2_conditioning,
        sdxl_conditioning,
        guidance: float,
        use_guidance_embedding: bool,
        base_text_tokens: int,
        extra_text_tokens: int,
        sdxl_clip_width: int,
        use_sdxl_extra: bool,
    ):
        from .rum_native import combine_rum_conditioning

        return (
            combine_rum_conditioning(
                flux2_conditioning,
                sdxl_conditioning,
                guidance=guidance if use_guidance_embedding else None,
                base_text_tokens=base_text_tokens,
                extra_text_tokens=extra_text_tokens,
                sdxl_clip_width=sdxl_clip_width,
                use_sdxl_extra=use_sdxl_extra,
            ),
        )



class RUMDiffusersExactTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "1girl, kisaki (blue archive), eating baozi, sitting, indoors"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "flux2_model_path": (
                    "STRING",
                    {
                        "default": str(NODE_DIR / "models" / "FLUX.2-klein-base-4B"),
                        "multiline": False,
                        "tooltip": "FLUX.2-klein diffusers 目录，里面要有 model_index.json / text_encoder / tokenizer。",
                    },
                ),
                "sdxl_text_model_path": (
                    "STRING",
                    {
                        "default": str(NODE_DIR / "models" / "waiIllustriousSDXL_v160_text"),
                        "multiline": False,
                        "tooltip": "SDXL text-only diffusers 目录，里面要有 tokenizer/tokenizer_2/text_encoder/text_encoder_2。",
                    },
                ),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "use_guidance_embedding": ("BOOLEAN", {"default": False}),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
                "negative_text_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "positive_qwen_layers": ("STRING", {"default": "10,20,30", "multiline": False}),
                "negative_qwen_layers": ("STRING", {"default": "9,18,27", "multiline": False}),
                "device": (["cuda", "cpu", "auto"], {"default": "cuda"}),
                "qwen_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "sdxl_dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
                "unload_after_encode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "status")
    FUNCTION = "encode"
    CATEGORY = "RUM/native"

    def encode(
        self,
        prompt: str,
        negative_prompt: str,
        flux2_model_path: str,
        sdxl_text_model_path: str,
        guidance: float,
        use_guidance_embedding: bool,
        base_text_tokens: int,
        negative_text_tokens: int,
        extra_text_tokens: int,
        positive_qwen_layers: str,
        negative_qwen_layers: str,
        device: str,
        qwen_dtype: str,
        sdxl_dtype: str,
        unload_after_encode: bool,
    ):
        from .rum_native import encode_diffusers_exact_rum_text

        positive, negative = encode_diffusers_exact_rum_text(
            prompt=prompt,
            negative_prompt=negative_prompt,
            flux2_model_path=flux2_model_path.strip(),
            sdxl_text_model_path=sdxl_text_model_path.strip(),
            device=device,
            qwen_dtype=qwen_dtype,
            sdxl_dtype=sdxl_dtype,
            unload_after_encode=unload_after_encode,
            guidance=guidance if use_guidance_embedding else None,
            base_text_tokens=base_text_tokens,
            negative_text_tokens=negative_text_tokens,
            extra_text_tokens=extra_text_tokens,
            positive_qwen_layers=positive_qwen_layers,
            negative_qwen_layers=negative_qwen_layers,
        )
        return (
            positive,
            negative,
            f"Diffusers exact text 已编码：pos={base_text_tokens}+{extra_text_tokens}, neg={negative_text_tokens}",
        )


class RUMSDXLDiffusersTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "1girl, kisaki (blue archive), eating baozi, sitting, indoors"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "RUM/native"

    def encode(self, clip, text: str):
        tokens = clip.tokenize(text, disable_weights=True)
        return (clip.encode_from_tokens_scheduled(tokens),)


class RUMFlux2SetQwenLayers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "layers": ("STRING", {"default": "10,20,30", "multiline": False}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "status")
    FUNCTION = "set_layers"
    CATEGORY = "RUM/native"

    @classmethod
    def IS_CHANGED(cls, clip, layers: str):
        return float("nan")

    def set_layers(self, clip, layers: str):
        parsed = []
        for part in layers.split(","):
            part = part.strip()
            if part:
                try:
                    parsed.append(int(part))
                except ValueError as exc:
                    raise ValueError("RUM Qwen layers 只能填写整数，例如 10,20,30。") from exc
        if len(parsed) != 3:
            raise ValueError("RUM Qwen layers 必须正好是 3 个整数，例如 10,20,30。")

        patched = clip.clone()
        cond_stage_model = getattr(patched, "cond_stage_model", None)
        clip_attr = getattr(cond_stage_model, "clip", None)
        text_model = getattr(cond_stage_model, clip_attr, None) if clip_attr else None
        if text_model is None or not hasattr(text_model, "layer"):
            raise ValueError("当前 CLIP 不是 FLUX.2/Klein Qwen text encoder，不能设置 Qwen 层号。")

        text_model.layer = parsed
        text_model.layer_idx = None
        if hasattr(text_model, "options_default"):
            projected_pooled = text_model.options_default[2] if len(text_model.options_default) > 2 else True
            text_model.options_default = (parsed, None, projected_pooled)
        patched.layer_idx = None
        return (patched, f"RUM Qwen layers 已设置：{','.join(str(x) for x in parsed)}")


class RUMFlux2DiffusersNoiseSource:
    def __init__(self, seed: int, dtype: str):
        self.seed = int(seed)
        self.dtype = dtype

    def generate_noise(self, input_latent):
        import torch
        import comfy.nested_tensor

        latent_image = input_latent["samples"]
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        noise_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        def make_noise(shape, output_dtype, layout):
            noise = torch.randn(shape, generator=generator, device="cpu", dtype=noise_dtype, layout=layout)
            return noise.to(dtype=output_dtype)

        if latent_image.is_nested:
            return comfy.nested_tensor.NestedTensor(
                [make_noise(tensor.shape, tensor.dtype, tensor.layout) for tensor in latent_image.unbind()]
            )
        return make_noise(latent_image.shape, latent_image.dtype, latent_image.layout)


class RUMFlux2DiffusersNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "get_noise"
    CATEGORY = "RUM/native"

    def get_noise(self, noise_seed: int, dtype: str):
        return (RUMFlux2DiffusersNoiseSource(noise_seed, dtype),)


class RUMFlux2DiffusersEulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "RUM/native"

    def get_sampler(self):
        from .rum_native import create_diffusers_euler_sampler

        return (create_diffusers_euler_sampler(),)


class RUMFlux2DiffusersScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 960, "min": 16, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "RUM/native"

    def get_sigmas(self, steps: int, width: int, height: int):
        from .rum_native import diffusers_flux2_sigmas

        return (diffusers_flux2_sigmas(steps=steps, width=width, height=height),)


class RUMFlux2DiffusersExactVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "diffusers_vae_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选：填写 FLUX.2-klein-base-4B diffusers 目录时，使用原始 diffusers VAE + PIL 量化路径，可对齐 reference PNG。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "RUM/native"

    def decode(self, samples, vae, diffusers_vae_model_path: str = ""):
        from .rum_native import decode_diffusers_exact_flux2_latent

        return (decode_diffusers_exact_flux2_latent(vae, samples, flux2_model_path=diffusers_vae_model_path),)


class RUMFlux2DiffusersCFGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "RUM/native"

    def get_guider(self, model, positive, negative, cfg: float):
        from .rum_native import create_diffusers_cfg_guider

        return (create_diffusers_cfg_guider(model, positive, negative, cfg),)


class RUMFlux2DiffusersMatchModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_text_tokens": ("INT", {"default": 200, "min": 1, "max": 4096, "step": 1}),
                "extra_text_tokens": ("INT", {"default": 77, "min": 1, "max": 512, "step": 1}),
                "disable_guidance": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply_patch"
    CATEGORY = "RUM/native"

    def apply_patch(self, model, base_text_tokens: int, extra_text_tokens: int, disable_guidance: bool):
        from .rum_native import apply_diffusers_match_model_wrapper

        patched = apply_diffusers_match_model_wrapper(
            model,
            base_text_tokens=base_text_tokens,
            extra_text_tokens=extra_text_tokens,
            disable_guidance=disable_guidance,
        )
        guidance_status = "disabled" if disable_guidance else "unchanged"
        return (
            patched,
            f"Diffusers-match token policy 已启用：base={base_text_tokens}, extra={extra_text_tokens}, guidance={guidance_status}",
        )


NODE_CLASS_MAPPINGS = {
    "RUMFlux2LoadNativeModel": RUMFlux2LoadNativeModel,
    "RUMFlux2ApplyModelPatch": RUMFlux2ApplyModelPatch,
    "RUMFlux2CombineConditioning": RUMFlux2CombineConditioning,
    "RUMDiffusersExactTextEncode": RUMDiffusersExactTextEncode,
    "RUMSDXLDiffusersTextEncode": RUMSDXLDiffusersTextEncode,
    "RUMFlux2SetQwenLayers": RUMFlux2SetQwenLayers,
    "RUMFlux2DiffusersMatchModelPatch": RUMFlux2DiffusersMatchModelPatch,
    "RUMFlux2DiffusersNoise": RUMFlux2DiffusersNoise,
    "RUMFlux2DiffusersEulerSampler": RUMFlux2DiffusersEulerSampler,
    "RUMFlux2DiffusersScheduler": RUMFlux2DiffusersScheduler,
    "RUMFlux2DiffusersCFGuider": RUMFlux2DiffusersCFGuider,
    "RUMFlux2DiffusersExactVAEDecode": RUMFlux2DiffusersExactVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RUMFlux2LoadNativeModel": "RUM FLUX.2 Load Native Model",
    "RUMFlux2ApplyModelPatch": "RUM FLUX.2 Apply Model Patch",
    "RUMFlux2CombineConditioning": "RUM FLUX.2 Combine Conditioning",
    "RUMDiffusersExactTextEncode": "RUM Diffusers Exact Text Encode",
    "RUMSDXLDiffusersTextEncode": "RUM SDXL Diffusers Text Encode",
    "RUMFlux2SetQwenLayers": "RUM FLUX.2 Set Qwen Layers",
    "RUMFlux2DiffusersMatchModelPatch": "RUM FLUX.2 Diffusers Match Model Patch",
    "RUMFlux2DiffusersNoise": "RUM FLUX.2 Diffusers Noise",
    "RUMFlux2DiffusersEulerSampler": "RUM FLUX.2 Diffusers Euler Sampler",
    "RUMFlux2DiffusersScheduler": "RUM FLUX.2 Diffusers Scheduler",
    "RUMFlux2DiffusersCFGuider": "RUM FLUX.2 Diffusers CFG Guider",
    "RUMFlux2DiffusersExactVAEDecode": "RUM FLUX.2 Diffusers Exact VAE Decode",
}
