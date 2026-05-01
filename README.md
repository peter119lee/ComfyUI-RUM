# ComfyUI-RUM

ComfyUI-RUM 是 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 ComfyUI native 节点适配。目标是把 RUM 的 FLUX.2-Klein + SDXL teacher CLIP 条件路径拆进 ComfyUI，让它既能作为普通 ComfyUI workflow 使用，也能用于检查 ComfyUI 适配是否接近原始 diffusers 推理。

RUM 的原始想法、模型和参考推理代码来自 RimoChan/RUM。本仓库只提供 ComfyUI 适配代码和 workflow，不包含模型权重。

## 当前状态

- 普通 native workflow 可以跑 RUM，但不承诺和原始 diffusers 输出同图。
- `diffusers-match` API workflow 是验证路径，尽量复刻 RimoChan/RUM 的 diffusers 推理细节。
- 已验证过一组本地 diffusers reference（seed `7478533297787`）可以和 ComfyUI workflow PNG 逐像素 0 差。
- RimoChan/RUM 仓库自带示例图 `output_0.png`（seed `1`）用当前 native diffusers-match workflow 肉眼基本一致，但像素比较仍不是 0 差；最近一次记录约为 `mean_abs=2.7757`、`rmse=10.1886`。这不是最终目标，仍在继续追踪。
- 曾验证过一个直接调用原始 diffusers pipeline 的 `RUMDiffusersExactGenerateAndSave` 实验节点，能让 `output_0.png` 字节级一致，但它要求在 ComfyUI 环境安装并加载 diffusers pipeline，不符合主分支“ComfyUI native custom node”的目标，所以只保留在 `codex/diffusers-exact-reference-node` 分支作为参考，不进 `main`。
- 当前重点继续放在 native 路线；`SaveImage` / VAE postprocess / PNG 量化路径是剩余像素差异的重点排查对象之一。

这段状态很重要：**不要把普通 workflow、seed747 本地 reference、上游 `output_0.png` 三者混成同一个验证目标。**

## 为什么需要专门的 diffusers-match 路径

RUM 不是只给 FLUX.2-Klein 加一组普通 LoRA。它把 FLUX.2 的 Qwen 文本条件和 SDXL teacher CLIP 条件拼到同一个 transformer 条件流里。要贴近原始 diffusers 推理，需要同时处理这些细节：

- Qwen 正面 prompt 使用特定 hidden states 层：`10,20,30`。
- 正面条件长度是 Qwen `200` tokens + SDXL `77` tokens。
- negative prompt 不是同样拼 SDXL，而是 Qwen `512` tokens，并使用层 `9,18,27`。
- SDXL teacher CLIP 来自 `Ine007/waiIllustriousSDXL_v160`，不能随便换成普通 `clip_l.safetensors` / `clip_g.safetensors`。
- 初始 noise 需要复刻 diffusers CPU generator + BF16 行为。
- scheduler sigma 和 timestep rounding 必须和 diffusers 一致。
- RUM transformer 的 FLUX guidance embedding 在 match 路径中需要按原始推理方式处理。
- FLUX.2 VAE decode 需要匹配 diffusers 的 BN、unpatchify、decode 和 PIL/postprocess 量化路径。

这些点少一个都会漂。漂移可能不是小数值误差，而是角色、构图、颜色和细节明显不同。

## 已解决过的关键问题

这些是适配过程中已经确认并修过的问题，保留在 README 里是为了后来的人不要重复踩坑：

1. **Qwen rotary buffer dtype 问题**
   - 问题：把 Qwen text encoder 整体 `.to(dtype=bfloat16)` 会把 `rotary_emb.inv_freq` 也转成 BF16。
   - 结果：text embedding 会偏，后面 denoise 全部跟着偏。
   - 处理：diffusers-match text encode 中，Qwen 通过 `Flux2KleinPipeline` 加载，只移动到目标 device，不强制把 FP32 rotary buffer 转 BF16。

2. **negative conditioning 配置问题**
   - 问题：negative 不能照正面条件拼 `200 + 77`。
   - 正确配置：`negative_text_tokens=512`，`negative_qwen_layers="9,18,27"`。
   - 处理：`RUMDiffusersExactTextEncode` 和 API workflow 默认值按这个配置设置。

3. **scheduler / timestep 问题**
   - 问题：ComfyUI FLUX scheduler 和 diffusers Flux2Klein scheduler 不完全一样。
   - 处理：`RUMFlux2DiffusersScheduler` 使用 diffusers 风格 `np.linspace(1.0, 1.0 / steps)` 加 time shift；模型调用时按 diffusers 的 bfloat16 timestep 逻辑处理。

4. **noise 问题**
   - 问题：普通 ComfyUI `RandomNoise` 不等于 diffusers `randn_tensor(..., generator=torch.Generator(device="cpu"), dtype=bfloat16)`。
   - 处理：`RUMFlux2DiffusersNoise` 单独复刻 diffusers CPU BF16 noise 生成。

5. **VAE decode 问题**
   - 问题：ComfyUI 的 FP32 VAE 文件/解码路径会和 diffusers reference PNG 有差异。
   - 处理：`RUMFlux2DiffusersExactVAEDecode` 可传入原始 `FLUX.2-klein-base-4B` diffusers 目录，走 diffusers FLUX.2 VAE decode 路径。

## Workflow 说明

仓库只保留两个示例 workflow，避免误用。

### 普通 native workflow

```text
examples/basic_workflow_api.json
```

用途：日常 ComfyUI 使用。它走 ComfyUI 常规模型、CLIP、scheduler、VAE 节点，比较容易接入其他 ComfyUI 工作流。

限制：不保证和 RimoChan/RUM diffusers reference 同图。

### diffusers-match API workflow

```text
examples/diffusers_match_workflow_api.json
```

用途：验证 ComfyUI 适配是否贴近原始 diffusers 推理。

默认参数：

```text
prompt=1girl, kisaki (blue archive), eating baozi, sitting, indoors
negative_prompt=
seed=1
steps=20
guidance_scale=5
width=960
height=1024
```

重要：不同 ComfyUI 发行版的模型下拉名可能不同。例如 Aki 里可能显示成 `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`。公开 workflow 不能写死每个人本地的模型分类路径；如果 API validation 失败，请把 `rum_checkpoint_name` 改成你本机 `/object_info/RUMFlux2LoadNativeModel` 里实际列出的名字。

## 模型文件

推荐位置：

| 用途 | 推荐文件 | ComfyUI 位置 |
| --- | --- | --- |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models/` |
| FLUX.2-Klein base | `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/flux-2-klein/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL CLIP-L | `clip_l.safetensors` | `models/text_encoders/` |
| SDXL CLIP-G | `clip_g.safetensors` | `models/text_encoders/` |
| diffusers-match teacher CLIP-L | `waiIllustriousSDXL_v160_clip_l.safetensors` | `models/text_encoders/` |
| diffusers-match teacher CLIP-G | `waiIllustriousSDXL_v160_clip_g.safetensors` | `models/text_encoders/` |

`diffusers-match` 还需要一个原始 `FLUX.2-klein-base-4B` diffusers 目录，用于 exact text encode 和 exact VAE decode。这个目录不是普通 ComfyUI `.safetensors` 模型文件。

如果你已经有 `waiIllustriousSDXL_v160_text` diffusers text-only 目录，可以用脚本安装 teacher CLIP：

```bash
python scripts/install_diffusers_teacher_clip.py /path/to/waiIllustriousSDXL_v160_text
```

也可以使用下载辅助脚本：

```bash
python scripts/download_models.py --all
```

## 安装和检查

把仓库放到 ComfyUI 的 custom nodes：

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

安装依赖后重启 ComfyUI：

```bash
pip install -r requirements.txt
```

检查节点和模型可见性：

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

提交 API workflow：

```bash
python scripts/queue_workflow.py examples/diffusers_match_workflow_api.json --server http://127.0.0.1:8188
```

## 数值验证建议

不要只靠看图判断适配是否正确。建议至少比较：

- text embedding：positive Qwen、positive combined、negative Qwen。
- initial noise：CPU generator、dtype、shape。
- scheduler sigmas 和每步 timestep。
- transformer 每步 raw noise / CFG noise / latent after step。
- final latent。
- VAE decode 后 RGB tensor。
- 最终 PNG 像素指标：`pixel_equal`、`max_abs`、`mean_abs`、`rmse`。

如果 PNG 不一致，先找第一处 tensor 非零差异。不要先调 prompt，也不要靠肉眼猜。

## 常见问题

### API workflow 报模型名不在列表

ComfyUI 的模型下拉名来自本机文件扫描，不同发行版会带不同子目录分类。解决方法：

1. 打开 `http://127.0.0.1:8188/object_info/RUMFlux2LoadNativeModel`。
2. 找到 `rum_checkpoint_name` 列表里你本机真实的 RUM checkpoint 名字。
3. 把 workflow 里的 `rum_checkpoint_name` 改成那个完整名字。

### 普通 workflow 和 diffusers-match workflow 该用哪个

- 要日常出图：用 `basic_workflow_api.json`。
- 要验证适配正确性：用 `diffusers_match_workflow_api.json`。
- 要和 RimoChan/RUM 的某张 reference 图对齐：必须先确认 reference 的 prompt、seed、模型文件、teacher CLIP、diffusers 版本和保存路径。

### 为什么同样 prompt/seed 还是不同

常见原因：

- 用了普通 native workflow，而不是 diffusers-match workflow。
- teacher CLIP 文件不一致。
- Qwen 层号或 token 长度不一致。
- negative prompt 路径不一致。
- noise dtype 或 generator device 不一致。
- scheduler 和 timestep rounding 不一致。
- VAE decode 不是 diffusers exact 路径。
- 模型文件不是同一个权重，或者同源但精度不同。

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM)：RUM 的原始项目、模型、训练和 diffusers 推理参考。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：ComfyUI 节点系统和执行环境。
- [Hugging Face diffusers](https://github.com/huggingface/diffusers)：FLUX.2-Klein pipeline、scheduler、VAE decode 参考实现。

## License note

本仓库不包含模型权重。RUM 上游项目在本适配开始时没有明确 LICENSE；模型和代码的再分发请遵守各自上游规则。

---

# ComfyUI-RUM English README

ComfyUI-RUM is a native ComfyUI node adapter for [RimoChan/RUM](https://github.com/RimoChan/RUM). It brings the RUM FLUX.2-Klein + SDXL teacher CLIP conditioning path into ComfyUI. The goal is to support normal ComfyUI workflows while also providing a stricter validation path for comparing against the original diffusers inference behavior.

The original RUM idea, model, training work, and reference inference code come from RimoChan/RUM. This repository only provides ComfyUI adapter code and example workflows. It does not include model weights.

## Current Status

- The normal native workflow can run RUM, but it does not promise image identity with the original diffusers output.
- The `diffusers-match` API workflow is the validation path. It tries to reproduce the RimoChan/RUM diffusers inference details inside ComfyUI.
- One local diffusers reference case, seed `7478533297787`, has been verified as pixel-identical with the ComfyUI workflow output.
- The upstream repository example image `output_0.png`, seed `1`, is not yet pixel-identical with the ComfyUI workflow. The current measured difference is about `mean_abs=2.7757`, `rmse=10.1886`.

Do not mix these targets together: the normal workflow, the seed `7478533297787` local reference, and upstream `output_0.png` are different validation cases.

## Why diffusers-match Exists

RUM is not a simple LoRA on top of FLUX.2-Klein. It combines FLUX.2 Qwen text conditioning with SDXL teacher CLIP conditioning in the transformer condition stream. Matching the original diffusers inference requires several details to line up:

- Positive Qwen prompt uses hidden-state layers `10,20,30`.
- Positive conditioning uses Qwen `200` tokens plus SDXL `77` tokens.
- Negative conditioning uses Qwen `512` tokens and layers `9,18,27`; it does not append the SDXL branch in the same way as the positive path.
- SDXL teacher CLIP comes from `Ine007/waiIllustriousSDXL_v160`; generic `clip_l.safetensors` and `clip_g.safetensors` are not equivalent.
- Initial noise must match diffusers CPU generator + BF16 behavior.
- Scheduler sigmas and timestep rounding must match diffusers.
- FLUX guidance embedding behavior must follow the original inference path in the match workflow.
- FLUX.2 VAE decode must match diffusers BN, unpatchify, decode, and PIL/postprocess quantization.

If one of these details is wrong, the result can drift across sampling steps. The drift can affect character identity, composition, colors, and fine details.

## Bugs Already Found and Fixed

These notes are kept here so future debugging does not repeat the same mistakes.

1. **Qwen rotary buffer dtype**
   - Problem: converting the Qwen text encoder with `.to(dtype=bfloat16)` also converts `rotary_emb.inv_freq` to BF16.
   - Effect: text embeddings drift, then the whole denoise path drifts.
   - Fix: in the diffusers-match text encoder path, Qwen is loaded through `Flux2KleinPipeline` and moved to the target device without forcing FP32 rotary buffers into BF16.

2. **Negative conditioning configuration**
   - Problem: negative conditioning should not mirror the positive `200 + 77` token layout.
   - Correct settings: `negative_text_tokens=512`, `negative_qwen_layers="9,18,27"`.
   - Fix: `RUMDiffusersExactTextEncode` and the API workflow use these defaults.

3. **Scheduler and timestep behavior**
   - Problem: ComfyUI's normal FLUX scheduler is not identical to the diffusers Flux2Klein scheduler.
   - Fix: `RUMFlux2DiffusersScheduler` uses diffusers-style `np.linspace(1.0, 1.0 / steps)` plus time shift, and the model call follows diffusers bfloat16 timestep behavior.

4. **Initial noise**
   - Problem: ComfyUI `RandomNoise` is not identical to diffusers `randn_tensor(..., generator=torch.Generator(device="cpu"), dtype=bfloat16)`.
   - Fix: `RUMFlux2DiffusersNoise` reproduces the diffusers CPU BF16 noise path.

5. **VAE decode path**
   - Problem: ComfyUI's FP32 VAE file/decode path can differ from the diffusers reference PNG.
   - Fix: `RUMFlux2DiffusersExactVAEDecode` can point to the original `FLUX.2-klein-base-4B` diffusers directory and use the diffusers FLUX.2 VAE decode path.

## Workflows

Only two public example workflows are kept to reduce confusion.

### Normal Native Workflow

```text
examples/basic_workflow_api.json
```

Use this for regular ComfyUI generation. It uses normal ComfyUI model loading, CLIP, scheduler, and VAE nodes.

Limitation: it does not guarantee image identity with RimoChan/RUM diffusers references.

### diffusers-match API Workflow

```text
examples/diffusers_match_workflow_api.json
```

Use this for validation against the original diffusers inference behavior.

Default parameters:

```text
prompt=1girl, kisaki (blue archive), eating baozi, sitting, indoors
negative_prompt=
seed=1
steps=20
guidance_scale=5
width=960
height=1024
```

Different ComfyUI distributions may expose different model names in the dropdown. For example, Aki may list the model as `Unknown\no tags\rum-flux2-klein-4b-preview.safetensors`. The public workflow cannot hard-code every local folder category. If API validation fails, read `/object_info/RUMFlux2LoadNativeModel` and set `rum_checkpoint_name` to the exact local name shown there.

## Model Files

Recommended locations:

| Purpose | Recommended file | ComfyUI location |
| --- | --- | --- |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models/` |
| FLUX.2-Klein base | `flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models/flux-2-klein/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL CLIP-L | `clip_l.safetensors` | `models/text_encoders/` |
| SDXL CLIP-G | `clip_g.safetensors` | `models/text_encoders/` |
| diffusers-match teacher CLIP-L | `waiIllustriousSDXL_v160_clip_l.safetensors` | `models/text_encoders/` |
| diffusers-match teacher CLIP-G | `waiIllustriousSDXL_v160_clip_g.safetensors` | `models/text_encoders/` |

The `diffusers-match` path also needs the original `FLUX.2-klein-base-4B` diffusers directory for exact text encoding and exact VAE decode. This directory is not a normal ComfyUI `.safetensors` model file.

If you already have a `waiIllustriousSDXL_v160_text` diffusers text-only directory, install the teacher CLIP files with:

```bash
python scripts/install_diffusers_teacher_clip.py /path/to/waiIllustriousSDXL_v160_text
```

You can also use the helper downloader:

```bash
python scripts/download_models.py --all
```

## Install and Check

Put this repository under:

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

Install dependencies and restart ComfyUI:

```bash
pip install -r requirements.txt
```

Check node import and model visibility:

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

Queue an API workflow:

```bash
python scripts/queue_workflow.py examples/diffusers_match_workflow_api.json --server http://127.0.0.1:8188
```

## Numeric Validation

Do not rely on visual inspection only. For alignment work, compare at least:

- text embeddings: positive Qwen, combined positive, negative Qwen.
- initial noise: CPU generator, dtype, shape.
- scheduler sigmas and per-step timestep.
- transformer raw noise, CFG noise, and latent after each step.
- final latent.
- RGB tensor after VAE decode.
- final PNG metrics: `pixel_equal`, `max_abs`, `mean_abs`, `rmse`.

When PNGs differ, find the first tensor stage with a non-zero difference before changing prompts or judging by eye.

## FAQ

### API workflow says the model name is not in the list

ComfyUI model names come from local file scanning. Different packages may add different subfolder categories. Fix:

1. Open `http://127.0.0.1:8188/object_info/RUMFlux2LoadNativeModel`.
2. Find the exact local RUM checkpoint name in `rum_checkpoint_name`.
3. Put that full name into the workflow.

### Which workflow should I use?

- For normal image generation: `basic_workflow_api.json`.
- For adapter validation: `diffusers_match_workflow_api.json`.
- For matching a specific RimoChan/RUM reference image: first confirm the exact prompt, seed, model files, teacher CLIP, diffusers version, and save path.

### Why can the same prompt and seed still differ?

Common causes:

- The normal native workflow was used instead of the diffusers-match workflow.
- Teacher CLIP files differ.
- Qwen layers or token lengths differ.
- Negative prompt path differs.
- Noise dtype or generator device differs.
- Scheduler or timestep rounding differs.
- VAE decode is not using the diffusers exact path.
- Model weights are not identical, or are from the same source but different precision.

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM): original RUM project, model, training, and diffusers inference reference.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI): node system and execution environment.
- [Hugging Face diffusers](https://github.com/huggingface/diffusers): FLUX.2-Klein pipeline, scheduler, and VAE decode reference implementation.

## License Note

This repository does not include model weights. At the time this adapter work started, upstream RUM did not provide a clear LICENSE. Follow upstream rules for model and code redistribution.
