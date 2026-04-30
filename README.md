# ComfyUI-RUM

![ComfyUI-RUM preview](assets/rum-preview.png)

这是给 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 **ComfyUI 原生节点适配器**，目标模型是 `RUM-FLUX.2-klein-4B-preview`。RUM 的原始想法、模型和参考推理代码都来自 RimoChan 的项目。

主分支是 **native ComfyUI 版**：模型加载、conditioning、latent、sampler、VAE decode 都尽量走 ComfyUI 自己的系统；旧的 diffusers 一体式版本保留在 [`diffusers-pipeline`](https://github.com/peter119lee/ComfyUI-RUM/tree/diffusers-pipeline) 分支。

> **AI 编写说明**
>
> 这个 ComfyUI 适配仓库的代码与 README 由 Codex 5.5 编写和整理；RUM 原始项目、模型和参考推理代码仍来自 [RimoChan/RUM](https://github.com/RimoChan/RUM)。

> **授权提醒**
>
> 这个项目参考并适配了 RUM 公开推理代码的行为。创建本项目时，上游 `RimoChan/RUM` 没有明确 LICENSE。除非上游补 LICENSE 或明确授权，否则不建议提交到 Comfy Registry 或做大范围公开分发。
>
> 本仓库不包含模型权重；模型文件遵守各自上游授权和访问规则。

## 它能做什么

RUM 的重点是：在 FLUX.2-Klein 之外，额外接入一路 SDXL CLIP 文本条件。普通 ComfyUI 的 FLUX.2 节点不知道怎么把这一路额外条件喂进 RUM transformer，所以这个节点包提供：

- `RUM FLUX.2 Apply Model Patch`：把 RUM 权重和双文本投影层套到普通 ComfyUI FLUX.2-Klein `MODEL` 上。
- `RUM FLUX.2 Combine Conditioning`：把 FLUX.2-Klein 的 Qwen 文本条件和 SDXL CLIP 条件合成 RUM 需要的格式。
- `RUM FLUX.2 Load Native Model`：直接把 RUM diffusers 格式 checkpoint 转成 ComfyUI `MODEL`，主要给 diffusers-match workflow 用。
- `RUM FLUX.2 Diffusers Noise` / `RUM FLUX.2 Diffusers CFG Guider`：只用于尽量贴近旧 diffusers 分支的示例图，不建议普通 workflow 使用。

简单说：普通 native workflow 适合日常 ComfyUI 使用；diffusers-match workflow 只是把旧 diffusers 推理里最关键的几个差异补回来，不保证复现同一张图。

## 为什么 native 不能 100% 复现 diffusers

直接说：**因为这不是同一个执行环境**。公开的 RUM 参考推理路径是 diffusers；ComfyUI native 是把这条路径重新拆进 ComfyUI 的模型、CLIP、sampler、noise、conditioning 系统里跑。只要其中一个细节不同，扩散模型就会越采样越偏，最后角色和构图都可能变。

最关键的差异是：

- **SDXL teacher CLIP 不同**：旧 diffusers 用的是 `Ine007/waiIllustriousSDXL_v160` 的 text encoder；普通 `clip_l.safetensors` / `clip_g.safetensors` 不是同一套语义条件。
- **Qwen hidden states 不同**：旧推理正面 prompt 用 Qwen 的特定层 `10,20,30`；普通 ComfyUI encode 默认不这么做。
- **token 长度不同**：旧推理正面是 Qwen 200 tokens + SDXL 77 tokens；负面又不是同样拼法，而是 Qwen 512 tokens。
- **noise 不同**：旧 diffusers 参考路径用 CPU BF16 风格 noise；ComfyUI 原生 `RandomNoise` 不是完全一样。
- **CFG 分支行为不同**：diffusers pipeline 里 positive / negative 的进入方式和 ComfyUI 标准 `CFGGuider` 不完全一致。
- **浮点和调度细节不同**：dtype、设备、scheduler、patch 顺序、模型缓存都会影响结果。

所以目标只能是：**native 节点可用、结构合理、尽量接近 diffusers；不是保证同 seed 同图**。如果你要“最像旧图”，用 `diffusers-match` workflow；如果你要“最 ComfyUI”，用普通 native workflow。

## 模型应该放哪里

请使用 ComfyUI 标准模型路径，不要把 diffusers 文件夹当成 ComfyUI 模型直接加载。

| 用途 | 来源 | 推荐文件名 | 放置位置 |
| --- | --- | --- | --- |
| FLUX.2-Klein 底模 | [black-forest-labs/FLUX.2-klein-base-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) | `flux-2-klein/flux-2-klein-4b-fp8.safetensors` | `ComfyUI/models/diffusion_models` |
| RUM checkpoint | [rimochan/RUM-FLUX.2-klein-4B-preview](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview) | `rum-flux2-klein-4b-preview.safetensors` | `ComfyUI/models/diffusion_models` |
| FLUX.2-Klein Qwen 文本编码器 | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | `qwen_3_4b.safetensors` | `ComfyUI/models/text_encoders` |
| FLUX.2 VAE | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) | `flux2-vae.safetensors` | `ComfyUI/models/vae` |
| 普通 native SDXL CLIP-L | [Comfy-Org/stable-diffusion-3.5-fp8](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) | `clip_l.safetensors` | `ComfyUI/models/text_encoders` 或 `ComfyUI/models/clip` |
| 普通 native SDXL CLIP-G | [Comfy-Org/stable-diffusion-3.5-fp8](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) | `clip_g.safetensors` | `ComfyUI/models/text_encoders` 或 `ComfyUI/models/clip` |
| diffusers-match teacher CLIP-L | [Ine007/waiIllustriousSDXL_v160](https://huggingface.co/Ine007/waiIllustriousSDXL_v160) | `waiIllustriousSDXL_v160_clip_l.safetensors` | `ComfyUI/models/text_encoders` |
| diffusers-match teacher CLIP-G | [Ine007/waiIllustriousSDXL_v160](https://huggingface.co/Ine007/waiIllustriousSDXL_v160) | `waiIllustriousSDXL_v160_clip_g.safetensors` | `ComfyUI/models/text_encoders` |

如果你看到 `model_index.json not found`，通常代表你把 diffusers 文件夹拿去给 ComfyUI native loader 读了。native ComfyUI loader 要的是 `.safetensors` 文件，不是 diffusers 目录。

## 安装

把仓库 clone 到 `custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

然后重启 ComfyUI。

如果你用的是秋叶整合包，路径通常像这样：

```text
I:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-RUM
```

## 下载模型

### Windows / 秋叶整合包推荐命令

在 `I:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-RUM` 里打开终端，然后运行：

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\download_models.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI --all
```

`--all` 会下载：底模、RUM checkpoint、Qwen、VAE、普通 native CLIP、diffusers-match teacher CLIP。

如果 Hugging Face 要求登录或同意协议，先去网页接受模型条款，然后用 token：

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\download_models.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI --all --token YOUR_HF_TOKEN
```

### 通用命令

只下载最基础文件：

```bash
python scripts/download_models.py
```

普通 native workflow 还需要普通 SDXL CLIP：

```bash
python scripts/download_models.py --include-sdxl-clip
```

`diffusers-match` workflow 还需要 teacher CLIP：

```bash
python scripts/download_models.py --include-teacher-clip
```

一次全部下载：

```bash
python scripts/download_models.py --all
```

### 手动下载对应关系

如果脚本下载失败，可以手动从 Hugging Face 下载后改名放进 ComfyUI：

| Hugging Face 文件 | 放到 ComfyUI 后的路径 |
| --- | --- |
| `black-forest-labs/FLUX.2-klein-base-4b-fp8/flux-2-klein-base-4b-fp8.safetensors` | `models/diffusion_models/flux-2-klein/flux-2-klein-4b-fp8.safetensors` |
| `rimochan/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors` | `models/diffusion_models/rum-flux2-klein-4b-preview.safetensors` |
| `Comfy-Org/z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors` | `models/text_encoders/qwen_3_4b.safetensors` |
| `Comfy-Org/flux2-dev/split_files/vae/flux2-vae.safetensors` | `models/vae/flux2-vae.safetensors` |
| `Comfy-Org/stable-diffusion-3.5-fp8/text_encoders/clip_l.safetensors` | `models/text_encoders/clip_l.safetensors` |
| `Comfy-Org/stable-diffusion-3.5-fp8/text_encoders/clip_g.safetensors` | `models/text_encoders/clip_g.safetensors` |
| `Ine007/waiIllustriousSDXL_v160/text_encoder/model.safetensors` | `models/text_encoders/waiIllustriousSDXL_v160_clip_l.safetensors` |
| `Ine007/waiIllustriousSDXL_v160/text_encoder_2/model.safetensors` | `models/text_encoders/waiIllustriousSDXL_v160_clip_g.safetensors` |

如果你已经有旧 diffusers 分支用的 `waiIllustriousSDXL_v160_text` 文件夹，也可以直接复制 teacher CLIP：

```bash
python scripts/install_diffusers_teacher_clip.py /path/to/waiIllustriousSDXL_v160_text
```

复制后在 `DualCLIPLoader` 里选择：

```text
waiIllustriousSDXL_v160_clip_l.safetensors
waiIllustriousSDXL_v160_clip_g.safetensors
```

## 示例 workflow

### 普通 native workflow

适合正常在 ComfyUI 里使用：

```text
examples/basic_workflow_api.json
```

需要这些模型：

```text
models/diffusion_models/flux-2-klein/flux-2-klein-4b-fp8.safetensors
models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
models/text_encoders/qwen_3_4b.safetensors
models/text_encoders/clip_l.safetensors
models/text_encoders/clip_g.safetensors
models/vae/flux2-vae.safetensors
```

节点结构：

1. `Load Diffusion Model` 读取 `flux-2-klein/flux-2-klein-4b-fp8.safetensors`。
2. `RUM FLUX.2 Apply Model Patch` 读取 `rum-flux2-klein-4b-preview.safetensors`。
3. `Load CLIP` 读取 `qwen_3_4b.safetensors`，`type=flux2`。
4. `DualCLIPLoader` 读取 `clip_l.safetensors` + `clip_g.safetensors`，`type=sdxl`。
5. 正面 prompt 分别走 Qwen 和 SDXL CLIP，再接 `RUM FLUX.2 Combine Conditioning`。
6. 负面 prompt 也同样合并。
7. 使用 FLUX.2 原生采样链：`Empty Flux 2 Latent` → `RandomNoise` → `CFGGuider` → `KSamplerSelect` → `Flux2Scheduler` → `SamplerCustomAdvanced` → `VAE Decode` → `Save Image`。

推荐初始参数：

```text
base_text_tokens=512
extra_text_tokens=77
sdxl_clip_width=2048
use_guidance_embedding=false
CFGGuider cfg=5.0
sampler=euler
steps=20
width=960
height=1024
```

### diffusers-match workflow

如果你想使用旧 diffusers 推理路径对应的 native 近似设置，用：

```text
examples/diffusers_match_workflow.json
examples/diffusers_match_workflow_api.json
```

需要这些模型：

```text
models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
models/text_encoders/qwen_3_4b.safetensors
models/text_encoders/waiIllustriousSDXL_v160_clip_l.safetensors
models/text_encoders/waiIllustriousSDXL_v160_clip_g.safetensors
models/vae/flux2-vae.safetensors
```

这个 workflow 和普通 native workflow 不一样：

- 使用 `RUM FLUX.2 Load Native Model` 直接加载 RUM checkpoint。
- 正面 Qwen 层号设成 `10,20,30`。
- 正面 conditioning 使用 `base_text_tokens=200` 和 SDXL extra 77 tokens。
- 负面 conditioning 使用默认 Qwen 512 tokens，不拼 SDXL extra。
- 使用 `RUM FLUX.2 Diffusers Noise` 生成 CPU BF16 noise。
- 使用 `RUM FLUX.2 Diffusers CFG Guider` 标记 positive / negative 分支。
- `DualCLIPLoader` 必须选 `waiIllustriousSDXL_v160_clip_l.safetensors` + `waiIllustriousSDXL_v160_clip_g.safetensors`，否则会明显不像旧 diffusers 图。

这不是给日常工作流用的“魔法增强节点”。它只是为了让 native ComfyUI 更接近旧 diffusers 推理路径；实际图像仍可能和 diffusers 参考图明显不同。

## 验证安装

用 ComfyUI 同一个 Python 环境运行：

```bash
python scripts/check_install.py
```

秋叶整合包示例：

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\check_install.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI
```

它会检查节点能否 import，以及 ComfyUI 是否看得到 `diffusion_models`、`text_encoders`、`vae` 里的推荐文件。`clip_l.safetensors` / `clip_g.safetensors` 是普通 native workflow 用；`waiIllustriousSDXL_v160_clip_l.safetensors` / `waiIllustriousSDXL_v160_clip_g.safetensors` 是 diffusers-match workflow 用。

## 当前限制

- native 版仍然是实验性质。
- `RUM FLUX.2 Apply Model Patch` 需要 FLUX.2-Klein 4B 兼容底模。
- 普通 native workflow 的 `base_text_tokens` 建议用 `512`。
- diffusers-match workflow 的 `base_text_tokens` 是 `200`，不要混到普通 workflow。
- 旧 diffusers 分支和 native ComfyUI 不保证像素级一致。
- 上游 RUM 没有明确 LICENSE 前，不建议发布到 Comfy Registry。

## 常见问题

### 图片很糊、全是噪声、结构明显不对

最常见原因是采样链错了。FLUX.2-Klein 不应该用普通 `KSampler` 乱接；请使用：

```text
RandomNoise → CFGGuider → KSamplerSelect → Flux2Scheduler → SamplerCustomAdvanced
```

如果你在复现 diffusers 参考图，请用 `examples/diffusers_match_workflow_api.json`，不要用普通 native workflow。

### 生成出来不像旧 diffusers 图

直接说：这是正常风险，不是你眼睛有问题。几个地方会让图差很多：

- 你用了普通 native workflow，而不是 diffusers-match workflow。
- `DualCLIPLoader` 选了普通 `clip_l.safetensors` / `clip_g.safetensors`，而不是 teacher CLIP。
- 正面 Qwen 层号不是 `10,20,30`。
- positive / negative token 长度被改错。
- noise 不是 CPU BF16 diffusers 风格。
- ComfyUI 缓存了旧 CLIP 或旧节点结果。

当前 diffusers-match workflow 已经修正这些明显坑，但仍不承诺角色、构图或像素级复刻。

### 下拉列表看不到 RUM checkpoint

确认文件在：

```text
ComfyUI/models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
```

然后刷新模型列表或重启 ComfyUI。

### `model_index.json not found`

你选的是 diffusers 文件夹，不是 `.safetensors` 文件。ComfyUI native loader 需要 `.safetensors`。

### 爆显存

FLUX.2-Klein 4B 加 Qwen3 4B 文本编码器很吃显存。建议：

- 先用 FP8 底模。
- 降低分辨率测试。
- 关掉其它已加载模型。
- 改模型后重启 ComfyUI。

## 特别感谢

- **RimoChan / RUM**：原始 RUM 项目、模型与参考推理代码来自 [RimoChan/RUM](https://github.com/RimoChan/RUM)。没有这个项目，就没有这个 ComfyUI 适配。
- **ComfyUI**：节点系统和 native workflow 基础来自 [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)。

---

# ComfyUI-RUM English README

This is a **native ComfyUI node adapter** for [RimoChan/RUM](https://github.com/RimoChan/RUM), targeting `RUM-FLUX.2-klein-4B-preview`. The original RUM idea, model, and reference inference code come from RimoChan's project.

The `main` branch is the **native ComfyUI version**: model loading, conditioning, latent creation, sampling, VAE decode, and image saving are routed through ComfyUI systems as much as possible. The old all-in-one diffusers wrapper is kept on the [`diffusers-pipeline`](https://github.com/peter119lee/ComfyUI-RUM/tree/diffusers-pipeline) branch.

> **AI Writing Notice**
>
> The adapter code and README in this ComfyUI repository were written and organized by Codex 5.5; the original RUM project, model, and reference inference code still come from [RimoChan/RUM](https://github.com/RimoChan/RUM).

> **License Notice**
>
> This project adapts behavior from the public RUM reference inference code. At the time this project was created, upstream `RimoChan/RUM` did not provide a clear LICENSE. Unless upstream adds a LICENSE or gives explicit permission, publishing this to Comfy Registry or redistributing it widely is not recommended.
>
> This repository does not include model weights. Model files follow their own upstream licenses and access rules.

## What It Does

RUM adds one extra SDXL CLIP text-conditioning path on top of FLUX.2-Klein. Standard ComfyUI FLUX.2 nodes do not know how to feed that extra conditioning into the RUM transformer, so this node pack provides:

- `RUM FLUX.2 Apply Model Patch`: applies RUM weights and dual text-projection layers onto a normal ComfyUI FLUX.2-Klein `MODEL`.
- `RUM FLUX.2 Combine Conditioning`: combines FLUX.2-Klein Qwen conditioning and SDXL CLIP conditioning into the format expected by RUM.
- `RUM FLUX.2 Load Native Model`: converts the RUM diffusers-format checkpoint directly into a ComfyUI `MODEL`, mainly for the diffusers-match workflow.
- `RUM FLUX.2 Diffusers Noise` / `RUM FLUX.2 Diffusers CFG Guider`: helper nodes used only to approximate the old diffusers reference path. They are not recommended for normal workflows.

In short: the normal native workflow is for regular ComfyUI usage; the diffusers-match workflow approximates the most important differences from the old diffusers path, but it does not guarantee the same image.

## Why Native ComfyUI Cannot Match Diffusers 100%

The direct reason is: **it is not the same execution environment**. The public RUM reference path uses diffusers. This native adapter rebuilds that path inside ComfyUI's model, CLIP, sampler, noise, and conditioning systems. If even one detail changes, the diffusion process can drift over sampling steps, causing the final character, composition, or style to change.

The most important differences are:

- **Different SDXL teacher CLIP**: the old diffusers path used the text encoders from `Ine007/waiIllustriousSDXL_v160`; generic `clip_l.safetensors` / `clip_g.safetensors` are not the same semantic condition.
- **Different Qwen hidden states**: the old positive prompt path used specific Qwen layers `10,20,30`; normal ComfyUI encoding does not do this by default.
- **Different token lengths**: the old positive path used Qwen 200 tokens + SDXL 77 tokens, while the negative path keeps Qwen 512 tokens instead of using the same merge logic.
- **Different noise**: the old reference path used CPU BF16-style noise; ComfyUI's standard `RandomNoise` is not identical.
- **Different CFG branch behavior**: positive and negative branches are not routed exactly the same way as standard ComfyUI `CFGGuider`.
- **Different floating-point and scheduler details**: dtype, device, scheduler, patch order, and ComfyUI cache behavior can all affect the result.

So the realistic goal is: **make the native node usable, structured, and reasonably close to diffusers; not promise same-seed same-image output**. Use the `diffusers-match` workflow if you want the closest native approximation to the old diffusers path. Use the normal native workflow if you want a more standard ComfyUI setup.

## Model Locations

Use standard ComfyUI model paths. Do not load a diffusers folder directly with native ComfyUI loaders.

| Purpose | Source | Recommended Filename | Location |
| --- | --- | --- | --- |
| FLUX.2-Klein base model | [black-forest-labs/FLUX.2-klein-base-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) | `flux-2-klein/flux-2-klein-4b-fp8.safetensors` | `ComfyUI/models/diffusion_models` |
| RUM checkpoint | [rimochan/RUM-FLUX.2-klein-4B-preview](https://huggingface.co/rimochan/RUM-FLUX.2-klein-4B-preview) | `rum-flux2-klein-4b-preview.safetensors` | `ComfyUI/models/diffusion_models` |
| FLUX.2-Klein Qwen text encoder | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) | `qwen_3_4b.safetensors` | `ComfyUI/models/text_encoders` |
| FLUX.2 VAE | [Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev) | `flux2-vae.safetensors` | `ComfyUI/models/vae` |
| Normal native SDXL CLIP-L | [Comfy-Org/stable-diffusion-3.5-fp8](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) | `clip_l.safetensors` | `ComfyUI/models/text_encoders` or `ComfyUI/models/clip` |
| Normal native SDXL CLIP-G | [Comfy-Org/stable-diffusion-3.5-fp8](https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8) | `clip_g.safetensors` | `ComfyUI/models/text_encoders` or `ComfyUI/models/clip` |
| diffusers-match teacher CLIP-L | [Ine007/waiIllustriousSDXL_v160](https://huggingface.co/Ine007/waiIllustriousSDXL_v160) | `waiIllustriousSDXL_v160_clip_l.safetensors` | `ComfyUI/models/text_encoders` |
| diffusers-match teacher CLIP-G | [Ine007/waiIllustriousSDXL_v160](https://huggingface.co/Ine007/waiIllustriousSDXL_v160) | `waiIllustriousSDXL_v160_clip_g.safetensors` | `ComfyUI/models/text_encoders` |

If you see `model_index.json not found`, you probably selected a diffusers folder instead of a `.safetensors` file. Native ComfyUI loaders expect `.safetensors`, not diffusers directories.

## Installation

Clone this repository into `custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

Restart ComfyUI after installation.

For Aki's ComfyUI package on Windows, the path usually looks like:

```text
I:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-RUM
```

## Download Models

### Recommended Windows / Aki Command

Open a terminal inside `I:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-RUM`, then run:

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\download_models.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI --all
```

`--all` downloads the base model, RUM checkpoint, Qwen text encoder, VAE, generic native CLIP files, and diffusers-match teacher CLIP files.

If Hugging Face requires login or model-gate approval, accept the model terms on the website first, then run:

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\download_models.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI --all --token YOUR_HF_TOKEN
```

### Generic Commands

Download only the basic files:

```bash
python scripts/download_models.py
```

The normal native workflow also needs generic SDXL CLIP files:

```bash
python scripts/download_models.py --include-sdxl-clip
```

The `diffusers-match` workflow also needs teacher CLIP files:

```bash
python scripts/download_models.py --include-teacher-clip
```

Download everything:

```bash
python scripts/download_models.py --all
```

### Manual Download Mapping

If the script fails, download manually from Hugging Face and place/rename the files as follows:

| Hugging Face File | ComfyUI Path |
| --- | --- |
| `black-forest-labs/FLUX.2-klein-base-4b-fp8/flux-2-klein-base-4b-fp8.safetensors` | `models/diffusion_models/flux-2-klein/flux-2-klein-4b-fp8.safetensors` |
| `rimochan/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors` | `models/diffusion_models/rum-flux2-klein-4b-preview.safetensors` |
| `Comfy-Org/z_image_turbo/split_files/text_encoders/qwen_3_4b.safetensors` | `models/text_encoders/qwen_3_4b.safetensors` |
| `Comfy-Org/flux2-dev/split_files/vae/flux2-vae.safetensors` | `models/vae/flux2-vae.safetensors` |
| `Comfy-Org/stable-diffusion-3.5-fp8/text_encoders/clip_l.safetensors` | `models/text_encoders/clip_l.safetensors` |
| `Comfy-Org/stable-diffusion-3.5-fp8/text_encoders/clip_g.safetensors` | `models/text_encoders/clip_g.safetensors` |
| `Ine007/waiIllustriousSDXL_v160/text_encoder/model.safetensors` | `models/text_encoders/waiIllustriousSDXL_v160_clip_l.safetensors` |
| `Ine007/waiIllustriousSDXL_v160/text_encoder_2/model.safetensors` | `models/text_encoders/waiIllustriousSDXL_v160_clip_g.safetensors` |

If you already have the old diffusers branch's `waiIllustriousSDXL_v160_text` folder, you can copy the teacher CLIP files directly:

```bash
python scripts/install_diffusers_teacher_clip.py /path/to/waiIllustriousSDXL_v160_text
```

Then select these in `DualCLIPLoader`:

```text
waiIllustriousSDXL_v160_clip_l.safetensors
waiIllustriousSDXL_v160_clip_g.safetensors
```

## Example Workflows

### Normal Native Workflow

Use this for regular ComfyUI work:

```text
examples/basic_workflow_api.json
```

Required models:

```text
models/diffusion_models/flux-2-klein/flux-2-klein-4b-fp8.safetensors
models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
models/text_encoders/qwen_3_4b.safetensors
models/text_encoders/clip_l.safetensors
models/text_encoders/clip_g.safetensors
models/vae/flux2-vae.safetensors
```

Recommended starting parameters:

```text
base_text_tokens=512
extra_text_tokens=77
sdxl_clip_width=2048
use_guidance_embedding=false
CFGGuider cfg=5.0
sampler=euler
steps=20
width=960
height=1024
```

### diffusers-match Workflow

Use this if you want a native approximation of the old diffusers reference path:

```text
examples/diffusers_match_workflow.json
examples/diffusers_match_workflow_api.json
```

Required models:

```text
models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
models/text_encoders/qwen_3_4b.safetensors
models/text_encoders/waiIllustriousSDXL_v160_clip_l.safetensors
models/text_encoders/waiIllustriousSDXL_v160_clip_g.safetensors
models/vae/flux2-vae.safetensors
```

This workflow differs from the normal native workflow:

- It uses `RUM FLUX.2 Load Native Model` to load the RUM checkpoint directly.
- Positive Qwen layers are set to `10,20,30`.
- Positive conditioning uses `base_text_tokens=200` plus 77 SDXL extra tokens.
- Negative conditioning keeps default Qwen 512 tokens and does not append SDXL extra tokens.
- It uses `RUM FLUX.2 Diffusers Noise` for CPU BF16-style noise.
- It uses `RUM FLUX.2 Diffusers CFG Guider` to mark positive and negative branches.
- `DualCLIPLoader` must use `waiIllustriousSDXL_v160_clip_l.safetensors` + `waiIllustriousSDXL_v160_clip_g.safetensors`; using the generic CLIP files will make the result much less similar to the old diffusers output.

This is not a magic quality node for normal workflows. It only tries to make native ComfyUI closer to the old diffusers reference path; the actual image can still differ significantly.

## Verify Installation

Run this with the same Python environment used by ComfyUI:

```bash
python scripts/check_install.py
```

Aki package example:

```bat
I:\ComfyUI-aki-v1.6\python\python.exe scripts\check_install.py --comfy-root I:\ComfyUI-aki-v1.6\ComfyUI
```

The script checks whether the node can import and whether ComfyUI can see recommended files under `diffusion_models`, `text_encoders`, and `vae`.

## Current Limitations

- The native version is still experimental.
- `RUM FLUX.2 Apply Model Patch` requires a compatible FLUX.2-Klein 4B base model.
- The normal native workflow should usually use `base_text_tokens=512`.
- The diffusers-match workflow uses `base_text_tokens=200`; do not mix that setting into the normal workflow.
- The old diffusers branch and native ComfyUI do not guarantee pixel-level or character-level reproduction.
- Until upstream RUM has a clear LICENSE, publishing this to Comfy Registry is not recommended.

## FAQ

### The image is blurry, noisy, or structurally broken

The most common cause is a wrong sampling chain. FLUX.2-Klein should not be connected to a random plain `KSampler` setup. Use:

```text
RandomNoise → CFGGuider → KSamplerSelect → Flux2Scheduler → SamplerCustomAdvanced
```

If you are trying to reproduce the old diffusers reference image, use `examples/diffusers_match_workflow_api.json`, not the normal native workflow.

### The image does not look like the old diffusers output

That is a real limitation, not your imagination. Common causes:

- You used the normal native workflow instead of the diffusers-match workflow.
- `DualCLIPLoader` used generic `clip_l.safetensors` / `clip_g.safetensors` instead of teacher CLIP.
- Positive Qwen layers were not `10,20,30`.
- Positive / negative token lengths were changed incorrectly.
- Noise was not CPU BF16 diffusers-style noise.
- ComfyUI cached old CLIP or old node outputs.

The current diffusers-match workflow fixes the obvious traps, but it still does not promise exact character, composition, or pixel reproduction.

### RUM checkpoint does not appear in the dropdown

Confirm this file exists:

```text
ComfyUI/models/diffusion_models/rum-flux2-klein-4b-preview.safetensors
```

Then refresh model lists or restart ComfyUI.

### `model_index.json not found`

You selected a diffusers folder, not a `.safetensors` file. Native ComfyUI loaders need `.safetensors`.

### Out of VRAM

FLUX.2-Klein 4B plus Qwen3 4B text encoder is heavy. Try:

- Use the FP8 base model first.
- Test at a lower resolution.
- Unload other models.
- Restart ComfyUI after changing models.

## Special Thanks

- **RimoChan / RUM**: the original RUM project, model, and reference inference code come from [RimoChan/RUM](https://github.com/RimoChan/RUM). This ComfyUI adapter would not exist without that project.
- **ComfyUI**: node system and native workflow foundation come from [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI).
