# ComfyUI-RUM

这是给 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 **ComfyUI 原生节点适配器**，目标模型是 `RUM-FLUX.2-klein-4B-preview`。

主分支是 **native ComfyUI 版**：模型加载、conditioning、latent、sampler、VAE decode 都尽量走 ComfyUI 自己的系统；旧的 diffusers 一体式版本保留在 [`diffusers-pipeline`](https://github.com/peter119lee/ComfyUI-RUM/tree/diffusers-pipeline) 分支。

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

直接说：**因为这不是同一个执行环境**。RUM 上游现在能稳定跑的是 diffusers 推理；ComfyUI native 是把它拆进 ComfyUI 的模型、CLIP、sampler、noise、conditioning 系统里跑。只要其中一个细节不同，扩散模型就会越采样越偏，最后角色和构图都可能变。

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

## Credits

- RUM / original research code: [RimoChan/RUM](https://github.com/RimoChan/RUM)
- ComfyUI: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
