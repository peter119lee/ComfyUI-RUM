# ComfyUI-RUM

Native ComfyUI adapter nodes for [RimoChan/RUM](https://github.com/RimoChan/RUM) `RUM-FLUX.2-klein-4B-preview`.

This main branch is **native ComfyUI mode**: it keeps ComfyUI's normal `MODEL`, `CONDITIONING`, `LATENT`, `KSampler`, and `VAE Decode` workflow, then adds only the RUM-specific adapter pieces.

The older self-contained diffusers pipeline version is kept in the [`diffusers-pipeline`](https://github.com/peter119lee/ComfyUI-RUM/tree/diffusers-pipeline) branch.

> **License / publishing warning**
>
> This wrapper references and adapts behavior from RUM's public inference code. At the time this project was created, upstream `RimoChan/RUM` did not include a clear LICENSE. Do not publish to Comfy Registry or make broad public claims unless upstream permission/license is clarified.
>
> This repository does not include model weights. Each model file follows its own upstream license and access rules.

## What It Does

RUM adds one extra SDXL CLIP conditioning route to FLUX.2-Klein. Normal ComfyUI FLUX.2 nodes do not know how to feed that extra route into the transformer, so this custom node pack provides:

- `RUM FLUX.2 Apply Model Patch`: patches a normal ComfyUI FLUX.2-Klein `MODEL` with RUM weights and a dual text projection layer.
- `RUM FLUX.2 Combine Conditioning`: combines FLUX.2-Klein text conditioning with SDXL CLIP conditioning into the format RUM expects.

After those two nodes, the rest is normal ComfyUI: `KSampler`, `Empty Flux 2 Latent`, `VAE Decode`, `Save Image`, etc.

## Why Not Only Diffusers?

The diffusers branch works, but it behaves like a separate mini pipeline inside ComfyUI. Native mode is better for ComfyUI users because:

- models live in normal ComfyUI folders
- workflows use normal sampler/latent/VAE nodes
- ComfyUI can manage model loading/offloading
- future LoRA, ControlNet, scheduler, and workflow compatibility is more realistic

## Required Models

Put files in the usual ComfyUI model folders:

| Purpose | Example file | ComfyUI folder |
| --- | --- | --- |
| FLUX.2-Klein base diffusion model | `flux-2-klein/flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models` |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models` |
| FLUX.2-Klein text encoder | `qwen_3_4b.safetensors` | `models/text_encoders` |
| SDXL CLIP-L | `clip_l.safetensors` | `models/text_encoders` or `models/clip` |
| SDXL CLIP-G | `clip_g.safetensors` | `models/text_encoders` or `models/clip` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae` |

If you already have an SDXL checkpoint loader workflow that outputs SDXL `CLIP`, you can use that CLIP instead of separate `clip_l` / `clip_g` files.

## Install

Clone into ComfyUI custom nodes:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

Restart ComfyUI.

## Download Helper

From this custom node folder:

```bash
python scripts/download_models.py
```

To also try downloading `clip_l.safetensors` and `clip_g.safetensors`:

```bash
python scripts/download_models.py --include-sdxl-clip
```

If Hugging Face requires authentication for any file:

```bash
python scripts/download_models.py --token YOUR_HF_TOKEN
```

## Basic Workflow

Recommended native graph:

1. `Load Diffusion Model` loads `flux-2-klein/flux-2-klein-4b-fp8.safetensors`.
2. `RUM FLUX.2 Apply Model Patch` patches that model with the RUM checkpoint.
3. `Load CLIP` loads `qwen_3_4b.safetensors` with `type=flux2`.
4. `DualCLIPLoader` loads `clip_l.safetensors` + `clip_g.safetensors` with `type=sdxl`.
5. Encode the same positive prompt with both CLIP paths.
6. Use `RUM FLUX.2 Combine Conditioning` for positive conditioning.
7. Repeat empty or negative prompt conditioning for negative.
8. Use normal `Empty Flux 2 Latent` → `KSampler` → `VAE Decode` → `Save Image`.

A minimal API workflow is available at:

```text
examples/basic_workflow_api.json
```

Recommended first settings:

```text
base_text_tokens=512
extra_text_tokens=77
sdxl_clip_width=2048
guidance=5.0
KSampler cfg=1.0
steps=20
width=960
height=1024
```

## Verification

Run with the same Python environment used by ComfyUI:

```bash
python scripts/check_install.py
```

This checks node imports and whether ComfyUI can see files in `diffusion_models`, `text_encoders`, and `vae`.

## Current Limitations

Native mode is still experimental.

Known limitations:

- The RUM patch node expects a FLUX.2-Klein 4B-compatible base model.
- `base_text_tokens` must match the FLUX.2 text conditioning length; normal ComfyUI FLUX.2-Klein uses `512`.
- LoRA and ControlNet compatibility has not been validated yet.
- The example uses a basic `KSampler`; more accurate FLUX.2 scheduler workflows may still need tuning.

## Troubleshooting

### RUM checkpoint does not appear in the dropdown

Put the RUM `.safetensors` file in:

```text
ComfyUI/models/diffusion_models
```

Then restart ComfyUI or refresh model lists.

You can also paste an absolute path into `rum_checkpoint_path` on `RUM FLUX.2 Apply Model Patch`.

### Shape mismatch in strict mode

You probably loaded the wrong base model. Use a FLUX.2-Klein 4B base model. If you know what you are doing, set `strict=false`, but bad outputs are likely.

### SDXL conditioning dimension looks wrong

Use SDXL CLIP-L + CLIP-G together through `DualCLIPLoader type=sdxl`. The expected combined SDXL width is `2048`.

### CUDA out of memory

Try:

- lower width and height
- keep text encoders on CPU if your ComfyUI build allows it
- close other loaded models
- restart ComfyUI after changing large models

## Credits

- RUM method and upstream inference scripts: [RimoChan/RUM](https://github.com/RimoChan/RUM)
- FLUX.2-Klein base: `black-forest-labs/FLUX.2-klein-base-4b-fp8`
- RUM preview weights: `rimochan/RUM-FLUX.2-klein-4B-preview`
- ComfyUI native FLUX.2 support: ComfyUI core

---

# 简体中文说明

这是给 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 `RUM-FLUX.2-klein-4B-preview` 做的 ComfyUI 原生适配节点。

主分支现在是 **Native ComfyUI 模式**：模型、conditioning、latent、KSampler、VAE Decode 都尽量走 ComfyUI 原生系统。旧的 diffusers 一体式版本已经放到 [`diffusers-pipeline`](https://github.com/peter119lee/ComfyUI-RUM/tree/diffusers-pipeline) 分支。

> **授权提醒**
>
> 这个项目参考并适配了 RUM 公开推理代码的行为。创建这个项目时，上游 `RimoChan/RUM` 没有明确 LICENSE。除非上游补 LICENSE 或明确授权，否则不建议提交到 Comfy Registry 或大范围公开发布。
>
> 本 repo 不包含模型权重；模型文件遵守各自上游授权。

## 它解决什么问题

RUM 在 FLUX.2-Klein 上额外加了一路 SDXL CLIP 条件。普通 ComfyUI 的 FLUX.2-Klein 节点不知道怎么把这一路喂进 transformer，所以作者才说“ComfyUI 跑不了了，只能用 diffusers”。

这个 custom node pack 补的就是这块：

- `RUM FLUX.2 Apply Model Patch`：把普通 ComfyUI FLUX.2-Klein `MODEL` patch 成 RUM 需要的模型。
- `RUM FLUX.2 Combine Conditioning`：把 FLUX.2-Klein 文本特征和 SDXL CLIP 文本特征拼成 RUM 需要的 conditioning。

后面继续用普通 ComfyUI 节点：`KSampler`、`Empty Flux 2 Latent`、`VAE Decode`、`Save Image`。

## 为什么不用 diffusers 版就好

diffusers 版能跑，但它更像在 ComfyUI 里面塞了一个外部 pipeline。Native 版更像真正 ComfyUI 节点：

- 模型放正常 ComfyUI 模型目录
- workflow 更接近普通 ComfyUI 用法
- ComfyUI 可以管理模型加载和卸载
- 以后接 LoRA、ControlNet、调度器、工作流会更现实

## 需要的模型

把文件放到正常 ComfyUI 模型目录：

| 用途 | 示例文件 | ComfyUI 文件夹 |
| --- | --- | --- |
| FLUX.2-Klein base | `flux-2-klein/flux-2-klein-4b-fp8.safetensors` | `models/diffusion_models` |
| RUM checkpoint | `rum-flux2-klein-4b-preview.safetensors` | `models/diffusion_models` |
| FLUX.2-Klein 文本编码器 | `qwen_3_4b.safetensors` | `models/text_encoders` |
| SDXL CLIP-L | `clip_l.safetensors` | `models/text_encoders` 或 `models/clip` |
| SDXL CLIP-G | `clip_g.safetensors` | `models/text_encoders` 或 `models/clip` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae` |

如果你本来就有 SDXL checkpoint loader 工作流，可以直接用那个 loader 输出的 SDXL `CLIP`，不一定要单独的 `clip_l` / `clip_g` 文件。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

然后重启 ComfyUI。

## 下载模型

在这个 custom node 文件夹运行：

```bash
python scripts/download_models.py
```

如果也想尝试下载 SDXL 的 `clip_l.safetensors` 和 `clip_g.safetensors`：

```bash
python scripts/download_models.py --include-sdxl-clip
```

如果 Hugging Face 要 token：

```bash
python scripts/download_models.py --token YOUR_HF_TOKEN
```

## 基础工作流

推荐节点顺序：

1. `Load Diffusion Model` 载入 `flux-2-klein/flux-2-klein-4b-fp8.safetensors`。
2. `RUM FLUX.2 Apply Model Patch` 选择 RUM checkpoint。
3. `Load CLIP` 载入 `qwen_3_4b.safetensors`，`type=flux2`。
4. `DualCLIPLoader` 载入 `clip_l.safetensors` + `clip_g.safetensors`，`type=sdxl`。
5. 同一个正向 prompt 分别用 FLUX2 CLIP 和 SDXL CLIP encode。
6. 用 `RUM FLUX.2 Combine Conditioning` 合并正向 conditioning。
7. 负向 conditioning 也重复一次，可以用空 prompt。
8. 后面接正常 `Empty Flux 2 Latent` → `KSampler` → `VAE Decode` → `Save Image`。

示例 API workflow：

```text
examples/basic_workflow_api.json
```

推荐初始参数：

```text
base_text_tokens=512
extra_text_tokens=77
sdxl_clip_width=2048
guidance=5.0
KSampler cfg=1.0
steps=20
width=960
height=1024
```

## 验证安装

用 ComfyUI 同一个 Python 环境运行：

```bash
python scripts/check_install.py
```

它会检查节点 import，以及 ComfyUI 是否看得到 `diffusion_models`、`text_encoders`、`vae` 里的文件。

## 当前限制

直接说：这个 native 版还是 experimental。

目前限制：

- RUM patch node 需要 FLUX.2-Klein 4B 兼容底模。
- `base_text_tokens` 要和 FLUX.2 文本 conditioning 长度一致；普通 ComfyUI FLUX.2-Klein 通常是 `512`。
- LoRA / ControlNet 兼容性还没验证。
- 示例先用基础 `KSampler`，更准确的 FLUX.2 scheduler workflow 之后还可以继续调。

## 常见问题

### 下拉列表看不到 RUM checkpoint

把 RUM `.safetensors` 放到：

```text
ComfyUI/models/diffusion_models
```

然后重启 ComfyUI 或刷新模型列表。

也可以直接在 `RUM FLUX.2 Apply Model Patch` 的 `rum_checkpoint_path` 里填绝对路径。

### strict mode 报 shape mismatch

大概率是底模选错了。请用 FLUX.2-Klein 4B base。你可以关掉 `strict`，但很可能出烂图。

### SDXL conditioning 维度不对

请用 `DualCLIPLoader type=sdxl` 同时加载 CLIP-L 和 CLIP-G。RUM 预期 SDXL CLIP 合并宽度是 `2048`。

### 爆显存

先试：

- 降低宽高
- 文本编码器尽量放 CPU
- 关掉其他大模型
- 换模型后重启 ComfyUI
