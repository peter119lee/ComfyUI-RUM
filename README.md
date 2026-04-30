# ComfyUI-RUM

ComfyUI nodes for running [RimoChan/RUM](https://github.com/RimoChan/RUM)'s `RUM-FLUX.2-klein-4B-preview` checkpoint inside ComfyUI.

RUM transfers anime-generation behavior from an SDXL teacher model into newer model architectures. This custom node wraps the upstream diffusers inference path and adds the extra SDXL CLIP conditioning branch required by the RUM FLUX.2-klein checkpoint.

> **License / permission warning**
>
> This wrapper mirrors parts of the public RUM inference implementation. The upstream RUM repository did not include a license when this wrapper was created. Keep this repository private or treat public redistribution as pending upstream permission unless a license is added or permission is granted.
>
> Model files are not included and remain under their own upstream licenses.

## Features

- Loads the FLUX.2-klein base model.
- Loads the RUM FLUX.2-klein preview checkpoint.
- Loads SDXL text encoders/tokenizers for RUM's extra CLIP branch.
- Generates standard ComfyUI `IMAGE` output.
- Includes an unload node for clearing cached RUM pipelines.

## Limitations

This is currently a self-contained diffusers pipeline node, not a native ComfyUI sampler stack.

Not supported yet:

- KSampler-compatible model output
- latent input/output workflows
- ControlNet / IP-Adapter
- ComfyUI LoRA stacking
- negative prompt input

## Nodes

- `RUM FLUX.2-klein Loader`
- `RUM FLUX.2-klein Sampler`
- `RUM Unload Models`

## Installation

Clone into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Install dependencies into the same Python environment that runs ComfyUI.

## Download Models

The helper downloads about 24 GB of required files:

```bash
python scripts/download_models.py --write-local-config
```

Downloaded sources:

| Purpose | Source | Default folder |
| --- | --- | --- |
| FLUX.2-klein base | `black-forest-labs/FLUX.2-klein-base-4B` | `models/FLUX.2-klein-base-4B` |
| RUM checkpoint | `rimochan/RUM-FLUX.2-klein-4B-preview` | `models/RUM-FLUX.2-klein-4B-preview` |
| SDXL text encoders | `Ine007/waiIllustriousSDXL_v160` | `models/waiIllustriousSDXL_v160_text` |

`--write-local-config` creates `local_config.json`, which is ignored by Git.

## Manual Configuration

If you already have the models, create `local_config.json` next to `nodes.py`:

```json
{
  "base_model_path": "models/FLUX.2-klein-base-4B",
  "rum_checkpoint_path": "models/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors",
  "sdxl_model_path": "models/waiIllustriousSDXL_v160_text"
}
```

You can also paste absolute paths directly into the loader node.

Environment variables are also supported:

- `RUM_BASE_MODEL_PATH`
- `RUM_CHECKPOINT_PATH`
- `RUM_SDXL_MODEL_PATH`

## Basic Workflow

1. Add `RUM FLUX.2-klein Loader`.
2. Add `RUM FLUX.2-klein Sampler`.
3. Connect `pipeline` from the loader to the sampler.
4. Connect sampler `images` to `Preview Image` or `Save Image`.

A minimal API workflow is available at:

```text
examples/basic_workflow_api.json
```

Recommended first test settings:

```text
dtype=bfloat16
device=auto
sdxl_text_device=cpu
unload_sdxl_unet_vae=true
steps=20
guidance_scale=5
width=960
height=1024
max_sequence_length=200
text_encoder_out_layers=10,20,30
```

If VRAM is tight, reduce width and height first.

## Verify Installation

Run with the Python environment used by ComfyUI:

```bash
python scripts/check_install.py
```

This checks package versions, FLUX.2 diffusers imports, node imports, and optional model paths in `local_config.json`.

## Troubleshooting

### `No file named model_index.json`

Your FLUX.2-klein folder is incomplete. Run:

```bash
python scripts/download_models.py --write-local-config
```

or point `base_model_path` to a complete diffusers folder containing `model_index.json`.

### `cannot import name Flux2KleinPipeline`

Upgrade diffusers in ComfyUI's Python environment:

```bash
python -m pip install --upgrade "diffusers>=0.37.1"
```

### CUDA out of memory

Try:

- keep `sdxl_text_device=cpu`
- lower width and height
- restart ComfyUI
- run `RUM Unload Models`

### Slow first run

Expected. The first run loads FLUX.2-klein, the RUM checkpoint, and SDXL text encoders.

## Credits

- RUM method and upstream inference scripts: [RimoChan/RUM](https://github.com/RimoChan/RUM)
- FLUX.2-klein base: `black-forest-labs/FLUX.2-klein-base-4B`
- RUM preview weights: `rimochan/RUM-FLUX.2-klein-4B-preview`
- SDXL text encoder source used by the helper: `Ine007/waiIllustriousSDXL_v160`

---

# 简体中文说明

这是一组 ComfyUI 自定义节点，用来在 ComfyUI 里运行 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 `RUM-FLUX.2-klein-4B-preview` 模型。

RUM 的概念是把 SDXL 动漫模型的能力，蒸馏/转移到更新的模型架构上。这个节点包装了上游 diffusers 推理流程，并补上 RUM FLUX.2-klein 权重需要的额外 SDXL CLIP 条件分支。

> **授权 / 权限提醒**
>
> 这个 wrapper 参考并重现了 RUM 公开推理程序中的部分核心逻辑。建立此 wrapper 时，上游 RUM repo 没有附 LICENSE。因此在取得上游授权、或上游补上 LICENSE 前，不建议公开发布或提交到 Comfy Registry。
>
> 本 repo 不包含模型文件；模型文件仍遵守各自上游的授权条款。

## 功能

- 加载 FLUX.2-klein base model。
- 加载 RUM FLUX.2-klein preview checkpoint。
- 加载 SDXL text encoder/tokenizer，提供 RUM 额外需要的 CLIP 条件。
- 从 prompt 直接生成 ComfyUI 标准 `IMAGE` 输出。
- 提供 unload node 清除缓存中的 RUM pipeline。

## 当前限制

这一版是自包含的 diffusers pipeline 节点，不是原生 ComfyUI sampler stack。

目前还不支持：

- 输出可接 KSampler 的 model 对象
- latent 输入/输出工作流
- ControlNet / IP-Adapter
- ComfyUI 原生 LoRA 叠加
- negative prompt 输入

## 节点

- `RUM FLUX.2-klein Loader`
- `RUM FLUX.2-klein Sampler`
- `RUM Unload Models`

## 安装

把 repo clone 到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

然后重启 ComfyUI。

请注意：依赖要安装到「真正运行 ComfyUI 的 Python 环境」里。

## 下载模型

模型总共大约 24 GB：

```bash
python scripts/download_models.py --write-local-config
```

会下载：

| 用途 | 来源 | 默认文件夹 |
| --- | --- | --- |
| FLUX.2-klein base | `black-forest-labs/FLUX.2-klein-base-4B` | `models/FLUX.2-klein-base-4B` |
| RUM checkpoint | `rimochan/RUM-FLUX.2-klein-4B-preview` | `models/RUM-FLUX.2-klein-4B-preview` |
| SDXL text encoders | `Ine007/waiIllustriousSDXL_v160` | `models/waiIllustriousSDXL_v160_text` |

加上 `--write-local-config` 会自动生成 `local_config.json`。这个文件已被 Git ignore，不会被提交。

## 手动设置模型路径

如果你已经有模型，可以在 `nodes.py` 旁边建立 `local_config.json`：

```json
{
  "base_model_path": "models/FLUX.2-klein-base-4B",
  "rum_checkpoint_path": "models/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors",
  "sdxl_model_path": "models/waiIllustriousSDXL_v160_text"
}
```

你也可以直接在 Loader node 里粘贴绝对路径。

也支持环境变量：

- `RUM_BASE_MODEL_PATH`
- `RUM_CHECKPOINT_PATH`
- `RUM_SDXL_MODEL_PATH`

## 基本工作流

1. 新增 `RUM FLUX.2-klein Loader`。
2. 新增 `RUM FLUX.2-klein Sampler`。
3. 把 Loader 的 `pipeline` 接到 Sampler。
4. 把 Sampler 的 `images` 接到 `Preview Image` 或 `Save Image`。

最小 API workflow 在：

```text
examples/basic_workflow_api.json
```

第一次测试建议设置：

```text
dtype=bfloat16
device=auto
sdxl_text_device=cpu
unload_sdxl_unet_vae=true
steps=20
guidance_scale=5
width=960
height=1024
max_sequence_length=200
text_encoder_out_layers=10,20,30
```

如果爆 VRAM，先降低 width / height。

## 检查安装

用 ComfyUI 实际使用的 Python 执行：

```bash
python scripts/check_install.py
```

它会检查套件版本、FLUX.2 diffusers 导入、节点导入，以及 `local_config.json` 里的模型路径。

## 常见问题

### `No file named model_index.json`

代表 FLUX.2-klein base 文件夹不完整。请执行：

```bash
python scripts/download_models.py --write-local-config
```

或确认 `base_model_path` 指向完整的 diffusers 文件夹，里面要有 `model_index.json`。

### `cannot import name Flux2KleinPipeline`

代表 diffusers 版本太旧。请在 ComfyUI 的 Python 环境执行：

```bash
python -m pip install --upgrade "diffusers>=0.37.1"
```

### CUDA out of memory / 爆显存

可以试：

- 保持 `sdxl_text_device=cpu`
- 降低 width / height
- 重启 ComfyUI
- 执行 `RUM Unload Models`

### 第一次生成很慢

正常。第一次会加载 FLUX.2-klein、RUM checkpoint、SDXL text encoders。

## 致谢

- RUM 方法与上游推理程序：[RimoChan/RUM](https://github.com/RimoChan/RUM)
- FLUX.2-klein base：`black-forest-labs/FLUX.2-klein-base-4B`
- RUM preview weights：`rimochan/RUM-FLUX.2-klein-4B-preview`
- 下载脚本使用的 SDXL text encoder 来源：`Ine007/waiIllustriousSDXL_v160`
