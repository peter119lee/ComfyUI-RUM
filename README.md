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

# 繁體中文說明

這是一組 ComfyUI 自訂節點，用來在 ComfyUI 裡執行 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 `RUM-FLUX.2-klein-4B-preview` 模型。

RUM 的概念是把 SDXL 動漫模型的能力，蒸餾/轉移到更新的模型架構上。這個節點包裝了上游 diffusers 推理流程，並補上 RUM FLUX.2-klein 權重需要的額外 SDXL CLIP 條件分支。

> **授權 / 權限提醒**
>
> 這個 wrapper 參考並重現了 RUM 公開推理程式中的部分核心邏輯。建立此 wrapper 時，上游 RUM repo 沒有附 LICENSE。因此在取得上游授權、或上游補上 LICENSE 前，不建議公開發布或提交到 Comfy Registry。
>
> 本 repo 不包含模型檔；模型檔仍遵守各自上游的授權條款。

## 功能

- 載入 FLUX.2-klein base model。
- 載入 RUM FLUX.2-klein preview checkpoint。
- 載入 SDXL text encoder/tokenizer，提供 RUM 額外需要的 CLIP 條件。
- 從 prompt 直接產生 ComfyUI 標準 `IMAGE` 輸出。
- 提供 unload node 清除快取中的 RUM pipeline。

## 目前限制

這版是自包含的 diffusers pipeline 節點，不是原生 ComfyUI sampler stack。

目前還不支援：

- 輸出可接 KSampler 的 model 物件
- latent 輸入/輸出工作流
- ControlNet / IP-Adapter
- ComfyUI 原生 LoRA 疊加
- negative prompt 輸入

## 節點

- `RUM FLUX.2-klein Loader`
- `RUM FLUX.2-klein Sampler`
- `RUM Unload Models`

## 安裝

把 repo clone 到 ComfyUI 的 `custom_nodes` 目錄：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peter119lee/ComfyUI-RUM.git
cd ComfyUI-RUM
python -m pip install -r requirements.txt
```

然後重啟 ComfyUI。

請注意：依賴要安裝到「真正執行 ComfyUI 的 Python 環境」裡。

## 下載模型

模型總共大約 24 GB：

```bash
python scripts/download_models.py --write-local-config
```

會下載：

| 用途 | 來源 | 預設資料夾 |
| --- | --- | --- |
| FLUX.2-klein base | `black-forest-labs/FLUX.2-klein-base-4B` | `models/FLUX.2-klein-base-4B` |
| RUM checkpoint | `rimochan/RUM-FLUX.2-klein-4B-preview` | `models/RUM-FLUX.2-klein-4B-preview` |
| SDXL text encoders | `Ine007/waiIllustriousSDXL_v160` | `models/waiIllustriousSDXL_v160_text` |

加上 `--write-local-config` 會自動產生 `local_config.json`。這個檔案已被 Git ignore，不會被提交。

## 手動設定模型路徑

如果你已經有模型，可以在 `nodes.py` 旁邊建立 `local_config.json`：

```json
{
  "base_model_path": "models/FLUX.2-klein-base-4B",
  "rum_checkpoint_path": "models/RUM-FLUX.2-klein-4B-preview/model-checkpoint-608000.safetensors",
  "sdxl_model_path": "models/waiIllustriousSDXL_v160_text"
}
```

你也可以直接在 Loader node 裡貼絕對路徑。

也支援環境變數：

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

第一次測試建議設定：

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

## 檢查安裝

用 ComfyUI 實際使用的 Python 執行：

```bash
python scripts/check_install.py
```

它會檢查套件版本、FLUX.2 diffusers 匯入、節點匯入，以及 `local_config.json` 裡的模型路徑。

## 常見問題

### `No file named model_index.json`

代表 FLUX.2-klein base 資料夾不完整。請執行：

```bash
python scripts/download_models.py --write-local-config
```

或確認 `base_model_path` 指向完整的 diffusers 資料夾，裡面要有 `model_index.json`。

### `cannot import name Flux2KleinPipeline`

代表 diffusers 版本太舊。請在 ComfyUI 的 Python 環境執行：

```bash
python -m pip install --upgrade "diffusers>=0.37.1"
```

### CUDA out of memory / 爆顯存

可以試：

- 保持 `sdxl_text_device=cpu`
- 降低 width / height
- 重啟 ComfyUI
- 執行 `RUM Unload Models`

### 第一次生成很慢

正常。第一次會載入 FLUX.2-klein、RUM checkpoint、SDXL text encoders。

## 致謝

- RUM 方法與上游推理程式：[RimoChan/RUM](https://github.com/RimoChan/RUM)
- FLUX.2-klein base：`black-forest-labs/FLUX.2-klein-base-4B`
- RUM preview weights：`rimochan/RUM-FLUX.2-klein-4B-preview`
- 下載腳本使用的 SDXL text encoder 來源：`Ine007/waiIllustriousSDXL_v160`
