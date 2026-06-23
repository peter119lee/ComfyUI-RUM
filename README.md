# ComfyUI-RUM

ComfyUI-RUM 是 [RimoChan/RUM](https://github.com/RimoChan/RUM) 的 ComfyUI native 节点适配。本仓库只提供 ComfyUI 适配代码和 workflow，不包含模型权重，也不把上游训练仓库文件直接合并进来。

## 当前状态

- 上游 `RimoChan/RUM` 当前默认分支是 `slave`，2026-06-23 的 HEAD 是 `1662918`。本仓库和上游训练仓库没有共同祖先，不能直接 `merge` 或 `cherry-pick`；这里同步的是推理契约、默认参数、权重文件名和验证样本。
- 主线坚持 ComfyUI native diffusers-match 路线，不提供 diffusers pipeline wrapper 节点，也不要求用户在 ComfyUI 环境安装 diffusers。
- T2I 推荐权重：`model-checkpoint-1158000.safetensors`。
- 编辑推荐权重：`model-checkpoint-1202000.safetensors`。
- 新增 `RUMFlux2NativeMatchReferenceEncode`，用于把编辑参考图编码成专用 `RUM_REFERENCE_LATENTS`，只给 diffusers-match 编辑路径使用。

功能可用和像素级一致是两件事。当前 T2I 与 edit workflow 已在 `I:\ComfyUI-aki-v1.6\ComfyUI` 环境中生成成功；严格 `pixel_equal=true` 需要上游 reference 环境、ComfyUI 环境和模型精度完全一致。已知会影响逐像素结果的因素包括：

- `torch` / `transformers` 版本差异会改变 Qwen hidden states，采样会从 text embedding 阶段开始分歧。
- FLUX.2 VAE 使用 BF16 或 FP32 权重、以及 VAE 在 CPU/GPU 上运行，都会改变最终 RGB/PNG 的舍入结果。
- ComfyUI 启动参数带 `--cpu-vae` 时，图片内容可以正常生成，但不适合做严格像素对齐。

## 为什么需要 diffusers-match 路径

RUM 不是普通 FLUX.2 LoRA。严格贴近上游 `Flux2KleinPipeline` 需要同时对齐这些细节：

- Qwen hidden states 层：`10,20,30`。
- positive 条件长度：Qwen `200` tokens + SDXL teacher CLIP `77` tokens。
- negative prompt 默认为空字符串，并走同一 Qwen `200` + SDXL teacher `77` 的 text encode 契约。
- SDXL teacher CLIP 使用从上游 `waiNSFWIllustrious_v140.safetensors` 对应导出的 `waiNSFWIllustrious_v140_clip_l.safetensors` 和 `waiNSFWIllustrious_v140_clip_g.safetensors`。
- 初始 noise 复刻 diffusers CPU generator + BF16。
- scheduler 使用 Flux2KleinPipeline 的 sigma/time-shift 策略。
- VAE decode 使用 `RUMFlux2NativeMatchVAEDecode`，保存前接 `RUMRoundImageForSave`。
- 编辑参考图按上游 `image=...` 路径处理：校验尺寸和宽高比，最大面积 `1024*1024`，裁到 `vae_scale_factor*2` 倍数，VAE encode 后做 FLUX.2 BN 标准化，采样时把 reference tokens 拼到 denoise tokens 后面，reference T 坐标是 `10,20,...`。

## Workflows

### T2I strict workflow

```text
examples/diffusers_match_workflow_api.json
```

默认对齐上游 `推理.py` 第 1 张 T2I 样本：

```text
checkpoint=model-checkpoint-1158000.safetensors
prompt=1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
negative_prompt=
seed=1
steps=20
cfg=9
width=960
height=1152
```

上游当前 4 个 T2I prompt/seed：

```text
1. seed=1  1girl, kisaki (blue archive), holding baozi, eating, indoors, momoko (momopoco)
2. seed=2  1girl, momoi (blue archive), typing on keyboard, computer, blue necktie, open coat, multicolored coat, angry, indoors, ameto yuki
3. seed=3  1girl, yuuka (blue archive), holding cup, white jacket, black jacket, blue necktie, indoors, fuzichoco
4. seed=4  1girl, azusa (blue archive), holding pizza, eating, indoors, chen bin
```

### Edit strict workflow

```text
examples/diffusers_match_edit_workflow_api.json
```

默认对齐上游 `推理.py` 第 1 张编辑样本：

```text
checkpoint=model-checkpoint-1202000.safetensors
reference=rum_reference.jpg
prompt=将服装改为school uniform, short sleeves
negative_prompt=
seed=1
steps=20
cfg=9
width=656
height=1200
```

`rum_reference.jpg` 是上游仓库 `img/抓人.jpg` 的同图 ASCII 文件名副本。API workflow 使用 ASCII 文件名，避免 Windows/PowerShell 环境把中文文件名转成问号；UI 中也可以重新选择同一张上游参考图。

上游当前 6 个编辑 prompt/seed：

```text
1. seed=1  将服装改为school uniform, short sleeves
2. seed=2  add twintails
3. seed=3  1girl, remove shoes, bare foot
4. seed=4  1girl, change background to beach
5. seed=6  add choker
6. seed=5  改为1girl, fuzichoco
```

不同 ComfyUI 发行版的模型下拉名可能不同。如果 API validation 失败，请先读 `/object_info/RUMFlux2LoadNativeModel`，把 workflow 里的 `rum_checkpoint_name` 改成你本机实际显示的名称。

## 模型文件

严格 diffusers-match workflow 需要：

| 用途 | 文件 | ComfyUI 位置 |
| --- | --- | --- |
| RUM T2I checkpoint | `model-checkpoint-1158000.safetensors` | `models/diffusion_models/` |
| RUM edit checkpoint | `model-checkpoint-1202000.safetensors` | `models/diffusion_models/` |
| Qwen text encoder | `qwen_3_4b.safetensors` | `models/text_encoders/` |
| FLUX.2 VAE | `flux2-vae.safetensors` | `models/vae/` |
| SDXL teacher CLIP-L | `waiNSFWIllustrious_v140_clip_l.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G | `waiNSFWIllustrious_v140_clip_g.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-L exact HF 目录 | `waiNSFWIllustrious_v140_clip_l_dir/config.json` + `model.safetensors` | `models/text_encoders/` |
| SDXL teacher CLIP-G exact HF 目录 | `waiNSFWIllustrious_v140_clip_g_dir/config.json` + `model.safetensors` | `models/text_encoders/` |

下载默认 T2I 路径：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip
```

同时下载编辑 checkpoint：

```bash
python scripts/download_models.py --comfy-root /path/to/ComfyUI --include-teacher-clip --include-edit-checkpoint
```

检查安装：

```bash
python scripts/check_install.py --comfy-root /path/to/ComfyUI
```

`DualCLIPLoader` 使用 flat `.safetensors` 文件；exact text encode 会从同名前缀的 `*_clip_l_dir` 和 `*_clip_g_dir` HF 目录加载 `transformers` 模型。`scripts/download_models.py --include-teacher-clip` 会同时准备这两套文件。

## 安装

把仓库放到 ComfyUI custom nodes：

```text
ComfyUI/custom_nodes/ComfyUI-RUM
```

安装依赖后重启 ComfyUI：

```bash
pip install -r requirements.txt
```

## 像素验证

严格验证时不要只看图。建议按这个顺序定位第一处差异：

```text
text embedding -> noise -> sigmas -> raw noise -> latent -> VAE RGB -> PNG
```

目标指标：

```text
pixel_equal=true
max_abs=0
mean_abs=0
rmse=0
```

严格像素验证要求 VAE 在 GPU 上运行。启动 ComfyUI 时不要带 `--cpu-vae`，否则 CPU/GPU 浮点路径会导致最终 PNG 产生小幅像素差。

## Credit

- [RimoChan/RUM](https://github.com/RimoChan/RUM)：原始 RUM 项目、模型、训练和 diffusers 推理参考。
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)：节点系统和执行环境。
- [Hugging Face diffusers](https://github.com/huggingface/diffusers)：FLUX.2-Klein pipeline、scheduler、VAE 参考实现。

## License note

本仓库不包含模型权重。RUM 上游项目在本适配开始时没有明确 LICENSE；模型和代码的再分发请遵守各自上游规则。
