---
name: image-to-psd
description: 将一张或多张图片转换为经过严格质量校验的分层 PSD；自动准备运行环境，通过当前 Agent 执行转换。输出修复背景、独立透明视觉组件和可编辑 Photoshop 文字图层。仅支持图片输入，不用于 PDF 或 PPTX。
---

# Image to PSD

把图片重建为分层 PSD。文字只由可编辑文字图层贡献一次；视觉组件和背景不得残留文字像素。质量检查未通过时继续针对性修复并复检，不把整页图片伪装成分层结果。

全部文字包括艺术字均须可编辑，保留原有曲线、描边及多色；不得用文字截图、转曲轮廓或透明文字覆盖冒充。复用有效识别和组件资产，避免重复推理；只有实际渲染和编辑验收通过才能作为成品交付。

局部 OCR 找回文字后，先验证源图、manifest、资产哈希和文字增量依赖。可证明安全时只更新受影响像素及背景，保留其余有效组件；并行路径也必须使用新增文字清理后的图像。重建范围覆盖实际文字清理边缘，不能仅依据原 OCR 框。

## 输入与授权

识别阶段复用 text-context-cache 中匹配像素、语言和 OCR 实现的整行复核结果，避免重复处理相同冲突；不同栏位的文字不得因宽检测结果而丢失独立位置与样式。共享 OCR 的 words/runs 元数据不代表 PSD 已具备对应的艺术字渲染能力，必须核验实际文字图层效果。

- 仅支持 PNG、JPEG、BMP、TIFF 和 WebP。
- 单图输出一个 `.psd`；多图输出到目录，同名文件使用稳定序号区分。
- 每个 PSD 包含修复背景、按 z-order 排列的透明视觉组件和可编辑文字图层。
- PSD 写入依赖已授权的 Aspose.PSD。模型推理前必须设置 `ASPOSE_PSD_LICENSE`；授权缺失或无效时立即停止。

Windows PowerShell：

```powershell
$env:ASPOSE_PSD_LICENSE="C:\path\to\Aspose.PSD.lic"
```

Linux/macOS：

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

授权文件、模型权重、OCR 缓存和运行产物都不存放在此 skill 中。

## 环境准备与图片兼容入口

转换前必须阅读并执行 [自动环境准备](references/setup.md)。完整仓库和仅安装 Skill 在 Windows、macOS、Linux 都自动准备缺少的 Python、Git、项目 Runtime、依赖、OCR 和模型。不得为依赖或模型安装向用户询问确认；遵循宿主实际审批与权限限制。Windows 新安装优先 D 盘，再选其他非 C 本地磁盘，仅在不存在其他本地磁盘时使用 C 盘；macOS/Linux 优先其他已挂载的本地磁盘，否则使用用户目录。使用 `scripts/skill_environment.py` 统一环境、模型、下载缓存与临时目录。已有可用环境和模型继续复用。

准备后默认使用下文产品 Runtime 的 Agent 流程；下面的 standalone CLI 是图片兼容入口。两者都使用自动安装的模型，不要求使用者手动配置 `SAM2_MODEL`、`LAMA_MODEL` 或 `GROUNDING_DINO_MODEL`。推理不会下载模型或回退 Hugging Face cache；准备阶段先校验 runtime receipt，已有显式模型路径须满足固定身份约束。LaMa 缺失或初始化失败时停止该无效路径并修复环境，不降低修复质量。

检查当前设备后再运行：

```bash
python -c "import sys, torch; print({'platform': sys.platform, 'cuda': torch.cuda.is_available(), 'rocm': torch.version.hip})"
```

CPU 仍使用完整模型和相同质量门，速度会明显慢于 GPU。macOS 在真实 Apple Silicon 回归完成前不自动把 MPS 设为默认。

从 skill 根目录运行 module，不要直接执行脚本文件：

```bash
cd skills/image-to-psd
python -m scripts.image_to_psd input.png
python -m scripts.image_to_psd input.png -o output.psd
python -m scripts.image_to_psd img1.png img2.png -o psd-output
python -m scripts.image_to_psd images/ -o psd-output --lang en
```

standalone CLI 只负责图片重建，不接受 `--agent-provider`。它先完成全部页面的严格准备，再发布 PSD；任一页面失败时不会留下部分输出。

## 产品 Runtime

完整仓库或已安装的 `image2editable` 只支持 `host` Provider，使用统一的组件动作、最多 5 批修复和相同质量门。

完整仓库中缺少 PSD 依赖时，在仓库根目录安装对应 extra：

```bash
python -m pip install -e ".[psd]"
```

仅已安装 `image2editable` distribution、没有仓库源码时，直接安装同一 PSD writer 依赖，不对调用者的当前项目执行 editable install：

```bash
python -m pip install "aspose-psd>=26.5.0"
```

随后以非交互方式安装并校验固定的 SAM、LaMa 和 DINO runtime：

```bash
image2editable models install runtime --yes
image2editable doctor
```

`host` 直接使用当前支持视觉、本地文件读取、工具调用和结构化 JSON 的宿主，不探测、下载或要求配置其他组件决策模型。处理敏感文件前，确认宿主服务的数据策略符合要求。

Host 模式先准备 Run，再推进到 `awaiting_agent`：

```bash
image2editable prepare input.png -o output.psd \
  --run-dir runs/psd-job --format psd --agent-provider host
image2editable run execute runs/psd-job
image2editable agent next runs/psd-job
image2editable agent record runs/psd-job --plan response.json
image2editable run execute runs/psd-job
```

第一次 `agent next` 返回视觉能力 challenge。必须实际查看 `image_path`，记录观察到的 `shape`、`color` 和 `count`，不能从文件名或 metadata 猜测。之后每轮只查看 request 中按顺序列出的 `review_evidence`，同时核验完整 request、hash、组件图、候选和冻结状态；`quality-report.json` 作为质量证据读取，不能当图片发送。

计划必须绑定当前 `request_sha256`。每个 action 只使用请求组件图中的 ID，并限定为现有十四类动作：`accept`、`discard`、`merge`、`split`、`expand`、`shrink`、`retry_with_box`、`retry_with_points`、`attach_text`、`suppress_text`、`collapse_to_parent`、`rebuild_background`、`absorb_residual`、`absorb_into_parent`。Agent confidence 不能放宽硬失败。

若绑定的 `unexplained-mask.png` 中有经验证的结构碎片，可用 `absorb_residual` 并入相关候选；请求图中的 inactive visual 有对应来源证据时，该动作仅恢复绑定残差，不恢复整个已停用复合对象，也不调用 SAM。随后按需 `rebuild_background` 并重新验证。只有残差证据不足以确定结构时才重新分割，不得将残差归为背景来消除违规。

## 质量与失败

卡片底色连接多个独立纯色图形时，`split` 可复用原图色块边界拆分，保留全部像素，不调用 SAM。`parts` 对应实际完整单元并包含底色，不能按期望数量任意切块。文字框外的标点应修复 OCR 字形范围，不能作为图形残差吸收。

- 每张图片独立判断，不能跨图片套用拆分结果。
- 每个视觉组件应是可独立移动的最小完整单元，不得残缺、重叠、吸收相邻对象或只保留阴影碎片。
- 已通过组件立即冻结并复用；检测无进展和重复产物，停止无效策略并切换针对性修复，不为耗尽轮数重复执行。有实际进展的任务不因总耗时较长而放弃。
- `rebuild_background.margin_ratio` 使用能覆盖残影且不触及相邻结构的最小值，不固定写死。
- `unexplained_visual_residual` 必须由 active visual owner 覆盖；不能用 `accept`、`discard` 或归为背景来消除违规。
- 可靠 OCR 文字必须全部写为可编辑文字图层，并且只能出现一次。
- `preserved_with_warning` 是内部未完成状态，不是分层交付。当前运行时仍有修复周期耗尽后无法继续的路径；须解决该交付能力缺口并补通用回归，不能将低质量结果标为成功或宣称已具备发布条件。
- standalone 质量异常包含指标和诊断路径，由宿主检查 `source.png`、`ownership.png`、`reconstructed.png` 和 `report.json` 并修复。诊断不能代替最终文件，不把修复责任交给使用者；不得放宽门禁、删内容、伪造通过或回退为整页图片。用户主动取消时停止处理并保留恢复依据。
