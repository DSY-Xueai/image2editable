<div align="center">

# image2editable

中文 | [English](README_EN.md)

**图片、PDF、图片版 PPTX → 可编辑 PPTX**

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

![image2editable 介绍](docs/images/readme-intro.png)

image2editable 用于把图片、PDF 和截图式 PPT 转换成可以继续修改的 PowerPoint。它适合课件截图、设计稿、报告页面和图片化幻灯片，减少从零复刻整页版式的工作。

转换后，可以直接在 PowerPoint 中修改识别出的文字、移动拆分出的视觉元素，并继续调整页面内容；处理混合 PPTX 时，原本可编辑的内容会保留在文件中。

---

## 转换效果演示：

|                             原图                             |                      转换后的可编辑效果                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![原图 1](docs/images/demo-source-1.png) | ![转换结果 1](docs/images/demo-result-1.png) |
| ![原图 2](docs/images/demo-source-2.png) | ![转换结果 2](docs/images/demo-result-2.png) |
| ![原图 3](docs/images/demo-source-3.png) | ![转换结果 3](docs/images/demo-result-3.png) |

**若是单独一张图片，为获得最佳的 16:9 PPT 视觉效果，建议输入图片采用 16:9 比例。**

## 特点

| 能力 | 说明 |
|------|------|
| 可编辑文字 | 尽量恢复为 PowerPoint 原生文本框，可在输出文件中直接修改。 |
| 可移动视觉元素 | 将可独立处理的视觉元素拆分为透明图片组件，便于移动或替换。 |
| 混合 PPTX 保护 | 未参与重建的原生文字、形状、表格、图表、备注和层级顺序保持不变。 |
| 多种输入 | 支持图片、图片目录、PDF、图片版 PPTX 和混合 PPTX。 |
| 批量转换 | 多张图片或多页文档按顺序生成多页 PPTX。 |
| 质量门禁 | 每页最多进行五轮重修，质量无改善时提前停止；只有通过质量门禁的重建结果才标记为可编辑转换完成。 |

## 使用前了解

- 这是把**已有页面**重建为可继续编辑 PPT 的工具，不是根据文章或大纲生成全新演示文稿。
- **⚠️ 复杂视觉元素通常会以可移动图片组件保留**，不能保证其内部元素都能恢复为原生 PowerPoint 形状。
- **🔒 Host Agent 模式可能把诊断图交由当前宿主服务处理**；处理敏感文件前，请确认宿主服务的数据策略符合要求。

## 快速上手

### 使用 **Skills CLI** 安装

```bash
npx skills add DSY-Xueai/image2editable --skill image-to-ppt
```

### **让 Agent 自动安装**

```bash
请从 https://github.com/DSY-Xueai/image2editable 安装 <image-to-ppt> skill。
```

安装后，可直接向支持视觉、文件读取和工具调用的 Agent 描述需求。在 Codex 中使用 `$image-to-ppt`，在 Claude Code 中使用 `/image-to-ppt`。图片、PDF 和 `.pptx` 可以直接粘贴或附加到对话框，也可以提供本地路径：

```text
# Codex
$image-to-ppt 把 input.pptx 转成可编辑 PPTX，保留没有命中的原生对象。
$image-to-ppt 把 input.png 转成可编辑 PPTX。
$image-to-ppt 把 <input.pdf> 转成可编辑 PPT。
# Claude Code
/image-to-ppt 把 input.pptx 转成可编辑 PPTX，保留没有命中的原生对象。
/image-to-ppt 把 input.png 转成可编辑 PPTX。
/image-to-ppt 把 <input.pdf> 转成可编辑 PPT。
```

完整仓库或已安装 `image2editable` 的环境会由 Skill 自动准备固定依赖、OCR 和 runtime 模型，并通过当前 Host Agent 完成转换。仅安装 standalone Skill 时，需要预先设置 `SAM2_MODEL`、`LAMA_MODEL` 和 `GROUNDING_DINO_MODEL` 的绝对本地路径；路径缺失时会列出缺失项并停止。

## 项目结构

```
image2editable/
├── .claude-plugin/            # Claude Code 插件清单
│   └── plugin.json
├── .github/                   # CI、Issue 表单和 PR 模板
├── docs/
│   └── images/                # README 图片资源
├── image2editable/            # 统一 CLI、运行时和转换模块
├── scripts/                   # 识别、重建和 PPTX/PSD 组装模块
├── skills/
│   ├── image-to-ppt/          # 可安装的图片转 PPT Skill
│   └── image-to-psd/          # 兼容的图片转 PSD Skill
├── tests/                     # 自动化测试
├── third_party/
│   └── licenses/              # 第三方许可证资料
├── .gitignore
├── CITATION.cff               # 引用信息
├── image_to_ppt.py            # 旧版图片专用技术路线，非当前推荐入口
├── image_to_psd.py            # 兼容的图片转 PSD 入口
├── LICENSE                    # MIT 许可证
├── pyproject.toml             # Python 包与 CLI 配置
├── README.md                  # 中文说明
├── README_EN.md               # English documentation
├── requirements.txt           # 核心依赖
└── THIRD_PARTY_NOTICES.md     # 第三方依赖与许可证说明
```

## 已知问题

- **⚠️ 复杂页面建议人工复核。** 艺术字、密集表格、渐变和复杂插画可能无法逐像素还原；请在交付前检查文字、组件位置和页面布局。
- 图片中的文字越清晰、背景越规整，重建通常越可靠；艺术字、密集表格、渐变和复杂插画不保证逐像素一致。
- **💳 Host Agent 会消耗模型的 Token / 上下文额度。** 复杂页面可能经过多轮诊断与重修，实际消耗取决于所用 Agent、模型和页面复杂度。
- **⏱️ 多页 PDF、复杂页面和高分辨率图片耗时较长。** 每页都会经过 OCR、视觉拆分、重建与质量检查，最多可进行 5 轮重修；Host 模式还需要等待 Agent 完成视觉判断。

## 支持的输入

| 输入 | 使用建议 | 说明 |
|------|----------|------|
| 图片或图片目录 | Skill | 支持 PNG、JPG/JPEG、BMP、TIFF/TIF、WebP；目录只扫描第一层图片。 |
| PDF | Skill | 按页渲染并按顺序重建为多页 PPTX。 |
| 图片版 PPTX、混合 PPTX | Skill | 会识别可处理的图片页；未命中的原生对象保持不变。 |

其他第三方依赖及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，引用信息见 [CITATION.cff](CITATION.cff)。

## 许可证

MIT
