<div align="center">

# image2editable

[中文](README.md) | English

**Images, PDFs, and image-based PPTX → Editable PPTX**

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

![image2editable workflow](docs/images/readme-workflow-en.png)

image2editable turns images, PDFs, and screenshot-based PowerPoint slides into PowerPoint files that can be edited again. It is designed for courseware screenshots, design mockups, report pages, and image-based slides, reducing the work of recreating a page from scratch.

After conversion, you can edit recovered text, move separated visual elements, and keep refining the page in PowerPoint. When processing a mixed PPTX, existing editable content is retained in the file.

---

## Conversion examples

| Source | Editable result |
| :----: | :-------------: |
| ![Source 1](docs/images/demo-source-1.png) | ![Conversion result 1](docs/images/demo-result-1.png) |
| ![Source 2](docs/images/demo-source-2.png) | ![Conversion result 2](docs/images/demo-result-2.png) |
| ![Source 3](docs/images/demo-source-3.png) | ![Conversion result 3](docs/images/demo-result-3.png) |

**For the best visual result in a 16:9 PowerPoint deck, use a 16:9 input image when converting a single image.**

## Features

| Capability | What it does |
|------------|--------------|
| Editable text | Recovers readable text as native PowerPoint text boxes whenever possible. |
| Movable visual elements | Separates independently processable visuals into transparent image components that can be moved or replaced. |
| Mixed PPTX preservation | Native text, shapes, tables, charts, notes, and z-order that are not rebuilt remain unchanged. |
| Multiple inputs | Supports images, image directories, PDFs, image-based PPTX files, and mixed PPTX files. |
| Batch conversion | Converts multiple images or document pages into a multi-slide PPTX in order. |
| Quality protection | Makes up to five repair rounds per page; pages that do not pass checks retain their source content with a warning. |

## Before you start

- This is a tool for rebuilding **existing pages** into editable PowerPoint files. It does not create a new presentation from an article or outline.
- **⚠️ Complex visuals are usually kept as movable image components.** Their internal elements are not guaranteed to become native PowerPoint shapes.
- **🔒 Host Agent mode may send diagnostic images to the current host service.** Choose Local Agent for sensitive files.

## Quick start

### Runtime environment

Prefer a correctly installed hardware-accelerated environment on the current platform that passes `doctor` and the device preflight; do not leave an already working environment solely because of a WSL recommendation. Windows and Linux use the existing PyTorch device interface: CUDA is used when PyTorch reports it as available, while ROCm uses PyTorch's compatible device interface. macOS keeps the current supported device selection; MPS will not become a new automatic default until real Apple Silicon regression testing is complete.

CPU still runs the full model and the same quality gates, including SAM 2.1 Large; it does not switch to a lightweight segmentation model, but it is significantly slower than a working hardware-accelerated environment. Runtime depends on the platform, environment, and input complexity, so the project does not promise results for a specific GPU model or a uniform speedup factor.

### Install with the **skills CLI**

```bash
npx skills add DSY-Xueai/image2editable --skill image-to-ppt
```

### **Let an Agent install it**

```text
Install the <image-to-ppt> skill from https://github.com/DSY-Xueai/image2editable.
```

After installation, describe the task to an Agent that supports vision, file access, and tool calls. Images, PDFs, and `.pptx` files can be pasted or attached in the chat, or provided as local paths:

```text
# Codex
$image-to-ppt Convert input.pptx to an editable PPTX and preserve native objects that are not selected for reconstruction.
$image-to-ppt Convert input.png to an editable PPTX.
$image-to-ppt Convert <input.pdf> to an editable PPTX.

# Claude Code
/image-to-ppt Convert input.pptx to an editable PPTX and preserve native objects that are not selected for reconstruction.
/image-to-ppt Convert input.png to an editable PPTX.
/image-to-ppt Convert <input.pdf> to an editable PPTX.
```

> **💡 Tip:** Use the Skill when Codex, Claude Code, or a similar host is available. Use Local CLI when images or PDFs need to stay on your own machine.

### Local CLI

Deploy and start a local service with **vision capability** first. The service must expose an **OpenAI-compatible `/v1/chat/completions` API**, accept image input, and return JSON. vLLM, LM Studio, or Ollama with its OpenAI-compatible interface enabled can be used.

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
cd image2editable
pip install .
```

For first-time setup, copy the template in the project root and **fill in your own service settings**.

```powershell
Copy-Item .env.example .env
```

Edit `.env`: set `IMAGE2EDITABLE_LOCAL_BASE_URL` to the **service address** and `IMAGE2EDITABLE_LOCAL_MODEL` to the **model name exposed by the service**. Leave `IMAGE2EDITABLE_LOCAL_API_KEY` empty when the service does not require a key.

After configuration, use `--agent-provider local` to call the local service:

```bash
# Image → editable PPTX
image2editable convert input.png -o output.pptx --slide-size 16:9 --agent-provider local

# PDF → editable PPTX
image2editable convert input.pdf -o output.pptx --slide-size 16:9 --agent-provider local
```

| Option | Default | Purpose |
|--------|---------|---------|
| `sources` | Required | Input to convert: image(s), an image directory, one PDF, or one PPTX. Document inputs cannot be mixed with other sources. |
| `-o, --output` | Derived from input name | Sets the output file. With `--slide-size both`, it is used as the output base name. |
| `--lang` | `ch` | OCR language; common values are `ch` and `en`. |
| `--agent-provider` | `host` | Selects the processing mode: `host` works with the current Agent/Skill, while `local` uses the configured local vision service. |
| `--slide-size` | `both` | Selects output layout: `original` keeps the input ratio, `16:9` makes widescreen slides, and `both` creates both. |
| `--run-dir` | Generated automatically | Sets a run directory so you can inspect progress, resume an unfinished task, or troubleshoot. |

In the command above, `input.pdf` is `sources`, equivalent to:

```text
sources = ["input.pdf"]
```

## Choose a processing mode

| Mode | Best for | How it works |
|------|----------|--------------|
| Host Agent | Users already working in Codex, Claude Code, or a similar host and who want Agent assistance with page-structure decisions. | The Skill gives diagnostic material to the current host Agent; **the CLI does not call a fixed cloud AI API directly**. |
| Local Agent | Users who have deployed their own local vision-model service and want to process images or PDFs on their own device. | Uses the configured OpenAI-compatible local service. |

## Project layout

```
image2editable/
├── .claude-plugin/            # Claude Code plugin manifest
│   └── plugin.json
├── docs/
│   └── images/                # README image assets
├── image2editable/            # Unified CLI, runtime, and conversion modules
├── scripts/                   # Recognition, reconstruction, and PPTX/PSD assembly modules
├── skills/
│   ├── image-to-ppt/          # Installable image-to-PPT Skill
│   └── image-to-psd/          # Compatible image-to-PSD Skill
├── tests/                     # Automated tests
├── third_party/
│   └── licenses/              # Third-party license materials
├── .env.example               # Local Agent configuration example
├── .gitignore
├── CITATION.cff               # Citation information
├── image_to_ppt.py            # Compatible image-to-PPTX entry point
├── image_to_psd.py            # Compatible image-to-PSD entry point
├── LICENSE                    # MIT license
├── pyproject.toml             # Python package and CLI configuration
├── README.md                  # Chinese documentation
├── README_EN.md               # English documentation
├── requirements.txt           # Core dependencies
└── THIRD_PARTY_NOTICES.md     # Third-party dependency and license notices
```

## Known limitations

- **⚠️ Review complex pages manually.** Decorative text, dense tables, gradients, and complex illustrations may not be restored pixel for pixel. Check text, component positions, and layout before delivery.
- Clear text and regular backgrounds generally reconstruct more reliably. Decorative text, dense tables, gradients, and complex illustrations are not guaranteed to match pixel for pixel.
- **💳 Host Agent consumes the current model's token/context allowance.** Complex pages may require several diagnostic and repair rounds; actual usage depends on the Agent, model, and page complexity. Local mode does not use project tokens, but consumes inference resources from the local model service. CPU inference may be slow.
- **⏱️ Multi-page PDFs, complex pages, and high-resolution images take longer.** Each page goes through OCR, visual separation, reconstruction, and quality checks, with up to five repair rounds. Host mode also waits for Agent visual decisions.

## Supported inputs and common options

| Input | Recommended route | Notes |
|-------|-------------------|-------|
| Images or an image directory | Skill or Local CLI | PNG, JPG/JPEG, BMP, TIFF/TIF, and WebP are supported. Directories scan images in the first level only. |
| PDF | Skill or Local CLI | Pages are rendered and rebuilt into a multi-slide PPTX in order. |
| Image-based or mixed PPTX | Skill recommended | Processable image pages are selected for reconstruction; unmatched native objects stay unchanged. |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party dependencies and licenses, and [CITATION.cff](CITATION.cff) for citation information.

## License

MIT
