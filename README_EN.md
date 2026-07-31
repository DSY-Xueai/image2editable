<div align="center">

# image2editable

[中文](README.md) | English

**Images → Editable PPTX / Layered PSD**

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

</div>

Convert PowerPoint screenshots, page captures, or design images into separate background, foreground component, and text layers, then export them as editable PPTX or layered PSD files.

---

## Demo

> Input image | Multiple images are also supported
<img width="2154" height="1127" alt="image" src="https://github.com/user-attachments/assets/867e95ba-a7ba-4966-8fd4-a3208a5fc924" />

> High-confidence PPTX layers keep foreground elements movable and text boxes editable. When layer quality is uncertain, the visual background is preserved while text remains editable.
>
> For the best visual results in a 16:9 PPT, using a 16:9 input image is recommended.
<img width="2022" height="1058" alt="image" src="https://github.com/user-attachments/assets/cf86c0dc-515e-4d86-a6fb-a42f084518fd" />

---

## Core Features

| Feature | Description |
|---------|-------------|
| Background repair | PPTX uses OpenCV for small or narrow masks and LaMa for large or deep masks; PSD uses two-pass background modeling and inpainting |
| Foreground separation | PPTX uses Grounding DINO semantic proposals and SAM 2.1 segmentation; PSD uses differences, edges, and connected components |
| OCR text reconstruction | Detects text and estimates font size, color, weight, and alignment |
| PPTX export | High-confidence pages use independent transparent components and editable text boxes; uncertain pages use a text-clean fidelity background with editable text; outputs original-aspect-ratio and 16:9 versions by default |
| Resource protection | Processes heavy pages serially and isolates OCR, LaMa, DINO, SAM, and full-page visual phases in sequential subprocesses; SAM2.1 Large uses a batch size of one by default |
| PSD export | Generates a layered PSD with a background layer, foreground pixel layers, and Photoshop text layers |
| Batch processing | Accepts multiple images or a directory; PPTX files are combined into multiple slides, while PSD exports one file per image |

---

## Quick Start

### Requirements

- Python 3.10–3.12 (the upper limit comes from the NumPy/Pillow constraints of `simple-lama-inpainting 0.1.2`)
- `torch>=2.5.1`, `torchvision>=0.20.1`, `transformers>=4.40.0`, `accelerate>=0.26.0`, and `simple-lama-inpainting==0.1.2`
- SAM officially recommends Linux/WSL; WSL is recommended on Windows
- OCR requires at least one complete route: `paddleocr` + `paddlepaddle`, or `pytesseract` + a system Tesseract executable; `doctor` uses this requirement when deciding whether the environment is ready
- PSD export additionally requires the Aspose.PSD package and license, plus the `ASPOSE_PSD_LICENSE` environment variable

### Installation

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
cd image2editable
pip install .

# Install the optional dependency when PSD export is needed
pip install .[psd]
```

### Models and First Run

PPTX conversion depends on Grounding DINO, SAM 2.1, and LaMa. The required models are downloaded to the local cache on first run; model weights are not included in this repository. The pipeline uses CUDA when available and also supports CPU execution, although CPU mode is significantly slower. If you already have a local LaMa TorchScript model, set `LAMA_MODEL` to its path.

### Resource and Quality Policy

The default policy limits heavy-page concurrency to one, numerical threads to at most eight, and SAM `points_per_batch` to one. OCR detection/recognition, LaMa, DINO, SAM prompted/automatic/final hole recheck, and full-page visual processing run in sequential subprocess phases, so process exit becomes the resource-release boundary. This still uses SAM2.1 Large and does not reduce the 200 DPI PDF baseline, SAM sampling points, or candidate thresholds; conversion takes longer as a tradeoff.

The final quality gate checks non-text P99 error, changed-pixel ratio, and the largest contiguous artifact. If layered reconstruction has visible risk, the page falls back to the existing text-clean fidelity background while keeping OCR text boxes editable. Visual objects are not independently layered on fallback pages.

### OCR Engines

**Option A: PaddleOCR (higher accuracy for Chinese text)**

```bash
pip install paddleocr paddlepaddle
```

**Option B: Tesseract (lighter weight)**

```bash
pip install pytesseract
# Install Tesseract on the system:
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

Configure the PSD export license:

Windows PowerShell:

```powershell
$env:ASPOSE_PSD_LICENSE = "C:\path\to\Aspose.PSD.lic"
```

macOS/Linux:

```bash
export ASPOSE_PSD_LICENSE=/path/to/Aspose.PSD.lic
```

Aspose.PSD is a commercial component. Make sure you have a license that complies with its official EULA before use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for other third-party dependencies and licenses, and [CITATION.cff](CITATION.cff) for citation information.

---

## Usage

### Skill Installation

The project provides two independent Skills:

- `skills/image-to-ppt/`: convert images to editable PPTX files
- `skills/image-to-psd/`: convert images to layered PSD files

**Method 1: Use the skills CLI**

```bash
npx skills add DSY-Xueai/image2editable --skill <skill_name>
```

Replace `<skill_name>` with the Skill directory name, such as `image-to-ppt`.

**Method 2: Let an Agent install it automatically**

```text
Install the <skill_name> skill from https://github.com/DSY-Xueai/image2editable.
```

**Method 3: Claude Code plugin**

```bash
claude plugin marketplace add https://github.com/DSY-Xueai/image2editable
claude plugin install image2editable@image2editable --scope user
```

**Method 4: Manual installation**

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
mkdir -p ~/.claude/skills
cp -R image2editable/skills/image-to-ppt ~/.claude/skills/<skill_name>
```

### Command Line

#### Unified Runtime (P1)

```bash
# Images and directories continue through the existing editable-PPTX pipeline
image2editable convert input.png -o output.pptx --slide-size 16:9

# Convert a PDF directly, or prepare it and request one detail rerender per page before execution
image2editable convert input.pdf -o output.pptx --slide-size 16:9
image2editable prepare input.pdf --run-dir runs/pdf-job
image2editable run render-detail runs/pdf-job --page page_001
image2editable run execute runs/pdf-job
image2editable run recover runs/pdf-job

# P1 preserves PPTX inputs losslessly
image2editable convert input.pptx -o preserved.pptx

# Inspect local dependencies
image2editable doctor
```

The Unified CLI calls its positional inputs `sources`. It accepts image files/directories, one PDF, or one PPTX. A document cannot be mixed with other sources or supplied more than once. The existing `python image_to_ppt.py` and `python image_to_psd.py` image entry points remain compatible.

`run recover` only resumes an orphaned task whose execution lock is gone; it does not terminate an active conversion process.

PDF pages are rendered adaptively and then reuse the existing image-to-editable-PPTX pipeline. The standard target is 200 DPI. Small pages are raised to a 1200 px short-edge floor without exceeding 300 DPI; every render is capped at a 6000 px long edge and 24 MP. An Agent or user may call `render-detail` once per page to rerender it with a 300 DPI target. PDF pages with the same physical aspect ratio can be combined into a ratio-preserving multi-slide PPTX. With mixed aspect ratios, `original` produces one output per page while a uniform 16:9 version remains available. Layout is always scaled uniformly, with no non-uniform stretching.

For PPTX inputs, P1 read-only scans native OOXML objects, notes, image relationships, and stable fingerprints. Only structurally safe large images covering at least 80% of a slide are marked as candidates. P1 execution produces a byte-identical copy of the input PPTX, preserving existing editable text, shapes, tables, charts, and other native objects; they do not pass through CV, and images are not automatically separated or replaced. Final-screenshot classification by the Agent, OCR/reconstruction, and in-place OOXML replacement belong to P2 and are not implemented yet.

Python 3.10–3.12 is supported; `doctor` now checks PDFium as well.

#### Compatible Entry Points

```bash
# One image → generates input_original.pptx and input_16x9.pptx by default
python image_to_ppt.py input.png

# Generate only one slide size
python image_to_ppt.py input.png --slide-size original
python image_to_ppt.py input.png --slide-size 16:9

# Multiple images → generates a multi-slide 16:9 PPTX and original-size single-slide PPTX files in the *_original directory
python image_to_ppt.py img1.png img2.png img3.png -o slides.pptx

# Directory input → also supports original, 16:9, or both
python image_to_ppt.py ./my_slides/ -o presentation.pptx

# Add the original image as a reference slide after each content slide
python image_to_ppt.py img1.png img2.png --reference

# One image → PSD
python image_to_psd.py input.png

# Multiple images → one PSD per image
python image_to_psd.py img1.png img2.png -o psd_output_dir

# Directory input → one PSD per image
python image_to_psd.py ./my_slides/ -o psd_output_dir

# Adjust PSD parameters
python image_to_psd.py input.png --lang en --diff-threshold 15 --min-area 30
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `images` | Required | An image file, multiple image files, or a directory; directory input scans only the first level |
| `-o, --output` | Same name as input | PPTX: a file path for `original` or `16:9`; an output base name for the default `both` mode. PSD: a file path for one image or an output directory for multiple images |
| `--lang` | `ch` | OCR language, commonly `ch` or `en` |
| `--period` | `32` | PPTX: retained for compatibility and has no effect. PSD: background-model tile period |
| `--diff-threshold` | `20.0` | PPTX: retained for compatibility and has no effect. PSD: foreground detection threshold |
| `--min-area` | `20` | PPTX: retained for compatibility and has no effect. PSD: minimum component area |
| `--reference` | Disabled | PPTX only: add the original image as a reference slide after each content slide |
| `--no-reference` | Default behavior | PPTX only: explicitly disable original-image reference slides |
| `--slide-size` | `both` | PPTX only: `original` preserves the input ratio, `16:9` outputs widescreen slides, and `both` generates both sizes |

---

## Project Structure

```
image2editable/
├── .claude-plugin/
│   └── plugin.json        # Claude Code plugin configuration exposing two independent Skills
├── image_to_ppt.py        # Image-to-PPTX entry point (CLI + Python API)
├── image_to_psd.py        # Image-to-PSD entry point (CLI + Python API)
├── scripts/               # Core processing and export modules
│   ├── text_detect.py     # OCR text detection and style estimation
│   ├── bg_model.py        # Background modeling and repair
│   ├── fg_extract.py      # Foreground component extraction and separation
│   ├── ppt_assemble.py    # Layered PPTX assembly
│   ├── psd_assemble.py    # Layered PSD assembly (Aspose.PSD)
│   └── visual_compare_qa.py # Manual visual comparison QA tool
├── skills/                # Distributable Agent Skills
│   ├── image-to-ppt/      # Image-to-PPTX Skill
│   └── image-to-psd/      # Image-to-PSD Skill
└── requirements.txt       # Python dependencies
```

---

## Tech Stack

| Area | Technology |
|------|------------|
| Image processing | OpenCV, Pillow, NumPy |
| OCR | PaddleOCR, Tesseract |
| PPTX generation | python-pptx |
| PSD generation | Aspose.PSD |
| Background repair | OpenCV inpainting (small/narrow masks) + LaMa (large/deep masks) |
| PPTX visual segmentation | Grounding DINO semantic proposals + SAM 2.1 masks + unique ownership |
| PSD foreground detection | Difference thresholding + Canny edges + morphological operations |

---

## Use Cases

- Convert PowerPoint screenshots, course pages, or design previews into editable PPTX files
- Convert screenshots or design images into layered Photoshop PSD files
- Images with relatively regular backgrounds and clear text produce better results
- Supports Chinese and English content

---

## Supported Image Formats

PNG · JPG / JPEG · BMP · TIFF / TIF · WebP

## LICENSE

MIT
