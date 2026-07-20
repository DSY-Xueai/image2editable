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

> In the PPTX output, foreground elements can be moved and text boxes can be edited.
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
| PPTX export | Generates a background, independent transparent components, and editable text boxes; outputs both original-aspect-ratio and 16:9 versions by default |
| PSD export | Generates a layered PSD with a background layer, foreground pixel layers, and Photoshop text layers |
| Batch processing | Accepts multiple images or a directory; PPTX files are combined into multiple slides, while PSD exports one file per image |

---

## Quick Start

### Requirements

- Python 3.10–3.12 (the upper limit comes from the NumPy/Pillow constraints of `simple-lama-inpainting 0.1.2`)
- `torch>=2.5.1`, `torchvision>=0.20.1`, `transformers>=4.40.0`, and `simple-lama-inpainting==0.1.2`
- SAM officially recommends Linux/WSL; WSL is recommended on Windows
- At least one OCR engine must be installed
- PSD export requires an Aspose.PSD license and the `ASPOSE_PSD_LICENSE` environment variable

### Installation

```bash
git clone https://github.com/DSY-Xueai/image2editable.git
cd image2editable
pip install -r requirements.txt
```

### Models and First Run

PPTX conversion depends on Grounding DINO, SAM 2.1, and LaMa. The required models are downloaded to the local cache on first run; model weights are not included in this repository. The pipeline uses CUDA when available and also supports CPU execution, although CPU mode is significantly slower. If you already have a local LaMa TorchScript model, set `LAMA_MODEL` to its path.

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

```bash
# Windows
set ASPOSE_PSD_LICENSE=C:\path\to\Aspose.PSD.lic

# macOS/Linux
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
