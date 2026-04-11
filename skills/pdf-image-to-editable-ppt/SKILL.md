---
name: pdf-image-to-editable-ppt
description: Convert PDF files, multi-page PDFs, posters, scans, or images into a PPT while preserving the original look page by page. Use this whenever the user wants editable PowerPoint output but also insists on original layout, original colors, no visual damage, and conservative fallback for anything that cannot be safely reconstructed.
---

# PDF/Image To Editable PPT

## Core rules
- Preserve visual fidelity first.
- Build a background layer for every page.
- Add an editable layer only for high-confidence text and images.
- Support single images, single-page PDFs, and multi-page PDFs.
- If extraction could shift layout, do not force extraction; keep the source content in the background layer.

## Default mapping
- Map one PDF page to one slide by default.
- Map one source image to one slide by default.
- Keep the original visual as the slide baseline before adding editable objects.

## Long-page handling
- If the user asks for a split, divide a long page or long image into multiple slides.
- Each split slide still starts from a matching background layer segment.

## Fallback behavior
- Process each page independently.
- If text or image extraction is not reliable enough, leave that content in the background layer.
- A partial extraction failure must not block the rest of the document.

## Runtime dependencies
- `PyMuPDF` (`fitz`) for PDF page access and rendering.
- `Pillow` (`PIL`) for image handling.
- `PaddleOCR` (`paddleocr`) for OCR-assisted text recovery when available.
- `python-pptx` (`pptx`) for writing `.pptx` output.

Probe these packages at runtime before enabling extraction paths that depend on them. If `PaddleOCR` is unavailable, keep scanned text in the background layer and continue with the conservative fallback pipeline.

## Script entry points
- `scripts/render_pdf_pages.py` renders per-page background assets.
- `scripts/extract_text_layout.py` returns candidate text blocks with layout metadata.
- `scripts/extract_images.py` returns candidate image blocks.
- `scripts/convert_to_ppt.py` orchestrates PDF/image conversion end to end.
- `scripts/build_ppt.py` assembles the final PPT from page plans.

## Runtime flow
- Prefer native PDF text extraction through `PyMuPDF` before OCR.
- Use `PaddleOCR` when available for scanned pages and image inputs; allow other OCR fallbacks when the runtime environment requires it.
- Keep background-first PPT output as the invariant, then add filtered editable text and image layers.

## Stage 2 rules
- Stage 2 only promotes text when layout can remain fully identical.
- Stage 2 may map common effects when they are simple and safe to reproduce.
- If exact text fitting or effect mapping cannot be verified, fail closed and keep the content in the background layer.

## 2B rules
- 2B covers layering, vector boundary handling, and blend-heavy reconstruction cases.
- Treat layered objects as first-class page-plan data so later stages can preserve the original stack order.
- Keep vector boundaries conservative; if a vector boundary cannot be reconstructed safely, leave it in the background layer.
- For blend-heavy pages, fail closed and prefer the background layer over speculative reconstruction.
