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

## Script entry points
- `scripts/render_pdf_pages.py` renders per-page background assets.
- `scripts/extract_text_layout.py` returns candidate text blocks with layout metadata.
- `scripts/extract_images.py` returns candidate image blocks.
- `scripts/build_ppt.py` assembles the final PPT from page plans.
