# References

## Supported inputs
- Single images
- Single-page PDFs
- Multi-page PDFs
- Supports multi-page PDF conversion with page-by-page fallback

## Runtime dependencies
- `PyMuPDF` / `fitz`
- `Pillow` / `PIL`
- `PaddleOCR` / `paddleocr`
- `python-pptx` / `pptx`

## Dependency probing
- Use `scripts/dependencies.py` to detect runtime support before enabling optional extraction paths.
- Treat OCR support as optional; missing OCR support must not block fidelity-first PPT output.

## Stage 2 scope
- 2A covers strict text fitting validation and common effects.
- 2B is reserved for complex vectors, layered transparency, and harder reconstruction tasks.
- Current 2B groundwork focuses on layering, vector boundaries, and fail-closed handling for blend-heavy groupings rather than full blend reconstruction.

## Fallback policy
- Preserve background fidelity first
- Only promote high-confidence text and images
- If OCR is unavailable at runtime, keep scanned content in the background layer and continue rendering the slide.

## Known limits
- Missing or subset font mappings can force a fallback to the background layer
- OCR uncertainty on scanned pages can block editable text promotion
- Complex effects remain in the background layer
- Stage-one runtime upgrade still prioritizes text and image layers over complex vector/effect reconstruction
- Exact text reproduction still depends on deterministic fit validation, otherwise the text must remain in the background.
