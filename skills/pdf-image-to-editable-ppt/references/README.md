# References

## Supported inputs
- Single images
- Single-page PDFs
- Multi-page PDFs
- Supports multi-page PDF conversion with page-by-page fallback

## Fallback policy
- Preserve background fidelity first
- Only promote high-confidence text and images

## Known limits
- Missing or subset font mappings can force a fallback to the background layer
- OCR uncertainty on scanned pages can block editable text promotion
- Complex effects remain in the background layer
