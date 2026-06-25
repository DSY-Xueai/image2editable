"""any2ppt core package — image to editable PowerPoint conversion."""

from scripts.bg_model import build_background
from scripts.fg_extract import extract_foreground_mask, split_components
from scripts.ppt_assemble import assemble_pptx, assemble_pptx_multi
from scripts.text_detect import detect_text

__all__ = [
    "build_background",
    "extract_foreground_mask",
    "split_components",
    "assemble_pptx",
    "assemble_pptx_multi",
    "detect_text",
]
