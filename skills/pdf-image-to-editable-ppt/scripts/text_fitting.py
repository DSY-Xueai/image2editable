from __future__ import annotations


def validate_text_block_fit(candidate: dict) -> bool:
    if candidate.get("expected_lines") != candidate.get("predicted_lines"):
        return False
    if float(candidate.get("position_delta", 0.0)) != 0.0:
        return False
    if not candidate.get("font_name"):
        return False
    return True


def mark_fit_verified(text_block):
    text_block.fit_verified = True
    return text_block
