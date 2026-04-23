#!/usr/bin/env python3
"""Multi-provider AI vision analyzer for image-to-editable-elements conversion.

Supports: Anthropic (Claude), OpenAI (GPT-4V), any OpenAI-compatible API.
Reads config from environment variables.

Usage:
    from vision_analyzer import analyze_image
    elements = analyze_image("slide.png")
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

ANALYSIS_PROMPT = """你是一个专业的图片→PPT还原引擎。分析这张图片，识别出所有视觉元素，返回JSON。

要求：
1. 识别背景（纯色、渐变、或描述）
2. 识别所有元素：文字、矩形、圆形、线条、箭头、图标等
3. 每个元素给出精确的位置（相对于图片尺寸的百分比 0-100）
4. 颜色用 hex 格式

返回格式（严格JSON，不要markdown代码块）：
{
  "width": 图片宽度px,
  "height": 图片高度px,
  "background": {
    "type": "solid" | "gradient" | "image",
    "color": "#hex",
    "gradient_start": "#hex",
    "gradient_end": "#hex",
    "gradient_direction": "top_to_bottom" | "left_to_right" | "diagonal"
  },
  "elements": [
    {
      "type": "text",
      "text": "文字内容",
      "x_pct": 左边距百分比,
      "y_pct": 上边距百分比,
      "width_pct": 宽度百分比,
      "height_pct": 高度百分比,
      "font_size_pct": 字号相对图片高度的百分比,
      "font_weight": "normal" | "bold",
      "color": "#hex",
      "align": "left" | "center" | "right"
    },
    {
      "type": "rect",
      "x_pct": 左边距百分比,
      "y_pct": 上边距百分比,
      "width_pct": 宽度百分比,
      "height_pct": 高度百分比,
      "fill": "#hex",
      "stroke": "#hex" | null,
      "stroke_width": 数字,
      "corner_radius": 数字,
      "opacity": 0-1
    },
    {
      "type": "circle",
      "cx_pct": 圆心x百分比,
      "cy_pct": 圆心y百分比,
      "r_pct": 半径百分比(相对宽度),
      "fill": "#hex",
      "stroke": "#hex" | null
    },
    {
      "type": "line",
      "x1_pct": 起点x百分比,
      "y1_pct": 起点y百分比,
      "x2_pct": 终点x百分比,
      "y2_pct": 终点y百分比,
      "stroke": "#hex",
      "stroke_width": 数字
    },
    {
      "type": "image_region",
      "description": "描述这个图片区域的内容",
      "x_pct": 左边距百分比,
      "y_pct": 上边距百分比,
      "width_pct": 宽度百分比,
      "height_pct": 高度百分比
    }
  ]
}

注意：
- 位置用百分比（0-100），不是像素
- 元素按从底层到顶层排列（先背景元素，后前景元素）
- 文字内容要完整准确，中英文都要识别
- 颜色要尽量精确匹配原图
- 不要遗漏任何可见元素
- 对于复杂的图片区域（照片、截图等无法用形状重建的），用 image_region 类型标记"""


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 and detect media type."""
    data = image_path.read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    return b64, media_type


def _extract_json(text: str) -> dict:
    """Extract JSON from AI response (handles markdown code blocks)."""
    # Try direct parse
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from AI response: {text[:200]}...")


def _call_anthropic(image_path: Path) -> dict:
    """Call Anthropic Claude API with vision."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key:
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    b64, media_type = _encode_image(image_path)

    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    )
    return _extract_json(response.content[0].text)


def _call_openai(image_path: Path) -> dict:
    """Call OpenAI-compatible API with vision."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=base_url)
    b64, media_type = _encode_image(image_path)

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                        },
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    )
    return _extract_json(response.choices[0].message.content)


def detect_provider() -> str:
    """Auto-detect which AI provider is configured."""
    if os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No AI provider configured. Set ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY."
    )


def analyze_image(image_path: str | Path, provider: str | None = None) -> dict:
    """Analyze image and return structured element descriptions.

    Args:
        image_path: Path to the image file
        provider: "anthropic", "openai", or None (auto-detect)

    Returns:
        Dict with background and elements descriptions
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if provider is None:
        provider = detect_provider()

    print(f"[AI] Using provider: {provider}")
    print(f"[AI] Analyzing image: {image_path.name}...")

    if provider == "anthropic":
        result = _call_anthropic(image_path)
    elif provider == "openai":
        result = _call_openai(image_path)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    elem_count = len(result.get("elements", []))
    print(f"[AI] Detected {elem_count} elements")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vision_analyzer.py <image_path>")
        sys.exit(1)

    result = analyze_image(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
