"""Recover outlined text styling from local pixels without model inference."""

from __future__ import annotations

import cv2
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=8)
def _font_metrics(font_name):
    from scripts.font_match import resolve_font
    return resolve_font(font_name, bold=True)


def _line_ink(region):
    if region.size == 0 or min(region.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    fill_mask = np.zeros(gray.shape, np.uint8)
    # Filled strokes form holes inside dark outlines. Sample the background
    # outside the enclosing contour, since image borders may be decorative.
    for i, contour in enumerate(contours):
        parent = hierarchy[0, i, 3]
        depth, ancestor = 0, parent
        while ancestor >= 0:
            depth += 1
            ancestor = hierarchy[0, ancestor, 3]
        if depth % 2 != 1 or cv2.contourArea(contour) < 15:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        hole = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(hole, [contour], -1, 255, -1)
        interior_mask = hole & cv2.bitwise_not(dark)
        child = hierarchy[0, i, 2]
        while child >= 0:
            cv2.drawContours(interior_mask, [contours[child]], -1, 0, -1)
            child = hierarchy[0, child, 0]
        interior = cv2.erode(interior_mask, np.ones((3, 3), np.uint8)) > 0
        if np.count_nonzero(interior) < 8:
            continue
        fill = np.median(region[interior], axis=0)
        outer = np.zeros(gray.shape, np.uint8)
        enclosing = parent
        while hierarchy[0, enclosing, 3] >= 0:
            enclosing = hierarchy[0, enclosing, 3]
        cv2.drawContours(outer, [contours[enclosing]], -1, 255, -1)
        ex, ey, ew, eh = cv2.boundingRect(contours[enclosing])
        margin = max(8, min(ew, eh))
        ys = slice(max(0, ey-margin), min(gray.shape[0], ey+eh+margin))
        xs = slice(max(0, ex-margin), min(gray.shape[1], ex+ew+margin))
        outside = outer[ys, xs] == 0
        if np.count_nonzero(outside) < 8:
            continue
        background = np.median(region[ys, xs][outside], axis=0)
        if np.linalg.norm(fill-background) < 30:
            continue
        ring = (cv2.dilate(interior_mask, np.ones((7, 7), np.uint8)) > 0) & (dark > 0)
        if not np.any(ring):
            continue
        # Use the adjacent outline, not the row threshold: pastel letters
        # can be close to Otsu while faint decorative contours are not ink.
        stroke = np.median(region[ring], axis=0)
        if float((fill-stroke) @ [0.299, 0.587, 0.114]) < 30:
            continue
        fill_mask[interior_mask > 0] = 255
    if np.count_nonzero(fill_mask) < 16:
        return None
    # Restrict connected outline pixels to the neighborhood of verified fills.
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    near = cv2.dilate(fill_mask, np.ones((5, 5), np.uint8)) > 0
    stroke_samples = distance[near & (dark > 0)]
    if len(stroke_samples) < 8:
        return None
    radius = max(2, int(round(float(np.percentile(stroke_samples, 90)) * 2)))
    near = cv2.dilate(fill_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0
    ink = (fill_mask > 0) | (near & (dark > 0))
    return fill_mask > 0, ink, dark > 0


def _outlined_ink(region, *, ownership=None, return_mask=False):
    measured = _line_ink(region)
    if measured is None:
        return None
    fill_mask, ink, dark = measured
    if ownership is not None:
        ink &= ownership > 0
        fill_mask &= ownership > 0
    if return_mask:
        return ink.astype(np.uint8) * 255
    return _measure_ink(region, fill_mask, ink, dark)


def _linear_gradient(region, interior, fill_mask, rotation):
    yy, xx = np.nonzero(interior)
    coords = np.column_stack((xx, yy)).astype(float)
    samples = region[interior].astype(float)
    design = np.column_stack((coords, np.ones(len(coords))))
    coefficients = np.linalg.lstsq(design, samples, rcond=None)[0]
    errors = np.linalg.norm(design @ coefficients-samples, axis=1)
    inliers = errors <= max(8, float(np.percentile(errors, 85)))
    coefficients = np.linalg.lstsq(design[inliers], samples[inliers], rcond=None)[0]
    errors = np.linalg.norm(design @ coefficients-samples, axis=1)
    if np.mean(errors < 20) < .95 or np.sqrt(np.mean(errors[inliers]**2)) > 8:
        return None
    directions, strengths, _ = np.linalg.svd(coefficients[:2], full_matrices=False)
    if strengths[0] < .05 or strengths[1] > strengths[0]*.15:
        return None
    direction = directions[:, 0]
    # Orient predominantly vertical/horizontal gradients top-to-bottom/left-to-right.
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    angle = np.radians(rotation)
    unrotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    local_direction = direction @ unrotate
    fy, fx = np.nonzero(fill_mask)
    local = np.column_stack((fx, fy)) @ unrotate
    low, high = local.min(axis=0), local.max(axis=0)
    corners = np.array([[low[0], low[1]], [high[0], low[1]], [low[0], high[1]], [high[0], high[1]]])
    limits = corners @ local_direction
    center = coords.mean(axis=0)
    base = np.append(center, 1) @ coefficients
    slope = direction @ coefficients[:2]
    colors = [np.clip(base+(position-center @ direction)*slope, 0, 255) for position in (limits.min(), limits.max())]
    if np.linalg.norm(colors[0]-colors[1]) < 20:
        return None
    return {"angle": float(np.degrees(np.arctan2(local_direction[1], local_direction[0])) % 360),
            "colors": ["#"+"".join(f"{round(value):02x}" for value in color) for color in colors]}


def _measure_ink(region, fill_mask, ink, dark, *, rotation=0, return_gradient=False):
    near = cv2.dilate(fill_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ink = fill_mask | (ink & dark & near)
    interior = cv2.erode(fill_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    stroke_mask = ink & dark
    if np.count_nonzero(interior) < 8 or np.count_nonzero(stroke_mask) < 8:
        return None
    samples = region[interior]
    quantized, counts = np.unique(samples // 16, axis=0, return_counts=True)
    dominant = quantized[np.argmax(counts)]
    fill = np.median(samples[np.all(samples // 16 == dominant, axis=1)], axis=0)
    gradient = None
    variation = np.linalg.norm(np.percentile(samples, 90, axis=0)-np.percentile(samples, 10, axis=0))
    consistent = np.mean(np.linalg.norm(samples.astype(float)-fill, axis=1) < 35) >= .75
    if variation > 20:
        gradient = _linear_gradient(region, interior, fill_mask, rotation)
        if gradient is None and not consistent:
            return None
    elif not consistent:
        return None
    stroke_color = np.median(region[stroke_mask], axis=0)
    distance = cv2.distanceTransform(stroke_mask.astype(np.uint8), cv2.DIST_L2, 5)
    stroke = max(.5, 2 * float(np.percentile(distance[stroke_mask], 90)) - 1)
    hex_color = lambda color: "#" + "".join(f"{round(value):02x}" for value in color)
    result = cv2.boundingRect(ink.astype(np.uint8)), hex_color(fill), hex_color(stroke_color), stroke
    return (*result, gradient) if return_gradient else result


def estimate_art_text_runs(pixels: np.ndarray, item: dict, *, reference_width: int):
    from scripts.font_match import match_glyph
    from scripts.text_runs import validate_text_words

    if not item.get("words") or len(item["text"].splitlines()) != 1:
        return None
    validate_text_words(item)
    x, y, width, height = item["box"]
    left, top = max(0, int(x)), max(0, int(y))
    line = pixels[top:min(pixels.shape[0], int(y+height)), left:min(pixels.shape[1], int(x+width))]
    measured = _line_ink(line)
    if measured is None:
        return None
    fill_mask, ink, dark = measured
    bx, by, bw, bh = cv2.boundingRect(ink.astype(np.uint8))
    words = item["words"]
    centers = [(word["box"][0]+word["box"][2]/2)*width+x-left for word in words]
    word_left = min(word["box"][0] for word in words)*width+x-left
    word_right = max(word["box"][0]+word["box"][2] for word in words)*width+x-left
    offset = ((bx-word_left)+(bx+bw-word_right))/2
    centers = np.clip(np.asarray(centers)+offset, bx, bx+bw-1)
    profile = fill_mask.sum(axis=0).astype(float)
    boundaries = [bx]
    for first, second in zip(centers, centers[1:]):
        low, high = max(boundaries[-1]+1, int(first)), min(line.shape[1], int(second)+1)
        if high <= low:
            return None
        costs = profile[low:high] + .03*np.abs(np.arange(low, high)-(first+second)/2)
        boundaries.append(low+int(np.argmin(costs)))
    boundaries.append(bx+bw)
    positioned = []
    for word, start_x, end_x in zip(words, boundaries, boundaries[1:]):
        text = word["text"]
        characters = [char for char in text if not char.isspace()]
        if len(characters) > 1:
            local_x, _, local_width, _ = cv2.boundingRect(ink[:, start_x:end_x].astype(np.uint8))
            char_edges = [start_x]
            for index in range(1, len(characters)):
                target = start_x + local_x + local_width*index/len(characters)
                margin = local_width/len(characters)/3
                low, high = max(char_edges[-1]+1, int(target-margin)), min(end_x, int(target+margin)+1)
                if high <= low:
                    return None
                costs = profile[low:high] + .03*np.abs(np.arange(low, high)-target)
                char_edges.append(low+int(np.argmin(costs)))
            char_edges.append(end_x)
            positioned.extend(zip(characters, char_edges, char_edges[1:]))
        else:
            positioned.append((text, start_x, end_x))
    runs, cursor = [], 0
    for word_text, start_x, end_x in positioned:
        region = line[:, start_x:end_x]
        local_fill = fill_mask[:, start_x:end_x]
        local_ink = ink[:, start_x:end_x]
        local_dark = dark[:, start_x:end_x]
        measured = _measure_ink(region, local_fill, local_ink, local_dark)
        if measured is None:
            return None
        stroke = measured[3]
        radius = max(1, round(stroke / 2))
        center_mask = cv2.dilate(local_fill.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))) > 0
        center_mask &= local_ink
        fitted = match_glyph(center_mask, word_text, item.get("font", "Arial"))
        if fitted is None:
            return None
        measured = _measure_ink(region, fill_mask[:, start_x:end_x], ink[:, start_x:end_x], dark[:, start_x:end_x],
                                rotation=fitted["rotation"], return_gradient=True)
        if measured is None:
            return None
        bounds, color, outline, stroke, gradient = measured
        rx, ry, rw, rh = bounds
        start = item["text"].find(word_text, cursor)
        if start < cursor or item["text"][cursor:start].strip():
            return None
        text = item["text"][cursor:start+len(word_text)]
        cursor = start+len(word_text)
        runs.append({"text": text, "box": [(left+start_x+rx-x)/width, (top+ry-y)/height, rw/width, rh/height],
                     "box_kind": "ink",
                     "font": fitted["font"], "bold": fitted["bold"],
                     "rotation": fitted["rotation"], "font_size": fitted["font_size"]*960/reference_width,
                     "color": color, "outline_color": outline, "outline_width": stroke*960/reference_width})
        if gradient is not None:
            runs[-1]["gradient"] = gradient
    if not runs or item["text"][cursor:].strip():
        return None
    runs[-1]["text"] += item["text"][cursor:]
    return runs
