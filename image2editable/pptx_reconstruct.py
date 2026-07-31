from __future__ import annotations

import hashlib
import json
from pathlib import Path

import image_to_ppt


def build_reconstruction_donor(
    image_path: str | Path,
    output_pptx: str | Path,
    work_root: str | Path,
    *,
    decision: dict,
    lang: str = "ch",
) -> dict:
    """Run the isolated CV pipeline for one Agent-approved screenshot."""
    if (
        decision.get("runtime_action") != "shadow_run"
        or decision.get("eligible_for_shadow_run") is not True
        or not isinstance(decision.get("source_shape_id"), str)
        or not decision["source_shape_id"]
    ):
        raise ValueError(
            "reconstruction requires an Agent-approved shadow_run decision"
        )

    image = Path(image_path).resolve()
    output = Path(output_pptx).resolve()
    root = Path(work_root).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if output.exists():
        raise FileExistsError(output)
    root.mkdir(parents=True, exist_ok=True)

    slide_data, assets = image_to_ppt._prepare_single_image(
        image,
        lang,
        _work_root=root,
        _resource_isolation=True,
    )
    image_to_ppt._assemble_prepared_slide(
        slide_data,
        output,
        False,
        "original",
    )
    if not output.is_file():
        raise RuntimeError("reconstruction pipeline did not create donor PPTX")

    manifest = {
        "source_shape_id": decision["source_shape_id"],
        "candidate_image": str(image),
        "candidate_sha256": _sha256(image),
        "donor_pptx": str(output),
        "assets": str(assets),
        "components": len(slide_data["components"]),
        "text_boxes": len(slide_data["text_items"]),
        "quality": slide_data.get("quality"),
        "background_residual": slide_data.get("background_residual"),
    }
    manifest_path = root / "reconstruction_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
