from __future__ import annotations

from contextlib import redirect_stdout
import importlib
from pathlib import Path
import sys
from typing import Any

from image2editable.contracts import validate_schema_version
from image2editable.inputs import sha256_file
from image2editable.store import RunStore


def _absolute_outputs(value: Any) -> Any:
    if isinstance(value, str):
        return str(Path(value).resolve())
    if isinstance(value, list):
        return [_absolute_outputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _absolute_outputs(item) for key, item in value.items()}
    return value


def _source_path(store: RunStore, page_id: str) -> Path:
    request = store.read_json(
        Path("pages") / page_id / "page_request.json"
    )
    validate_schema_version(request)
    source = (store.root / request["source"]).resolve()
    if not source.is_relative_to(store.root):
        raise ValueError(f"{page_id}: source is outside run directory")
    if not source.is_file():
        raise ValueError(f"{page_id}: source is not a file")
    if sha256_file(source) != request["sha256"]:
        raise ValueError(f"{page_id}: source sha256 mismatch")
    return source


def execute_legacy(store: RunStore) -> dict[str, Any]:
    manifest = store.read_json("job_manifest.json")
    validate_schema_version(manifest)
    sources = [_source_path(store, page_id) for page_id in manifest["pages"]]
    options = manifest["options"]
    slide_size = options["slide_size"]
    combine_original = (
        manifest["input"].get("type") == "pdf"
        and manifest["input"].get("page_ratios_equal") is True
    )
    original_aspect_ratio = manifest["input"].get("page_aspect_ratio")
    output_path = options["output_path"]
    if output_path is None:
        output_path = str(store.root / "final" / "output.pptx")

    module = importlib.import_module("image_to_ppt")
    with redirect_stdout(sys.stderr):
        if len(sources) == 1 and slide_size == "both":
            result = module.convert_variants(
                sources[0],
                output_path=output_path,
                lang=options["lang"],
            )
        elif len(sources) == 1:
            result = {
                slide_size: module.convert(
                    sources[0],
                    output_path=output_path,
                    lang=options["lang"],
                    slide_size=slide_size,
                )
            }
        elif slide_size == "both":
            kwargs = {
                "output_path": output_path,
                "lang": options["lang"],
            }
            if combine_original:
                kwargs["combine_original"] = True
                if original_aspect_ratio is not None:
                    kwargs["original_aspect_ratio"] = original_aspect_ratio
            result = module.convert_batch_variants(
                sources,
                **kwargs,
            )
        elif slide_size == "original":
            kwargs = {
                "output_path": output_path,
                "lang": options["lang"],
                "include_widescreen": False,
            }
            if combine_original:
                kwargs["combine_original"] = True
                if original_aspect_ratio is not None:
                    kwargs["original_aspect_ratio"] = original_aspect_ratio
            result = module.convert_batch_variants(
                sources,
                **kwargs,
            )
        else:
            result = {
                "16:9": module.convert_batch(
                    sources,
                    output_path=output_path,
                    lang=options["lang"],
                )
            }
    return _absolute_outputs(result)
