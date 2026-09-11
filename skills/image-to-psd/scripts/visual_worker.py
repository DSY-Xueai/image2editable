from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
from PIL import Image



@dataclass(frozen=True)
class PagePolicy:
    route: str
    confidence: float
    reasons: tuple[str, ...]
    automatic_sam: bool
    max_residual_rounds: int
    hole_recheck: bool
    max_lama_calls: int
    host_agent_allowed: bool


def strict_page_policy() -> PagePolicy:
    return PagePolicy(
        route="strict", confidence=0.0, reasons=("strict_mode",),
        automatic_sam=True, max_residual_rounds=3, hole_recheck=True,
        max_lama_calls=2, host_agent_allowed=True,
    )


def _is_link_or_reparse(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & reparse_flag
    )


def _read_source_snapshot(
    image_path: Path,
    work_dir: Path,
    expected_size: int,
    label: str,
) -> bytes:
    root = Path(os.path.abspath(work_dir))
    source = Path(os.path.abspath(image_path))
    if source.parent != root:
        raise ValueError(f"{label} must be directly inside the work directory")
    root_before = os.lstat(root)
    path_before = os.lstat(source)
    try:
        with source.open("rb") as stream:
            handle_before = os.fstat(stream.fileno())
            if handle_before.st_size != expected_size:
                raise ValueError(f"{label} size mismatch")
            content = stream.read(expected_size + 1)
            handle_after = os.fstat(stream.fileno())
        path_after = os.lstat(source)
        root_after = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"{label} changed while being read") from exc

    for status in (root_before, root_after):
        if _is_link_or_reparse(status) or not stat.S_ISDIR(status.st_mode):
            raise ValueError("visual work directory changed while being read")
    if (root_before.st_dev, root_before.st_ino) != (
        root_after.st_dev, root_after.st_ino
    ):
        raise ValueError("visual work directory changed while being read")
    statuses = (path_before, handle_before, handle_after, path_after)
    for status in statuses:
        if (
            _is_link_or_reparse(status)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
        ):
            raise ValueError(f"{label} changed while being read")
    identities = {
        (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
        for status in statuses
    }
    if len(identities) != 1 or len(content) != expected_size:
        raise ValueError(f"{label} changed while being read")
    return content


def _load_process_image():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    if (script_dir / "image_to_ppt.py").is_file():
        from scripts.image_to_ppt import _process_image
    else:
        from image_to_ppt import _process_image

    return _process_image


def _load_visual_tools():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir.parent))
    from scripts.object_detect import create_object_detector
    from scripts.visual_segment import create_sam_generator, resolve_sam_checkpoint

    return create_object_detector, create_sam_generator, resolve_sam_checkpoint


def _request_page_policy(request: dict) -> PagePolicy:
    payload = request.get("page_policy")
    if payload is None:
        return strict_page_policy()
    if not isinstance(payload, dict):
        raise ValueError("visual page policy is invalid")
    try:
        values = dict(payload)
        values["reasons"] = tuple(values.get("reasons", ()))
        return PagePolicy(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError("visual page policy is invalid") from exc


class _ResidentVisualProcessor:
    """Keep one sequential DINO/SAM/LaMa model lifecycle inside --serve."""

    def __init__(
        self,
        *,
        process_image=None,
        create_detector=None,
        create_generator=None,
        resolve_checkpoint=None,
    ) -> None:
        self._process_image = process_image or _load_process_image()
        if create_detector is None:
            create_detector, create_generator, resolve_checkpoint = _load_visual_tools()
        self._create_detector = create_detector
        self._create_generator = create_generator
        self._resolve_checkpoint = resolve_checkpoint
        self._detector = None
        self._generator = None

    def process(
        self,
        image_path: Path,
        work_dir: Path,
        lang: str,
        text_analysis: dict,
        source_image: np.ndarray,
        text_mask: np.ndarray,
        text_clean_image: np.ndarray | None,
        page_policy: PagePolicy,
    ) -> dict:
        if page_policy.route == "direct":
            detector = None
            generator = None
        else:
            if self._detector is None:
                self._detector = self._create_detector()
            if self._generator is None:
                self._generator = self._create_generator(
                    self._resolve_checkpoint(), resource_safe=True,
                )
            detector = self._detector
            generator = self._generator
        return self._process_image(
            image_path,
            work_dir,
            detector,
            generator,
            lang,
            text_analysis=text_analysis,
            defer_quality=True,
            _resource_isolation=False,
            _source_image=source_image,
            _text_mask=text_mask,
            _text_clean_image=text_clean_image,
            page_policy=page_policy,
        )

    def component_prompts(self, request_path: Path, result_path: Path) -> None:
        if self._generator is None:
            self._generator = self._create_generator(
                self._resolve_checkpoint(), resource_safe=True,
            )
        from scripts.sam_worker import run_component_prompt_batch_with_generator

        run_component_prompt_batch_with_generator(
            request_path,
            result_path,
            self._generator,
        )

    def close(self) -> None:
        self._detector = None
        self._generator = None
        try:
            from scripts.lama_inpaint import release_model

            release_model()
        finally:
            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _process_component_prompt_request(
    payload: dict,
    processor: _ResidentVisualProcessor,
) -> None:
    if set(payload) != {"kind", "request", "result"}:
        raise ValueError("visual component prompt request is invalid")
    if any(
        not isinstance(payload[name], str) or not payload[name]
        for name in ("request", "result")
    ):
        raise ValueError("visual component prompt request is invalid")
    processor.component_prompts(Path(payload["request"]), Path(payload["result"]))


def _process_request(payload: dict, processor: _ResidentVisualProcessor | None) -> None:
    required = {
        "image", "work_dir", "lang", "request", "request_sha256",
        "request_size", "source_sha256", "source_size", "result",
    }
    if set(payload) != required:
        raise ValueError("visual worker request is invalid")
    if (
        not isinstance(payload["request_size"], int)
        or isinstance(payload["request_size"], bool)
        or payload["request_size"] <= 0
    ):
        raise ValueError("visual request binding is invalid")
    if (
        not isinstance(payload["source_size"], int)
        or isinstance(payload["source_size"], bool)
        or payload["source_size"] <= 0
    ):
        raise ValueError("visual source binding is invalid")
    if any(
        not isinstance(payload[name], str) or not payload[name]
        for name in required - {"request_size", "source_size"}
    ):
        raise ValueError("visual worker request is invalid")

    work_dir = Path(payload["work_dir"])
    request_content = _read_source_snapshot(
        Path(payload["request"]), work_dir, payload["request_size"], "visual request"
    )
    if hashlib.sha256(request_content).hexdigest() != payload["request_sha256"]:
        raise ValueError("visual request sha256 mismatch")
    request = json.loads(request_content.decode("utf-8"))
    page_policy = _request_page_policy(request)
    expected_sha256 = payload["source_sha256"]
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("visual source binding is invalid")
    source_content = _read_source_snapshot(
        Path(payload["image"]), work_dir, payload["source_size"], "visual source"
    )
    if hashlib.sha256(source_content).hexdigest() != expected_sha256:
        raise ValueError("visual source sha256 mismatch")
    with Image.open(io.BytesIO(source_content)) as stored_source:
        source_image = np.asarray(stored_source.convert("RGB")).copy()
    text_analysis = request["text_analysis"]
    text_mask_content = _read_source_snapshot(
        Path(text_analysis["mask_path"]),
        work_dir,
        request["text_mask_size"],
        "visual text mask",
    )
    if hashlib.sha256(text_mask_content).hexdigest() != request["text_mask_sha256"]:
        raise ValueError("visual text mask sha256 mismatch")
    with Image.open(io.BytesIO(text_mask_content)) as stored_text_mask:
        text_mask = np.asarray(stored_text_mask.convert("L")).copy()
    text_clean_image = None
    if text_analysis.get("text_clean_path") is not None:
        text_clean_content = _read_source_snapshot(
            Path(text_analysis["text_clean_path"]),
            work_dir,
            request["text_clean_size"],
            "visual text clean image",
        )
        if (
            hashlib.sha256(text_clean_content).hexdigest()
            != request["text_clean_sha256"]
        ):
            raise ValueError("visual text clean image sha256 mismatch")
        with Image.open(io.BytesIO(text_clean_content)) as stored_text_clean:
            text_clean_image = np.asarray(stored_text_clean.convert("RGB")).copy()

    if processor is None:
        slide_data = _load_process_image()(
            Path(payload["image"]),
            work_dir,
            None,
            None,
            payload["lang"],
            text_analysis=text_analysis,
            defer_quality=True,
            _resource_isolation=True,
            _source_image=source_image,
            _text_mask=text_mask,
            _text_clean_image=text_clean_image,
            page_policy=page_policy,
        )
    else:
        slide_data = processor.process(
            Path(payload["image"]),
            work_dir,
            payload["lang"],
            text_analysis,
            source_image,
            text_mask,
            text_clean_image,
            page_policy,
        )
    result_path = Path(payload["result"])
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(json.dumps(slide_data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, result_path)


def _serve() -> int:
    processor = _ResidentVisualProcessor()
    try:
        for line in sys.stdin:
            request_id = None
            try:
                envelope = json.loads(line)
                if envelope == {"control": "close"}:
                    break
                if not isinstance(envelope, dict):
                    raise ValueError("visual worker envelope is invalid")
                request_id = envelope.get("request_id")
                payload = envelope.get("payload")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("visual worker request id is invalid")
                if not isinstance(payload, dict):
                    raise ValueError("visual worker payload is invalid")
                with redirect_stdout(sys.stderr):
                    if payload.get("kind") == "component_prompts":
                        _process_component_prompt_request(payload, processor)
                    else:
                        _process_request(payload, processor)
                response = {"request_id": request_id, "result": {}}
            except Exception as error:
                response = {
                    "request_id": request_id,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        processor.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--work-dir")
    parser.add_argument("--lang")
    parser.add_argument("--request")
    parser.add_argument("--request-sha256")
    parser.add_argument("--request-size", type=int)
    parser.add_argument("--source-sha256")
    parser.add_argument("--source-size", type=int)
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.serve:
        return _serve()
    payload = vars(args)
    payload.pop("serve")
    if any(value is None for value in payload.values()):
        parser.error("one-shot visual worker requires all request arguments")
    _process_request(payload, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
