from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _smoke_module():
    return importlib.import_module("scripts.runtime_model_smoke")


def test_runtime_model_smoke_runs_all_three_production_inference_paths() -> None:
    smoke = _smoke_module()
    calls: list[object] = []

    class Generator:
        def generate(self, image):
            assert image.shape == (64, 64, 3)
            assert image.dtype == np.uint8
            calls.append("sam-inference")
            return [{"segmentation": np.zeros((64, 64), dtype=bool)}]

    class Detector:
        def detect(self, image, prompt, box_threshold, text_threshold):
            assert image.shape == (64, 64, 3)
            assert prompt == "object."
            assert box_threshold == pytest.approx(0.25)
            assert text_threshold == pytest.approx(0.25)
            calls.append("dino-inference")
            return []

    checkpoint = Path("sam2.1_hiera_large.pt")

    def create_sam(path, *, device, resource_safe):
        calls.append((path, device, resource_safe))
        return Generator()

    def create_detector(*, device):
        calls.append(("dino", device))
        return Detector()

    def fake_inpaint(image, mask):
        assert image.shape == (64, 64, 3)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        calls.append("lama-inference")
        return image.copy()

    assert smoke._run_smoke(
        np,
        resolve_sam_checkpoint=lambda: checkpoint,
        create_sam_generator=create_sam,
        create_object_detector=create_detector,
        inpaint_large_mask=fake_inpaint,
    ) == {
        "models": ["sam2_large", "grounding_dino", "big_lama"],
        "ok": True,
    }
    assert calls == [
        (checkpoint, "cpu", True),
        "sam-inference",
        ("dino", "cpu"),
        "dino-inference",
        "lama-inference",
    ]


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("create_sam_generator", lambda *args, **kwargs: object()),
        (
            "create_object_detector",
            lambda **kwargs: type(
                "BadDetector",
                (),
                {"detect": lambda *args, **kwargs: "not-a-list"},
            )(),
        ),
        ("inpaint_large_mask", lambda image, mask: image.astype(np.float32)),
    ],
)
def test_runtime_model_smoke_rejects_invalid_model_output(attribute, replacement) -> None:
    smoke = _smoke_module()

    class Generator:
        def generate(self, image):
            return []

    class Detector:
        def detect(self, *args, **kwargs):
            return []

    dependencies = {
        "resolve_sam_checkpoint": lambda: Path("sam.pt"),
        "create_sam_generator": lambda *args, **kwargs: Generator(),
        "create_object_detector": lambda **kwargs: Detector(),
        "inpaint_large_mask": lambda image, mask: image.copy(),
    }
    dependencies[attribute] = replacement

    with pytest.raises(smoke.RuntimeModelSmokeError):
        smoke._run_smoke(np, **dependencies)


def test_runtime_model_smoke_main_has_fixed_safe_json(monkeypatch, capsys) -> None:
    smoke = _smoke_module()
    secret = r"C:\Users\private\model-cache"
    monkeypatch.setattr(
        smoke,
        "run_smoke",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"models": [], "ok": False}
    assert captured.err == ""
    assert secret not in captured.out


def test_runtime_model_smoke_hides_import_failures_in_a_real_process() -> None:
    secret = r"C:\Users\private\broken-binary"
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        sys.path.insert(0, {str(ROOT)!r})

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy":
                    raise RuntimeError({secret!r})
                return None

        sys.meta_path.insert(0, Blocker())
        runpy.run_path(
            str({str(ROOT / "scripts" / "runtime_model_smoke.py")!r}),
            run_name="__main__",
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 1
    assert json.loads(completed.stdout) == {"models": [], "ok": False}
    assert completed.stderr == ""
    assert secret not in completed.stdout
