from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS_DIR = (
    WORKTREE_ROOT / "skills" / "pdf-image-to-editable-ppt" / "scripts"
)


def load_skill_module(module_name: str) -> ModuleType:
    module_path = SKILL_SCRIPTS_DIR / f"{module_name}.py"
    package_name = "skill_scripts"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(SKILL_SCRIPTS_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
