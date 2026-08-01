from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import hashlib
import importlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil


CATALOG_PATH = Path(__file__).with_name("model_catalog.json")
INSTALL_COMMAND = "image2editable models install agent"
RECEIPT_NAME = "agent-receipt.json"
_GIB = 1024**3
_MODEL_KEYS = {
    "model_id",
    "revision",
    "stability",
    "minimum_vram_gib",
    "minimum_available_vram_gib",
    "minimum_ram_gib",
    "minimum_available_ram_gib",
    "required_free_disk_gib",
    "priority",
}
_LOCAL_DEPENDENCIES = {
    "accelerate": "1.8.0",
    "huggingface-hub": "0.34.0",
    "torch": "2.5.0",
    "transformers": "4.57.0",
}


@dataclass(frozen=True)
class HardwareProfile:
    vram_gib: float
    ram_gib: float
    free_disk_gib: float
    cuda: bool
    available_vram_gib: float | None = None
    available_ram_gib: float | None = None


def snapshot_download(**kwargs: object) -> str:
    try:
        from huggingface_hub import snapshot_download as huggingface_download
    except ImportError as exc:
        raise RuntimeError(
            "Local Agent dependencies are missing; install with "
            "`pip install .[agent-local]`"
        ) from exc
    return str(huggingface_download(**kwargs))


def default_model_cache() -> Path:
    configured = os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "image2editable" / "models").resolve()


def load_model_catalog(path: str | Path | None = None) -> dict[str, object]:
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: object) -> None:
    if not isinstance(catalog, dict):
        raise ValueError("model catalog must be an object")
    if set(catalog) != {"catalog_version", "models"}:
        raise ValueError("model catalog has unsupported fields")
    if catalog["catalog_version"] != 1:
        raise ValueError("model catalog version is unsupported")
    entries = catalog["models"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("model catalog must contain at least one model")
    for entry in entries:
        _validate_catalog_entry(entry)


def _validate_catalog_entry(entry: object) -> None:
    if not isinstance(entry, dict) or set(entry) != _MODEL_KEYS:
        raise ValueError("model catalog entry has unsupported fields")
    for key in ("model_id", "revision", "stability"):
        if not isinstance(entry[key], str) or not entry[key]:
            raise ValueError(f"model catalog entry has invalid {key}")
    for key in (
        "minimum_vram_gib",
        "minimum_available_vram_gib",
        "minimum_ram_gib",
        "minimum_available_ram_gib",
        "required_free_disk_gib",
        "priority",
    ):
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"model catalog entry has invalid {key}")


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _free_disk_gib(cache_dir: Path) -> float:
    return round(shutil.disk_usage(_existing_parent(cache_dir)).free / _GIB, 2)


def _ram_profile() -> tuple[float, float]:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return (
                round(status.total_physical / _GIB, 2),
                round(status.available_physical / _GIB, 2),
            )
        return 0.0, 0.0
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return (
                round((pages * page_size) / _GIB, 2),
                round((available_pages * page_size) / _GIB, 2),
            )
        except (OSError, TypeError, ValueError):
            pass
    return 0.0, 0.0


def _cuda_profile() -> tuple[bool, float, float]:
    try:
        if importlib.util.find_spec("torch") is None:
            return False, 0.0, 0.0
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return False, 0.0, 0.0
        device_count = int(torch.cuda.device_count())
        profiles = []
        for index in range(device_count):
            total = int(torch.cuda.get_device_properties(index).total_memory)
            free, _ = torch.cuda.mem_get_info(index)
            profiles.append((int(free), total))
        available, total = max(profiles)
        return True, round(total / _GIB, 2), round(available / _GIB, 2)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False, 0.0, 0.0


def detect_hardware(cache_dir: str | Path | None = None) -> HardwareProfile:
    cache = (
        Path(cache_dir).expanduser().resolve() if cache_dir else default_model_cache()
    )
    cuda, vram_gib, available_vram_gib = _cuda_profile()
    ram_gib, available_ram_gib = _ram_profile()
    return HardwareProfile(
        vram_gib=vram_gib,
        ram_gib=ram_gib,
        free_disk_gib=_free_disk_gib(cache),
        cuda=cuda,
        available_vram_gib=available_vram_gib,
        available_ram_gib=available_ram_gib,
    )


def _format_gib(value: object) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def _compatibility_reasons(
    hardware: HardwareProfile,
    entry: dict[str, object],
) -> list[str]:
    reasons = []
    if not hardware.cuda:
        reasons.append("未检测到 CUDA")
    if hardware.vram_gib < float(entry["minimum_vram_gib"]):
        reasons.append(
            f"显存 {_format_gib(hardware.vram_gib)} GiB < "
            f"{_format_gib(entry['minimum_vram_gib'])} GiB"
        )
    if (
        hardware.available_vram_gib is not None
        and hardware.available_vram_gib
        < float(entry["minimum_available_vram_gib"])
    ):
        reasons.append(
            f"可用显存 {_format_gib(hardware.available_vram_gib)} GiB < "
            f"{_format_gib(entry['minimum_available_vram_gib'])} GiB"
        )
    if hardware.ram_gib < float(entry["minimum_ram_gib"]):
        reasons.append(
            f"内存 {_format_gib(hardware.ram_gib)} GiB < "
            f"{_format_gib(entry['minimum_ram_gib'])} GiB"
        )
    if (
        hardware.available_ram_gib is not None
        and hardware.available_ram_gib
        < float(entry["minimum_available_ram_gib"])
    ):
        reasons.append(
            f"可用内存 {_format_gib(hardware.available_ram_gib)} GiB < "
            f"{_format_gib(entry['minimum_available_ram_gib'])} GiB"
        )
    if hardware.free_disk_gib < float(entry["required_free_disk_gib"]):
        reasons.append(
            f"可用磁盘 {_format_gib(hardware.free_disk_gib)} GiB < "
            f"{_format_gib(entry['required_free_disk_gib'])} GiB"
        )
    return reasons


def detect_local_dependencies() -> dict[str, str | None]:
    versions = {}
    for package in _LOCAL_DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _release_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _version_compatible(installed: str | None, minimum: str) -> bool:
    if installed is None:
        return False
    installed_release = _release_tuple(installed)
    minimum_release = _release_tuple(minimum)
    if installed_release is None or minimum_release is None:
        return False
    if installed_release != minimum_release:
        return installed_release > minimum_release
    suffix = installed[re.match(r"^\d+(?:\.\d+){0,2}", installed).end() :]
    return re.search(r"(?:dev|a|b|rc)\d*", suffix, re.IGNORECASE) is None


def _dependency_report(
    package_versions: dict[str, str | None],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    report = {}
    reasons = []
    for package, minimum in _LOCAL_DEPENDENCIES.items():
        installed = package_versions.get(package)
        compatible = _version_compatible(installed, minimum)
        report[package] = {
            "installed": installed,
            "minimum": minimum,
            "compatible": compatible,
        }
        if installed is None:
            reasons.append(f"未安装 {package}>={minimum}")
        elif not compatible:
            reasons.append(f"{package} {installed} < {minimum}")
    return report, reasons


def recommend_agent_model(
    hardware: HardwareProfile,
    *,
    catalog: dict[str, object] | None = None,
    cache_dir: str | Path | None = None,
    package_versions: dict[str, str | None] | None = None,
) -> dict[str, object]:
    selected_catalog = catalog if catalog is not None else load_model_catalog()
    _validate_catalog(selected_catalog)
    entries = selected_catalog["models"]
    if not isinstance(entries, list):
        raise ValueError("model catalog models must be a list")
    candidates = sorted(entries, key=lambda item: item["priority"], reverse=True)
    selected = next(
        (entry for entry in candidates if not _compatibility_reasons(hardware, entry)),
        candidates[0],
    )
    dependencies, dependency_reasons = _dependency_report(
        package_versions
        if package_versions is not None
        else detect_local_dependencies()
    )
    reasons = _compatibility_reasons(hardware, selected) + dependency_reasons
    cache = (
        Path(cache_dir).expanduser().resolve() if cache_dir else default_model_cache()
    )
    return {
        "model_id": selected["model_id"],
        "revision": selected["revision"],
        "stability": selected["stability"],
        "compatible": not reasons,
        "reason": (
            "；".join(reasons)
            if reasons
            else "CUDA、显存、内存、磁盘和本地依赖均满足目录要求"
        ),
        "minimum_vram_gib": selected["minimum_vram_gib"],
        "minimum_available_vram_gib": selected["minimum_available_vram_gib"],
        "minimum_ram_gib": selected["minimum_ram_gib"],
        "minimum_available_ram_gib": selected["minimum_available_ram_gib"],
        "required_free_disk_gib": selected["required_free_disk_gib"],
        "cache_dir": str(cache),
        "hardware": {
            key: value
            for key, value in asdict(hardware).items()
            if value is not None
        },
        "dependencies": dependencies,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(snapshot: Path, cache: Path) -> list[dict[str, object]]:
    files = []
    for path in sorted(snapshot.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(cache):
            raise RuntimeError("model snapshot contains a file outside the cache")
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError("downloaded model snapshot is empty")
    return files


def install_agent_model(
    *,
    cache_dir: str | Path | None = None,
    free_disk_gib: float | None = None,
    confirmed: bool = False,
    model_id: str | None = None,
    revision: str | None = None,
) -> dict[str, object]:
    cache = (
        Path(cache_dir).expanduser().resolve() if cache_dir else default_model_cache()
    )
    catalog = load_model_catalog()
    entries = catalog["models"]
    if (model_id is None) != (revision is None):
        raise ValueError("confirmed model_id and revision must be provided together")
    if model_id is None:
        entry = max(entries, key=lambda item: item["priority"])
    else:
        entry = next(
            (
                item
                for item in entries
                if item["model_id"] == model_id and item["revision"] == revision
            ),
            None,
        )
        if entry is None:
            raise ValueError("confirmed model and revision are not in the catalog")
    available = _free_disk_gib(cache) if free_disk_gib is None else free_disk_gib
    required = float(entry["required_free_disk_gib"])
    if available < required:
        raise RuntimeError(
            f"insufficient free disk: {_format_gib(available)} GiB available, "
            f"{_format_gib(required)} GiB required"
        )
    if not confirmed:
        raise PermissionError("explicit confirmation required before model download")

    cache.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=entry["model_id"],
            revision=entry["revision"],
            cache_dir=str(cache),
        )
    ).resolve()
    if not snapshot.is_dir() or not snapshot.is_relative_to(cache):
        raise RuntimeError("downloaded model snapshot is outside the model cache")
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", snapshot.name) is None:
        raise RuntimeError("downloaded model snapshot did not resolve to a commit SHA")

    receipt = {
        "schema_version": 1,
        "model_id": entry["model_id"],
        "requested_revision": entry["revision"],
        "resolved_revision": snapshot.name.lower(),
        "stability": entry["stability"],
        "snapshot_path": str(snapshot),
        "files": _snapshot_files(snapshot, cache),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = cache / RECEIPT_NAME
    temporary = receipt_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt_path)
    return receipt


def model_status(*, cache_dir: str | Path | None = None) -> dict[str, object]:
    cache = (
        Path(cache_dir).expanduser().resolve() if cache_dir else default_model_cache()
    )
    receipt_path = cache / RECEIPT_NAME
    base = {
        "installed": False,
        "valid": False,
        "install_command": INSTALL_COMMAND,
    }
    if not receipt_path.is_file():
        return base
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise RuntimeError("model receipt must be an object")
        if receipt.get("schema_version") != 1 or isinstance(
            receipt.get("schema_version"), bool
        ):
            raise RuntimeError("receipt schema version is unsupported")
        catalog = load_model_catalog()
        matching_models = [
            entry
            for entry in catalog["models"]
            if entry["model_id"] == receipt.get("model_id")
        ]
        if not matching_models:
            raise RuntimeError("receipt model is not in the current catalog")
        if not any(
            entry["revision"] == receipt.get("requested_revision")
            and entry["stability"] == receipt.get("stability")
            for entry in matching_models
        ):
            raise RuntimeError("receipt does not match the current catalog entry")
        resolved_revision = receipt.get("resolved_revision")
        if (
            not isinstance(resolved_revision, str)
            or re.fullmatch(r"[0-9a-f]{40,64}", resolved_revision) is None
        ):
            raise RuntimeError("receipt resolved revision is not a commit SHA")
        snapshot = Path(receipt["snapshot_path"]).resolve()
        if not snapshot.is_dir() or not snapshot.is_relative_to(cache):
            raise RuntimeError("snapshot path is missing or outside the model cache")
        if receipt["resolved_revision"] != snapshot.name.lower():
            raise RuntimeError("receipt commit does not match snapshot path")
        files = receipt["files"]
        if not isinstance(files, list) or not files:
            raise RuntimeError("receipt file manifest is empty")
        expected = {}
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise RuntimeError("receipt file entry is invalid")
            relative = Path(item["path"])
            if (
                not isinstance(item["path"], str)
                or not item["path"]
                or relative.is_absolute()
                or ".." in relative.parts
                or item["path"] in expected
            ):
                raise RuntimeError(f"snapshot file path is invalid: {item['path']}")
            if (
                isinstance(item["size"], bool)
                or not isinstance(item["size"], int)
                or item["size"] < 0
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            ):
                raise RuntimeError(f"receipt file entry is invalid: {item['path']}")
            expected[item["path"]] = item
        actual = {item["path"]: item for item in _snapshot_files(snapshot, cache)}
        if set(actual) != set(expected):
            raise RuntimeError("snapshot file set does not match receipt")
        for relative_path, item in expected.items():
            if (
                actual[relative_path]["size"] != item["size"]
                or actual[relative_path]["sha256"] != item["sha256"]
            ):
                raise RuntimeError(f"snapshot file checksum mismatch: {relative_path}")
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RuntimeError,
    ) as exc:
        return {
            **base,
            "installed": True,
            "reason": str(exc),
        }
    return {
        "installed": True,
        "valid": True,
        "install_command": INSTALL_COMMAND,
        "receipt": receipt,
    }
