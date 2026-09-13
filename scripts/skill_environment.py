"""Choose Skill storage and run preparation/conversion in the same environment."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile


def windows_data_drives() -> list[Path]:
    # DRIVE_FIXED excludes optical drives, removable media and network shares.
    return [
        Path(f"{letter}:/") for letter in "DEFGHIJKLMNOPQRSTUVWXYZAB"
        if ctypes.windll.kernel32.GetDriveTypeW(f"{letter}:\\") == 3
    ]


def writable_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=path):
        pass
    return path.resolve()


def mounted_data_volumes() -> list[Path]:
    if sys.platform == "darwin":
        volumes = []
        system_devices = {Path("/").stat().st_dev, Path.home().stat().st_dev}
        for path in sorted(Path("/Volumes").iterdir()):
            if not path.is_mount() or path.stat().st_dev in system_devices:
                continue
            result = subprocess.run(["diskutil", "info", "-plist", str(path)],
                                    capture_output=True, check=False)
            if result.returncode:
                continue
            info = plistlib.loads(result.stdout)
            if (info.get("DeviceNode", "").startswith("/dev/")
                    and info.get("BusProtocol") != "Disk Image"):
                volumes.append(path)
        return volumes
    result = subprocess.run(
        ["lsblk", "--json", "--output", "TYPE,MOUNTPOINTS"],
        capture_output=True, text=True, check=True,
    )
    volumes = []
    for device in json.loads(result.stdout)["blockdevices"]:
        if device["type"] != "disk":
            continue
        pending = [device]
        mounts = []
        while pending:
            node = pending.pop()
            mounts.extend(value for value in node.get("mountpoints", []) if value)
            pending.extend(node.get("children", []))
        if "/" not in mounts:
            volumes.extend(Path(value) for value in mounts
                           if value.startswith("/") and not value.startswith("/boot"))
    return sorted(set(volumes))


def installation_root() -> Path:
    drives = windows_data_drives() if sys.platform == "win32" else mounted_data_volumes()
    if not drives:
        fallback = (Path.home() / "image2editable" if sys.platform == "win32" else
                    Path.home() / ".local" / "share" / "image2editable")
        return writable_root(fallback)
    errors = []
    for drive in drives:
        try:
            return writable_root(drive / "image2editable")
        except OSError as error:
            errors.append(f"{drive}: {error}")
    raise OSError("Data drives exist but are not writable: " + "; ".join(errors))


def environment(root: Path) -> dict[str, str]:
    directories = {
        "PIP_CACHE_DIR": root / "cache" / "pip",
        "HF_HOME": root / "cache" / "huggingface",
        "HF_HUB_CACHE": root / "cache" / "huggingface" / "hub",
        "TORCH_HOME": root / "cache" / "torch",
        "PADDLE_HOME": root / "cache" / "paddle",
        "PADDLE_PDX_CACHE_HOME": root / "cache" / "paddlex",
        "XDG_CACHE_HOME": root / "cache",
        "UV_CACHE_DIR": root / "cache" / "uv",
        "UV_PYTHON_INSTALL_DIR": root / "tools" / "python",
        "TEMP": root / "tmp",
        "TMP": root / "tmp",
        "TMPDIR": root / "tmp",
    }
    # Reuse an explicitly configured or previously downloaded model receipt.
    previous = Path.home() / ".cache" / "image2editable" / "models" / "runtime"
    configured = os.environ.get("IMAGE2EDITABLE_MODEL_CACHE")
    directories["IMAGE2EDITABLE_MODEL_CACHE"] = (
        Path(configured).expanduser().resolve() if configured else
        previous if (previous / "runtime-receipt.json").is_file() else
        root / "models" / "runtime"
    )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    values = {name: str(path) for name, path in directories.items()}
    values["PYTHONIOENCODING"] = "utf-8"
    values["PYTHONNOUSERSITE"] = "1"
    executable_dirs = [root / "venv" / ("Scripts" if sys.platform == "win32" else "bin")]
    if sys.platform == "win32":
        executable_dirs.extend([root / "tools" / "python", root / "tools" / "git" / "cmd"])
        renderer = root / "tools" / "native-renderer" / "extracted" / "program" / "soffice.com"
        if renderer.is_file() and not os.environ.get("IMAGE2EDITABLE_LIBREOFFICE"):
            values["IMAGE2EDITABLE_LIBREOFFICE"] = str(renderer)
    else:
        executable_dirs.append(root / "tools" / "git" / "bin")
    values["PATH"] = os.pathsep.join([*(str(path) for path in executable_dirs), os.environ.get("PATH", "")])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=argparse.REMAINDER,
                        help="Run a command with the selected cache and temporary paths")
    args = parser.parse_args()
    root = installation_root()
    values = environment(root)
    if args.run:
        return subprocess.run(args.run, env={**os.environ, **values}, check=False).returncode
    print(json.dumps({"root": str(root), "environment": values}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
