from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts import skill_environment as setup
from scripts.release_notes import extract_notes
from scripts.verify_skill_runtime import verify


@pytest.mark.parametrize("available,denied,expected", [
    (["D", "E"], [], "D"), (["E", "F"], [], "E"),
    (["D", "E"], ["D"], "E"), ([], [], "system/image2editable"),
])
def test_windows_installation_prefers_data_drive(monkeypatch, tmp_path, available, denied, expected):
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup, "windows_data_drives", lambda: [tmp_path / name for name in available])
    monkeypatch.setattr(setup.Path, "home", lambda: tmp_path / "system")
    original = setup.writable_root

    def writable(path):
        if path.parent.name in denied:
            raise PermissionError(str(path))
        return original(path)

    monkeypatch.setattr(setup, "writable_root", writable)
    wanted = tmp_path / expected
    if available:
        wanted /= "image2editable"
    assert setup.installation_root() == wanted


def test_inaccessible_data_drive_does_not_fall_back_to_system(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.sys, "platform", "win32")
    monkeypatch.setattr(setup, "windows_data_drives", lambda: [tmp_path / "D"])
    monkeypatch.setattr(setup, "writable_root", lambda path: (_ for _ in ()).throw(PermissionError(str(path))))
    with pytest.raises(OSError, match="Data drives exist"):
        setup.installation_root()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("has_data", [True, False])
def test_unix_storage_selection(monkeypatch, tmp_path, platform, has_data):
    monkeypatch.setattr(setup.sys, "platform", platform)
    monkeypatch.setattr(setup, "mounted_data_volumes", lambda: [tmp_path / "data"] if has_data else [])
    monkeypatch.setattr(setup.Path, "home", lambda: tmp_path / "user")
    expected = tmp_path / ("data/image2editable" if has_data else "user/.local/share/image2editable")
    assert setup.installation_root() == expected


def test_linux_excludes_system_disk_and_loop_devices(monkeypatch):
    monkeypatch.setattr(setup.sys, "platform", "linux")
    devices = {"blockdevices": [
        {"type": "disk", "children": [{"mountpoints": ["/"]}, {"mountpoints": ["/home"]}]},
        {"type": "disk", "children": [{"mountpoints": ["/mnt/data"]}]},
        {"type": "loop", "mountpoints": ["/snap/app"]},
        {"type": "disk", "mountpoints": [None]},
    ]}
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(devices)))
    assert setup.mounted_data_volumes() == [Path("/mnt/data")]


def test_macos_excludes_system_volume_and_installer_images(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    paths = [tmp_path / name for name in ("System", "Data", "Installer", "Network")]
    monkeypatch.setattr(setup.Path, "iterdir", lambda self: iter(paths))
    monkeypatch.setattr(setup.Path, "is_mount", lambda self: True)
    monkeypatch.setattr(setup.Path, "stat", lambda self: SimpleNamespace(st_dev=1 if self not in paths or self.name == "System" else 2))

    def run(command, **kwargs):
        name = Path(command[-1]).name
        info = {"DeviceNode": "/dev/disk2s1", "BusProtocol": "USB"}
        if name == "Installer":
            info["BusProtocol"] = "Disk Image"
        return SimpleNamespace(returncode=1 if name == "Network" else 0,
                               stdout=setup.plistlib.dumps(info))

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.mounted_data_volumes() == [tmp_path / "Data"]


def test_cache_environment_and_existing_models(monkeypatch, tmp_path):
    monkeypatch.delenv("IMAGE2EDITABLE_MODEL_CACHE", raising=False)
    monkeypatch.setattr(setup.Path, "home", lambda: tmp_path / "user")
    root = tmp_path / "data"
    values = setup.environment(root)
    for key in ("PIP_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "PADDLE_PDX_CACHE_HOME",
                "TORCH_HOME", "TEMP", "TMP", "TMPDIR", "IMAGE2EDITABLE_MODEL_CACHE"):
        assert Path(values[key]).is_relative_to(root)
        assert Path(values[key]).is_dir()
    previous = tmp_path / "user/.cache/image2editable/models/runtime"
    previous.mkdir(parents=True)
    (previous / "runtime-receipt.json").write_text("{}")
    assert setup.environment(root)["IMAGE2EDITABLE_MODEL_CACHE"] == str(previous)
    configured = tmp_path / "explicit-models"
    monkeypatch.setenv("IMAGE2EDITABLE_MODEL_CACHE", str(configured))
    assert setup.environment(root)["IMAGE2EDITABLE_MODEL_CACHE"] == str(configured)
    assert "HOME" not in values and "USERPROFILE" not in values


def test_command_receives_selected_environment_and_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "installation_root", lambda: tmp_path)
    monkeypatch.setattr(setup.sys, "argv", ["skill_environment.py", "--run", "tool", "a path with spaces"])
    monkeypatch.setenv("TASK_EXISTING_SETTING", "preserved")

    def run(command, *, env, check):
        assert command == ["tool", "a path with spaces"]
        assert env["TASK_EXISTING_SETTING"] == "preserved"
        assert Path(env["TEMP"]).is_relative_to(tmp_path)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(setup.subprocess, "run", run)
    assert setup.main() == 7


def test_release_notes_extract_only_requested_version():
    changelog = "# Changes\n## [Unreleased]\nfuture\n## [0.3.0]\n### Fixes\ncurrent\n## [0.2.0]\nold\n"
    assert extract_notes(changelog, "v0.3.0") == "### Fixes\ncurrent\n"
    with pytest.raises(ValueError):
        extract_notes(changelog, "v0.4.0")


def test_installed_code_detects_old_same_version_missing_and_obsolete_files(tmp_path):
    source, installed = tmp_path / "source", tmp_path / "installed"
    names = ["image2editable/cli.py", "scripts/__init__.py", "image_to_ppt.py",
             "image_to_psd.py", "image2editable/runtime_model_catalog.json"]
    for name in names:
        for base in (source, installed):
            (base / name).parent.mkdir(parents=True, exist_ok=True)
            (base / name).write_text("current", encoding="utf-8")
    distribution = SimpleNamespace(files=names, locate_file=lambda name: installed / name)
    assert verify(source, distribution) == []
    (installed / "image2editable/cli.py").write_text("old", encoding="utf-8")
    (installed / "image_to_ppt.py").unlink()
    names.append("scripts/obsolete.py")
    assert verify(source, distribution) == [
        "different: image2editable/cli.py", "missing: image_to_ppt.py", "obsolete: scripts/obsolete.py",
    ]


@pytest.mark.parametrize("skill", ["image-to-ppt", "image-to-psd"])
def test_bundled_setup_tools_match_source(skill):
    root = Path(__file__).resolve().parents[1]
    for name in ("skill_environment.py", "verify_skill_runtime.py", "fetch_skill_source.py"):
        assert (root / "scripts" / name).read_bytes() == (root / "skills" / skill / "scripts" / name).read_bytes()
