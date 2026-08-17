from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
import subprocess
import textwrap
from types import SimpleNamespace

import pytest
import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import Resolver

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release-gate.yml"
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "constraints" / "runtime.txt"
EXPECTED_INSTALLED_MATRIX = {
    ("ubuntu-latest", "3.10"),
    ("ubuntu-latest", "3.11"),
    ("ubuntu-latest", "3.12"),
    ("windows-latest", "3.12"),
    ("macos-latest", "3.12"),
}
EXPECTED_RELEASE_MATRIX = {
    (os_name, version)
    for os_name in ("ubuntu-latest", "windows-latest", "macos-latest")
    for version in ("3.10", "3.11", "3.12")
}
FAST_INSTALL_COMMAND = (
    "python -m pip install --constraint constraints/runtime.txt pytest PyYAML "
    "pypdf reportlab python-pptx opencv-python-headless Pillow numpy pypdfium2 torch"
)
BUILD_INSTALL_COMMAND = "python -m pip install build twine"
FAST_PYTEST_COMMAND = (
    'python -m pytest "${{ github.workspace }}/tests" --import-mode=importlib '
    '-m "not powerpoint" -q'
)
LOCATE_WHEEL_COMMANDS = [
    "from pathlib import Path",
    "import os",
    'wheels = list((Path(os.environ["RUNNER_TEMP"]) / "distribution").glob("*.whl"))',
    "if len(wheels) != 1:",
    '    raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")',
    'with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as env_file:',
    '    print(f"WHEEL_PATH={wheels[0]}", file=env_file)',
]
INSTALL_WHEEL_COMMAND = (
    "python -m pip install --constraint constraints/runtime.txt --no-build-isolation "
    "--no-binary antlr4-python3-runtime "
    '"${{ env.WHEEL_PATH }}[test]"'
)
BOOTSTRAP_BUILD_RUNTIME_COMMAND = (
    "python -m pip install --constraint constraints/runtime.txt "
    "--extra-index-url https://download.pytorch.org/whl/cpu "
    "torch torchvision setuptools==84.0.0"
)
SMOKE_COMMAND = (
    'python "${{ github.workspace }}/scripts/installed_package_smoke.py" '
    '--checkout-root "${{ github.workspace }}"'
)
SMOKE_MODULES = ("image2editable", "scripts", "image_to_ppt")


class StrictSafeLoader(yaml.SafeLoader):
    pass


StrictSafeLoader.yaml_implicit_resolvers = {
    key: [
        (tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
StrictSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _construct_unique_mapping(
    loader: StrictSafeLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictSafeLoader.add_constructor(
    Resolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _workflow() -> dict[str, object]:
    return _load_workflow(WORKFLOW)


def _release_workflow() -> dict[str, object]:
    return _load_workflow(RELEASE_WORKFLOW)


def _load_workflow(path: Path) -> dict[str, object]:
    document = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictSafeLoader)
    assert isinstance(document, dict)
    return document


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list) and all(isinstance(step, dict) for step in steps)
    return steps


def _runs(job: dict[str, object]) -> list[str]:
    return [step["run"] for step in _steps(job) if isinstance(step.get("run"), str)]


def _commands(step: dict[str, object]) -> list[str]:
    run = step.get("run")
    assert isinstance(run, str)
    return [
        line.rstrip()
        for line in textwrap.dedent(run).strip().splitlines()
        if line.strip()
    ]


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    matches = [step for step in _steps(job) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _step_id(step: dict[str, object]) -> str:
    uses = step.get("uses")
    if isinstance(uses, str):
        return uses.split("@", 1)[0]
    name = step.get("name")
    assert isinstance(name, str)
    return name


def _action_step(job: dict[str, object], action: str) -> dict[str, object]:
    matches = [
        step
        for step in _steps(job)
        if isinstance(step.get("uses"), str) and step["uses"].split("@", 1)[0] == action
    ]
    assert len(matches) == 1
    return matches[0]


def _needs(job: dict[str, object]) -> set[str]:
    value = job.get("needs", [])
    if isinstance(value, str):
        return {value}
    assert isinstance(value, list) and all(isinstance(item, str) for item in value)
    return set(value)


def _all_mappings(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _all_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_mappings(child)


def _smoke_module():
    return importlib.import_module("scripts.installed_package_smoke")


def _configure_installed_distribution(smoke, tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    purelib = tmp_path / "venv" / "site-packages"
    scripts_root = purelib / "bin"
    launcher = scripts_root / "image2editable.exe"
    module_paths = {
        "image2editable": purelib / "image2editable" / "__init__.py",
        "scripts": purelib / "scripts" / "__init__.py",
        "image_to_ppt": purelib / "image_to_ppt.py",
    }
    catalog_paths = [
        purelib / "image2editable" / "model_catalog.json",
        purelib / "image2editable" / "runtime_model_catalog.json",
    ]
    for path in [*module_paths.values(), *catalog_paths, launcher]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    checkout.mkdir()

    modules = {
        name: SimpleNamespace(__file__=str(path))
        for name, path in module_paths.items()
    }
    distribution_files = [
        path.relative_to(purelib)
        for path in [*module_paths.values(), *catalog_paths, launcher]
    ]
    distribution = SimpleNamespace(
        files=distribution_files,
        locate_file=lambda path: purelib / path,
        entry_points=[
            SimpleNamespace(
                group="console_scripts",
                name="image2editable",
                value="image2editable.cli:main",
            )
        ],
    )
    monkeypatch.setattr(smoke.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        smoke.metadata,
        "distribution",
        lambda name: distribution if name == "image2editable" else None,
    )
    monkeypatch.setattr(
        smoke.sysconfig,
        "get_paths",
        lambda: {
            "purelib": str(purelib),
            "platlib": str(purelib),
            "scripts": str(scripts_root),
        },
    )
    monkeypatch.setattr(
        smoke,
        "shutil",
        SimpleNamespace(which=lambda name: str(launcher)),
        raising=False,
    )
    return checkout, purelib, modules, catalog_paths, launcher, distribution


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_ci_has_exact_jobs_and_supported_install_matrix() -> None:
    workflow = _workflow()
    assert set(workflow) == {"name", "on", "permissions", "jobs"}
    assert workflow["name"] == "CI"
    assert workflow["on"] == {
        "push": {"branches": ["main"]},
        "pull_request": None,
    }
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {
        "fast-model-free",
        "build-distribution",
        "installed-package",
    }
    for job in jobs.values():
        assert "with" not in _action_step(job, "actions/checkout")

    fast = jobs["fast-model-free"]
    assert "env" not in fast
    assert fast["runs-on"] == "ubuntu-latest"
    assert "strategy" not in fast
    setup_fast = _action_step(fast, "actions/setup-python")
    assert setup_fast["with"] == {"python-version": "3.12"}

    build = jobs["build-distribution"]
    assert build["runs-on"] == "ubuntu-latest"
    assert "strategy" not in build
    setup_build = _action_step(build, "actions/setup-python")
    assert setup_build["with"] == {"python-version": "3.12"}

    installed = jobs["installed-package"]
    assert "env" not in installed
    assert "defaults" not in installed
    assert installed["runs-on"] == "${{ matrix.os }}"
    strategy = installed["strategy"]
    assert set(strategy) == {"fail-fast", "matrix"}
    assert strategy["fail-fast"] is False
    matrix = strategy["matrix"]
    assert set(matrix) == {"include"}
    include = matrix["include"]
    assert all(set(entry) == {"os", "python-version"} for entry in include)
    actual = {(entry["os"], entry["python-version"]) for entry in include}
    assert len(include) == 5
    assert actual == EXPECTED_INSTALLED_MATRIX
    setup_installed = _action_step(installed, "actions/setup-python")
    assert setup_installed["with"] == {
        "python-version": "${{ matrix.python-version }}"
    }


def test_fast_job_runs_checkout_tests_without_installing_the_project() -> None:
    fast = _workflow()["jobs"]["fast-model-free"]
    assert "defaults" not in fast
    assert [_step_id(step) for step in _steps(fast)] == [
        "actions/checkout",
        "actions/setup-python",
        "Install model-free test dependencies",
        "Run model-free tests",
    ]
    install = _named_step(fast, "Install model-free test dependencies")
    assert _commands(install) == [FAST_INSTALL_COMMAND]
    pip_installs = [
        command
        for step in _steps(fast)
        if isinstance(step.get("run"), str)
        for command in _commands(step)
        if re.search(r"\bpip install\b", command)
    ]
    assert pip_installs == [FAST_INSTALL_COMMAND]

    pytest_step = _named_step(fast, "Run model-free tests")
    assert pytest_step["working-directory"] == "${{ runner.temp }}"
    assert pytest_step["env"] == {"PYTHONPATH": "${{ github.workspace }}"}
    assert _commands(pytest_step) == [FAST_PYTEST_COMMAND]
    assert all(
        "env" not in step
        for step in _steps(fast)
        if step is not pytest_step
    )


def test_installed_package_uses_the_built_wheel_outside_checkout() -> None:
    jobs = _workflow()["jobs"]
    build = jobs["build-distribution"]
    installed = jobs["installed-package"]
    assert [_step_id(step) for step in _steps(build)] == [
        "actions/checkout",
        "actions/setup-python",
        "Install distribution tools",
        "Build and check distributions",
        "actions/upload-artifact",
    ]
    assert [_step_id(step) for step in _steps(installed)] == [
        "actions/checkout",
        "actions/setup-python",
        "actions/download-artifact",
        "Locate the built wheel",
        "Bootstrap pinned build runtime",
        "Install built wheel",
        "Check installed dependencies",
        "installed_package_smoke",
        "Run tests against installed wheel",
    ]

    assert _needs(installed) == {"build-distribution"}
    upload = _action_step(build, "actions/upload-artifact")
    download = _action_step(installed, "actions/download-artifact")
    assert upload["with"] == {
        "name": "distribution",
        "path": "dist/*",
        "if-no-files-found": "error",
    }
    assert download["with"] == {
        "name": "distribution",
        "path": "${{ runner.temp }}/distribution",
    }

    distribution = _named_step(build, "Build and check distributions")
    build_tools = _named_step(build, "Install distribution tools")
    assert _commands(build_tools) == [BUILD_INSTALL_COMMAND]
    assert _commands(distribution) == [
        "python -m build",
        "python -m twine check dist/*",
    ]

    locate = _named_step(installed, "Locate the built wheel")
    assert locate["shell"] == "python"
    assert _commands(locate) == LOCATE_WHEEL_COMMANDS

    install = _named_step(installed, "Install built wheel")
    assert _commands(install) == [INSTALL_WHEEL_COMMAND]
    bootstrap = _named_step(installed, "Bootstrap pinned build runtime")
    assert _commands(bootstrap) == [BOOTSTRAP_BUILD_RUNTIME_COMMAND]
    pip_installs = [
        command
        for step in _steps(installed)
        if isinstance(step.get("run"), str)
        for command in _commands(step)
        if re.search(r"\bpip install\b", command)
    ]
    assert pip_installs == [BOOTSTRAP_BUILD_RUNTIME_COMMAND, INSTALL_WHEEL_COMMAND]

    pip_check = _named_step(installed, "Check installed dependencies")
    assert _commands(pip_check) == ["python -m pip check"]

    smoke = _named_step(installed, "installed_package_smoke")
    assert smoke["working-directory"] == "${{ runner.temp }}"
    assert _commands(smoke) == [SMOKE_COMMAND]

    installed_tests = _named_step(installed, "Run tests against installed wheel")
    assert installed_tests["working-directory"] == "${{ runner.temp }}"
    installed_test_env = installed_tests.get("env", {})
    assert isinstance(installed_test_env, dict)
    assert "PYTHONPATH" not in installed_test_env
    assert _commands(installed_tests) == [FAST_PYTEST_COMMAND]
    for step in _steps(installed):
        env = step.get("env", {})
        assert isinstance(env, dict)
        assert not any(key.casefold() == "pythonpath" for key in env)
        if step is not locate:
            assert "shell" not in step


def test_ci_cannot_hide_failures_and_pins_every_action() -> None:
    workflow = _workflow()
    for mapping in _all_mappings(workflow):
        assert "continue-on-error" not in mapping
        assert "if" not in mapping
    for run in (run for job in workflow["jobs"].values() for run in _runs(job)):
        assert not re.search(r"(?:\|\|\s*true\b|;\s*true\b|\bset\s+\+e\b)", run)
    for job in workflow["jobs"].values():
        assert "defaults" not in job
        for step in _steps(job):
            if step.get("name") == "Locate the built wheel":
                assert step.get("shell") == "python"
            else:
                assert "shell" not in step

    uses = [
        step["uses"]
        for job in workflow["jobs"].values()
        for step in _steps(job)
        if isinstance(step.get("uses"), str)
    ]
    assert uses
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action) for action in uses)


def test_release_gate_is_manual_and_has_exact_jobs_and_matrix() -> None:
    workflow = _release_workflow()
    assert set(workflow) == {"name", "on", "permissions", "jobs"}
    assert workflow["name"] == "Release Gate"
    assert workflow["on"] == {
        "workflow_dispatch": {
            "inputs": {
                "run_real_model_smoke": {
                    "description": "Run protected real-model smoke",
                    "required": True,
                    "type": "boolean",
                    "default": False,
                }
            }
        }
    }
    assert workflow["permissions"] == {"contents": "read"}

    jobs = workflow["jobs"]
    assert set(jobs) == {
        "build-distribution",
        "installed-package",
        "real-model-smoke",
    }
    installed = jobs["installed-package"]
    assert installed["runs-on"] == "${{ matrix.os }}"
    assert _needs(installed) == {"build-distribution"}
    assert installed["strategy"] == {
        "fail-fast": False,
        "matrix": {
            "include": [
                {"os": os_name, "python-version": version}
                for os_name, version in sorted(EXPECTED_RELEASE_MATRIX)
            ]
        },
    }
    assert _action_step(installed, "actions/setup-python")["with"] == {
        "python-version": "${{ matrix.python-version }}"
    }


def test_release_gate_builds_once_and_tests_the_same_artifact() -> None:
    jobs = _release_workflow()["jobs"]
    build = jobs["build-distribution"]
    installed = jobs["installed-package"]
    assert build["runs-on"] == "ubuntu-latest"
    assert [_step_id(step) for step in _steps(build)] == [
        "actions/checkout",
        "actions/setup-python",
        "Install distribution tools",
        "Build and check distributions",
        "actions/upload-artifact",
    ]
    assert "with" not in _action_step(build, "actions/checkout")
    assert _action_step(build, "actions/setup-python")["with"] == {
        "python-version": "3.12"
    }
    assert _commands(_named_step(build, "Install distribution tools")) == [
        BUILD_INSTALL_COMMAND
    ]
    assert _commands(_named_step(build, "Build and check distributions")) == [
        "python -m build",
        "python -m twine check dist/*",
    ]
    assert _action_step(build, "actions/upload-artifact")["with"] == {
        "name": "distribution",
        "path": "dist/*",
        "if-no-files-found": "error",
    }

    assert [_step_id(step) for step in _steps(installed)] == [
        "actions/checkout",
        "actions/setup-python",
        "actions/download-artifact",
        "Locate the built wheel",
        "Bootstrap pinned build runtime",
        "Install built wheel",
        "Check installed dependencies",
        "installed_package_smoke",
        "Run tests against installed wheel",
    ]
    assert "with" not in _action_step(installed, "actions/checkout")
    assert _action_step(installed, "actions/download-artifact")["with"] == {
        "name": "distribution",
        "path": "${{ runner.temp }}/distribution",
    }
    assert _commands(_named_step(installed, "Locate the built wheel")) == (
        LOCATE_WHEEL_COMMANDS
    )
    assert _named_step(installed, "Locate the built wheel")["shell"] == "python"
    assert _commands(_named_step(installed, "Bootstrap pinned build runtime")) == [
        BOOTSTRAP_BUILD_RUNTIME_COMMAND
    ]
    assert _commands(_named_step(installed, "Install built wheel")) == [
        INSTALL_WHEEL_COMMAND
    ]
    assert _commands(_named_step(installed, "Check installed dependencies")) == [
        "python -m pip check"
    ]
    installed_smoke = _named_step(installed, "installed_package_smoke")
    assert installed_smoke["working-directory"] == "${{ runner.temp }}"
    assert _commands(installed_smoke) == [SMOKE_COMMAND]
    installed_tests = _named_step(installed, "Run tests against installed wheel")
    assert installed_tests["working-directory"] == "${{ runner.temp }}"
    assert _commands(installed_tests) == [FAST_PYTEST_COMMAND]


def test_release_gate_real_models_are_protected_and_explicitly_opt_in() -> None:
    job = _release_workflow()["jobs"]["real-model-smoke"]
    assert job["runs-on"] == "ubuntu-latest"
    assert _needs(job) == {"build-distribution"}
    assert job["if"] == "${{ inputs.run_real_model_smoke }}"
    assert job["environment"] == "real-model-smoke"
    assert "strategy" not in job
    assert [_step_id(step) for step in _steps(job)] == [
        "Require protected model-smoke approval",
        "actions/checkout",
        "actions/setup-python",
        "actions/download-artifact",
        "Locate the built wheel",
        "Bootstrap pinned build runtime",
        "Install built wheel",
        "Verify installed model smoke",
        "Install Tesseract OCR",
        "Check installed dependencies",
        "Install runtime models",
        "Run doctor",
        "Run real model smoke",
    ]
    approval = _named_step(job, "Require protected model-smoke approval")
    assert approval["env"] == {
        "IMAGE2EDITABLE_REAL_MODEL_SMOKE_APPROVED": (
            "${{ secrets.IMAGE2EDITABLE_REAL_MODEL_SMOKE_APPROVED }}"
        )
    }
    assert _commands(approval) == [
        'test "${IMAGE2EDITABLE_REAL_MODEL_SMOKE_APPROVED}" = "approved"'
    ]
    assert _action_step(job, "actions/setup-python")["with"] == {
        "python-version": "3.12"
    }
    assert "with" not in _action_step(job, "actions/checkout")
    assert _action_step(job, "actions/download-artifact")["with"] == {
        "name": "distribution",
        "path": "${{ runner.temp }}/distribution",
    }
    assert _named_step(job, "Locate the built wheel")["shell"] == "python"
    assert _commands(_named_step(job, "Locate the built wheel")) == (
        LOCATE_WHEEL_COMMANDS
    )
    assert _commands(_named_step(job, "Bootstrap pinned build runtime")) == [
        BOOTSTRAP_BUILD_RUNTIME_COMMAND
    ]
    assert _commands(_named_step(job, "Install built wheel")) == [
        INSTALL_WHEEL_COMMAND
    ]
    installed_smoke = _named_step(job, "Verify installed model smoke")
    assert installed_smoke["shell"] == "python"
    assert _commands(installed_smoke) == [
        "from importlib.metadata import distribution",
        "import os",
        "from pathlib import Path, PurePosixPath",
        'relative = PurePosixPath("scripts/runtime_model_smoke.py")',
        'package = distribution("image2editable")',
        "if relative not in (package.files or []):",
        '    raise SystemExit("installed runtime model smoke is missing")',
        "installed = Path(package.locate_file(relative))",
        'checkout = Path(os.environ["GITHUB_WORKSPACE"], *relative.parts)',
        "if installed.read_bytes() != checkout.read_bytes():",
        '    raise SystemExit("installed runtime model smoke differs from checkout")',
    ]
    assert _commands(_named_step(job, "Install Tesseract OCR")) == [
        "sudo apt-get update",
        "sudo apt-get install --yes tesseract-ocr",
        (
            "python -m pip install --constraint constraints/runtime.txt "
            "pytesseract==0.3.13"
        ),
    ]
    assert _commands(_named_step(job, "Check installed dependencies")) == [
        "python -m pip check"
    ]
    assert _commands(_named_step(job, "Install runtime models")) == [
        "image2editable models install runtime --yes"
    ]
    assert _commands(_named_step(job, "Run doctor")) == ["image2editable doctor"]
    real_smoke = _named_step(job, "Run real model smoke")
    assert real_smoke["working-directory"] == "${{ runner.temp }}"
    assert _commands(real_smoke) == [
        'python "${{ github.workspace }}/scripts/runtime_model_smoke.py"'
    ]


def test_release_gate_cannot_hide_failures_or_use_unpinned_actions() -> None:
    workflow = _release_workflow()
    jobs = workflow["jobs"]
    for mapping in _all_mappings(workflow):
        assert "continue-on-error" not in mapping
        if mapping is not jobs["real-model-smoke"]:
            assert "if" not in mapping
    for job in jobs.values():
        assert "defaults" not in job
        assert "env" not in job
        for step in _steps(job):
            env = step.get("env", {})
            assert isinstance(env, dict)
            assert not any(key.casefold() == "pythonpath" for key in env)
            if step.get("name") in {
                "Locate the built wheel",
                "Verify installed model smoke",
            }:
                assert step.get("shell") == "python"
            else:
                assert "shell" not in step
        for run in _runs(job):
            assert not re.search(
                r"(?:\|\|\s*(?:true|:|exit\s+0)\b|;\s*(?:true|exit\s+0)\b|\bset\s+\+e\b)",
                run,
            )

    ci_action_shas = {
        step["uses"]
        for job in _workflow()["jobs"].values()
        for step in _steps(job)
        if isinstance(step.get("uses"), str)
    }
    release_actions = [
        step["uses"]
        for job in jobs.values()
        for step in _steps(job)
        if isinstance(step.get("uses"), str)
    ]
    assert release_actions
    assert set(release_actions) <= ci_action_shas
    assert all(
        re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", action)
        for action in release_actions
    )


def test_ci_parser_is_declared_and_candidate_pinned() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert project["optional-dependencies"]["test"].count("PyYAML>=6,<7") == 1
    constraint_lines = [
        line.strip()
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert constraint_lines.count("PyYAML==6.0.3") == 1


def test_smoke_verifies_modules_and_catalogs_from_the_installed_distribution(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, _, _, _, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )

    assert smoke.verify_imports(checkout) == SMOKE_MODULES


def test_smoke_rejects_an_import_from_the_checkout(tmp_path, monkeypatch) -> None:
    smoke = _smoke_module()
    checkout, _, modules, _, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    source_module = checkout / "image2editable" / "__init__.py"
    source_module.parent.mkdir()
    source_module.write_text("", encoding="utf-8")
    modules["image2editable"].__file__ = str(source_module)

    with pytest.raises(smoke.SmokeError):
        smoke.verify_imports(checkout)


def test_smoke_rejects_a_same_named_module_outside_the_distribution(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, purelib, modules, _, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    impostor = purelib / "impostor" / "image2editable.py"
    impostor.parent.mkdir()
    impostor.write_text("", encoding="utf-8")
    modules["image2editable"].__file__ = str(impostor)

    with pytest.raises(smoke.SmokeError):
        smoke.verify_imports(checkout)


def test_smoke_rejects_distribution_files_outside_the_install_scheme(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, _, _, _, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    unrelated = tmp_path / "unrelated-site-packages"
    monkeypatch.setattr(
        smoke.sysconfig,
        "get_paths",
        lambda: {"purelib": str(unrelated), "platlib": str(unrelated)},
    )

    with pytest.raises(smoke.SmokeError):
        smoke.verify_imports(checkout)


def test_smoke_rejects_missing_catalog_package_data(tmp_path, monkeypatch) -> None:
    smoke = _smoke_module()
    checkout, _, _, catalog_paths, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    catalog_paths[0].unlink()

    with pytest.raises(smoke.SmokeError):
        smoke.verify_imports(checkout)


def test_smoke_verifies_the_distribution_console_launcher(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, _, _, _, launcher, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )

    assert smoke._verified_launcher(checkout) == launcher.resolve()


def test_smoke_rejects_a_path_shadowed_console_launcher(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, _, _, _, _, _ = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    impostor = tmp_path / "path-shadow" / "image2editable.exe"
    impostor.parent.mkdir()
    impostor.write_text("", encoding="utf-8")
    monkeypatch.setattr(smoke.shutil, "which", lambda name: str(impostor))

    with pytest.raises(smoke.SmokeError):
        smoke._verified_launcher(checkout)


def test_smoke_rejects_wrong_console_entry_point_metadata(
    tmp_path, monkeypatch
) -> None:
    smoke = _smoke_module()
    checkout, _, _, _, _, distribution = _configure_installed_distribution(
        smoke, tmp_path, monkeypatch
    )
    distribution.entry_points = [
        SimpleNamespace(
            group="console_scripts",
            name="image2editable",
            value="other_package.cli:main",
        )
    ]

    with pytest.raises(smoke.SmokeError):
        smoke._verified_launcher(checkout)


def _stub_smoke_imports(smoke, monkeypatch, tmp_path) -> str:
    launcher = (tmp_path / "venv-bin" / "image2editable.exe").resolve()
    monkeypatch.setattr(smoke, "verify_imports", lambda checkout: SMOKE_MODULES)
    monkeypatch.setattr(
        smoke,
        "_verified_launcher",
        lambda checkout: launcher,
        raising=False,
    )
    return str(launcher)


def test_smoke_main_rejects_console_help_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    smoke = _smoke_module()
    launcher = _stub_smoke_imports(smoke, monkeypatch, tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(returncode=2, stderr="C:\\private\\help-error")

    monkeypatch.setattr(smoke.subprocess, "run", run)

    assert smoke.main(["--checkout-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"modules": list(SMOKE_MODULES), "ok": False}
    assert captured.err == ""
    assert [call[0] for call in calls] == [[launcher, "--help"]]
    assert "private" not in captured.out


def test_smoke_main_redacts_unexpected_internal_errors(
    tmp_path, monkeypatch, capsys
) -> None:
    smoke = _smoke_module()

    def fail(checkout):
        raise ValueError("C:\\Users\\private\\broken-metadata")

    monkeypatch.setattr(smoke, "verify_imports", fail)

    assert smoke.main(["--checkout-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"modules": [], "ok": False}
    assert captured.err == ""
    assert "private" not in captured.out


def test_smoke_main_redacts_console_timeout(tmp_path, monkeypatch, capsys) -> None:
    smoke = _smoke_module()
    _stub_smoke_imports(smoke, monkeypatch, tmp_path)

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            stderr="C:\\Users\\private\\timeout",
        )

    monkeypatch.setattr(smoke.subprocess, "run", timeout)

    assert smoke.main(["--checkout-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "modules": list(SMOKE_MODULES),
        "ok": False,
    }
    assert captured.err == ""
    assert "private" not in captured.out


@pytest.mark.parametrize(
    ("doctor", "expected_calls"),
    [
        (_completed(returncode=2, stdout='{"ready": false, "checks": {}}'), 2),
        (_completed(returncode=0, stdout="not-json"), 2),
        (_completed(returncode=0, stdout='{"ready": true, "checks": []}'), 2),
        (
            _completed(
                returncode=0,
                stdout='{"ready": true, "checks": {}, "extra": true}',
            ),
            2,
        ),
    ],
)
def test_smoke_main_rejects_invalid_doctor_results(
    tmp_path, monkeypatch, capsys, doctor, expected_calls
) -> None:
    smoke = _smoke_module()
    _stub_smoke_imports(smoke, monkeypatch, tmp_path)
    responses = iter([_completed(returncode=0), doctor])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(smoke.subprocess, "run", run)

    assert smoke.main(["--checkout-root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"modules": list(SMOKE_MODULES), "ok": False}
    assert captured.err == ""
    assert len(calls) == expected_calls


@pytest.mark.parametrize(("ready", "doctor_exit"), [(False, 1), (True, 0)])
def test_smoke_main_accepts_ready_false_and_true(
    tmp_path, monkeypatch, capsys, ready, doctor_exit
) -> None:
    smoke = _smoke_module()
    launcher = _stub_smoke_imports(smoke, monkeypatch, tmp_path)
    doctor = _completed(
        returncode=doctor_exit,
        stdout=json.dumps({"ready": ready, "checks": {"python": {}}}),
        stderr="C:\\private\\doctor-noise",
    )
    responses = iter([_completed(returncode=0, stdout="usage"), doctor])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(smoke.subprocess, "run", run)

    assert smoke.main(["--checkout-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"modules": list(SMOKE_MODULES), "ok": True}
    assert captured.err == ""
    assert [call[0] for call in calls] == [
        [launcher, "--help"],
        [launcher, "doctor"],
    ]
    assert all(
        call[1]["capture_output"] is True
        and call[1]["text"] is True
        and call[1]["check"] is False
        and call[1]["timeout"] == smoke.COMMAND_TIMEOUT_SECONDS
        for call in calls
    )
    assert "private" not in captured.out
