from __future__ import annotations

import re
from pathlib import Path
import textwrap

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
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / "constraints" / "runtime.txt"
EXPECTED_INSTALLED_MATRIX = {
    ("ubuntu-latest", "3.10"),
    ("ubuntu-latest", "3.11"),
    ("ubuntu-latest", "3.12"),
    ("windows-latest", "3.12"),
    ("macos-latest", "3.12"),
}
FAST_INSTALL_COMMAND = (
    "python -m pip install --constraint constraints/runtime.txt pytest PyYAML "
    "pypdf reportlab python-pptx opencv-python-headless Pillow numpy pypdfium2"
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
    "python -m pip install --constraint constraints/runtime.txt "
    '"${{ env.WHEEL_PATH }}[test]"'
)
SMOKE_COMMAND = (
    'python "${{ github.workspace }}/scripts/installed_package_smoke.py" '
    '--checkout-root "${{ github.workspace }}"'
)


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
    document = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=StrictSafeLoader)
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
    pip_installs = [
        command
        for step in _steps(installed)
        if isinstance(step.get("run"), str)
        for command in _commands(step)
        if re.search(r"\bpip install\b", command)
    ]
    assert pip_installs == [INSTALL_WHEEL_COMMAND]

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


def test_ci_parser_is_declared_and_candidate_pinned() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert project["optional-dependencies"]["test"].count("PyYAML>=6,<7") == 1
    constraint_lines = [
        line.strip()
        for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert constraint_lines.count("PyYAML==6.0.3") == 1
