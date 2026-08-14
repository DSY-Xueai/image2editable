import re
import types
from pathlib import Path

from scripts import visual_segment


SAM_PIN = "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
STANDALONE_REQUIREMENTS = ROOT / "skills" / "image-to-ppt" / "references" / "requirements.txt"
SKILL = ROOT / "skills" / "image-to-ppt" / "SKILL.md"
README = ROOT / "README.md"
README_EN = ROOT / "README_EN.md"
ROOT_VISUAL_SEGMENT = ROOT / "scripts" / "visual_segment.py"
STANDALONE_VISUAL_SEGMENT = (
    ROOT / "skills" / "image-to-ppt" / "scripts" / "visual_segment.py"
)
PYPROJECT = ROOT / "pyproject.toml"


def test_host_pptx_routes_screenshot_decisions_before_execute() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    host_section_match = re.search(
        r"Host 运行先准备并推进到 `awaiting_agent`："
        r"(?P<section>.*?)第一次 `agent next`",
        skill_text,
        re.DOTALL,
    )

    assert host_section_match is not None
    host_section = host_section_match.group("section")
    host_example_match = re.search(
        r"```bash\n(?P<commands>.*?)\n```",
        host_section,
        re.DOTALL,
    )

    assert host_example_match is not None
    commands = host_example_match.group("commands")
    host_prose = re.sub(r"```bash\n.*?\n```", "", host_section, flags=re.DOTALL)
    command_lines = []
    comment_lines = []
    for line in commands.splitlines():
        if line.lstrip().startswith("#"):
            comment_lines.append(line.lstrip()[1:].strip())
        command = line.split("#", 1)[0].strip()
        if command:
            command_lines.append(command)
    prepare = (
        "image2editable prepare input.pptx --run-dir runs/pptx-job "
        "--agent-provider host"
    )
    run_next = "image2editable run next runs/pptx-job"
    decision_record = "image2editable decision record runs/pptx-job \\"
    run_execute = "image2editable run execute runs/pptx-job"
    agent_next = "image2editable agent next runs/pptx-job"
    agent_record = "image2editable agent record runs/pptx-job --plan response.json"
    candidate_mapping = (
        "--page candidate.page_id --object candidate.source_shape_id"
    )

    def command_indexes(expected: str) -> list[int]:
        return [
            index
            for index, line in enumerate(command_lines)
            if line == expected
        ]

    prepare_indexes = command_indexes(prepare)
    run_next_indexes = command_indexes(run_next)
    decision_record_indexes = command_indexes(decision_record)
    run_execute_indexes = command_indexes(run_execute)
    agent_next_indexes = command_indexes(agent_next)
    agent_record_indexes = command_indexes(agent_record)

    assert re.search(
        r"每个非 `null` 的 `candidate`.*`decision record`.*`run next`",
        host_prose,
    )
    assert re.search(
        r"仅当返回对象的 `candidate` 字段为 `null` 时.*首次.*`run execute`",
        host_prose,
    )
    assert f"`{candidate_mapping}`" in host_prose
    assert any(candidate_mapping in comment for comment in comment_lines)
    assert len(prepare_indexes) == 1
    assert len(run_next_indexes) == 2
    assert len(decision_record_indexes) == 1
    assert command_lines[decision_record_indexes[0] + 1] == (
        f"{candidate_mapping} \\"
    )
    assert len(run_execute_indexes) == 2
    assert len(agent_next_indexes) == 1
    assert len(agent_record_indexes) == 1
    assert (
        prepare_indexes[0]
        < run_next_indexes[0]
        < decision_record_indexes[0]
        < run_next_indexes[-1]
        < run_execute_indexes[0]
        < agent_next_indexes[0]
        < agent_record_indexes[0]
        < run_execute_indexes[1]
    )


def test_sam_ref_is_full_commit_sha() -> None:
    sam_line = next(
        line
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.startswith("SAM-2 @ git+")
    )

    assert re.fullmatch(r"[0-9a-f]{40}", sam_line.rsplit("@", 1)[1])


def test_requirements_contains_sam_pin() -> None:
    assert SAM_PIN in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_requirements_has_no_sam_main_reference() -> None:
    assert not any(
        line.startswith("SAM-2 @ git+") and line.endswith("@main")
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    )


def test_standalone_sam_dependency_matches_product_pin() -> None:
    root_lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    standalone_lines = STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()

    assert root_lines.count(SAM_PIN) == 1
    assert standalone_lines.count(SAM_PIN) == 1


def test_standalone_sam_dependency_does_not_follow_a_branch() -> None:
    sam_lines = [
        line
        for line in STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.startswith("SAM-2 @ git+")
    ]

    assert sam_lines == [SAM_PIN]
    assert re.fullmatch(r"[0-9a-f]{40}", sam_lines[0].rsplit("@", 1)[1])


def test_standalone_declares_accelerate_used_by_visual_segmentation(
    monkeypatch,
) -> None:
    expected = "accelerate>=0.26.0"
    events = []

    assert (
        ROOT_VISUAL_SEGMENT.read_bytes()
        == STANDALONE_VISUAL_SEGMENT.read_bytes()
    )

    class EmptyWeights:
        def __enter__(self):
            events.append("empty-enter")

        def __exit__(self, *args):
            events.append("empty-exit")

    class Model:
        def load_state_dict(self, state, assign=False):
            return [], []

        def eval(self):
            return self

    model = Model()
    build_sam = types.SimpleNamespace(
        compose=lambda **kwargs: {"model": "config"},
        OmegaConf=types.SimpleNamespace(resolve=lambda config: None),
        instantiate=lambda config, **kwargs: (
            events.append("instantiate") or model
        ),
    )
    torch = types.SimpleNamespace(load=lambda *args, **kwargs: {"model": {}})

    def fake_import(name):
        events.append(name)
        return types.SimpleNamespace(init_empty_weights=lambda: EmptyWeights())

    monkeypatch.setattr(visual_segment.importlib, "import_module", fake_import)
    visual_segment._build_resource_safe_sam_model(
        build_sam,
        torch,
        Path("sam.pt"),
        "cpu",
    )

    assert events == ["accelerate", "empty-enter", "instantiate", "empty-exit"]
    assert expected in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert expected in STANDALONE_REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_cross_platform_docs_prefer_the_verified_current_environment() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "优先使用 Linux/WSL" not in skill_text
    assert "优先使用当前平台" in skill_text
    assert "通过 `doctor`" in skill_text
    assert "优先使用当前平台" in readme_text
    assert "通过 `doctor`" in readme_text
    assert "prefer Linux/WSL" not in readme_en_text
    assert "current platform" in readme_en_text
    assert "passes `doctor`" in readme_en_text
    assert "设备预检" in skill_text
    assert "设备预检" in readme_text
    assert "device preflight" in readme_en_text
    probe = (
        "python -c \"import sys, torch; print({'platform': sys.platform, "
        "'cuda': torch.cuda.is_available(), 'rocm': torch.version.hip})\""
    )
    assert probe in skill_text
    assert probe in readme_text
    assert probe in readme_en_text
    assert "我这台机器" not in skill_text + readme_text
    assert "this machine" not in readme_en_text.lower()


def test_cross_platform_docs_define_supported_device_interfaces() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    for text in (skill_text, readme_text):
        assert "Windows/Linux" in text
        assert "PyTorch" in text
        assert "CUDA" in text
        assert "ROCm" in text
        assert "真实 Apple Silicon 回归前" in text
        assert "不把 MPS 自动设为新默认" in text

    assert "Windows and Linux" in readme_en_text
    assert "PyTorch" in readme_en_text
    assert "CUDA" in readme_en_text
    assert "ROCm" in readme_en_text
    assert "real Apple Silicon regression testing" in readme_en_text
    assert "MPS will not become a new automatic default" in readme_en_text


def test_cross_platform_docs_keep_the_full_quality_model_on_cpu() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "SAM 2.1 large" in skill_text
    assert "CPU 仍运行完整模型" in skill_text
    assert "相同质量门禁" in skill_text
    assert "不替换为轻量分割模型" in skill_text
    assert "显著较慢" in skill_text
    assert "SAM 2.1 Large" in readme_text
    assert "CPU 仍运行完整模型" in readme_text
    assert "相同质量门禁" in readme_text
    assert "不替换为轻量分割模型" in readme_text
    assert "显著慢" in readme_text
    assert "SAM 2.1 Large" in readme_en_text
    assert "CPU still runs the full model" in readme_en_text
    assert "same quality gates" in readme_en_text
    assert "does not switch to a lightweight segmentation model" in readme_en_text
    assert "significantly slower" in readme_en_text


def test_readmes_do_not_promise_hardware_specific_or_uniform_speedups() -> None:
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "不对特定 GPU 型号或统一加速倍数作承诺" in readme_text
    assert (
        "does not promise results for a specific GPU model or a uniform speedup factor"
        in readme_en_text
    )


def test_requirements_keeps_pillow_floor() -> None:
    assert "Pillow>=9.0.0" in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_pyproject_keeps_supported_python_range() -> None:
    assert 'requires-python = ">=3.10,<3.13"' in PYPROJECT.read_text(
        encoding="utf-8"
    ).splitlines()
