import json
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
CLAUDE_PLUGIN = ROOT / ".claude-plugin" / "plugin.json"
GITHUB_CI = ROOT / ".github" / "workflows" / "ci.yml"


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


def test_opencv_stays_within_the_verified_major_version() -> None:
    expected = "opencv-python>=4.5.0,<5"

    assert expected in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert expected in STANDALONE_REQUIREMENTS.read_text(
        encoding="utf-8"
    ).splitlines()


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


def test_skill_docs_prefer_the_verified_current_environment() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")

    assert "优先使用 Linux/WSL" not in skill_text
    assert "优先使用当前平台" in skill_text
    assert "通过 `doctor`" in skill_text
    assert "设备预检" in skill_text
    probe = (
        "python -c \"import sys, torch; print({'platform': sys.platform, "
        "'cuda': torch.cuda.is_available(), 'rocm': torch.version.hip})\""
    )
    assert probe in skill_text
    assert "我这台机器" not in skill_text


def test_skill_docs_define_supported_device_interfaces() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")

    assert "Windows/Linux" in skill_text
    assert "PyTorch" in skill_text
    assert "CUDA" in skill_text
    assert "ROCm" in skill_text
    assert "真实 Apple Silicon 回归前" in skill_text
    assert "不把 MPS 自动设为新默认" in skill_text


def test_skill_docs_keep_the_full_quality_model_on_cpu() -> None:
    skill_text = SKILL.read_text(encoding="utf-8")

    assert "SAM 2.1 large" in skill_text
    assert "CPU 仍运行完整模型" in skill_text
    assert "相同质量门禁" in skill_text
    assert "不替换为轻量分割模型" in skill_text
    assert "显著较慢" in skill_text


def test_readmes_keep_hardware_policy_out_of_quick_start() -> None:
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "### 运行环境" not in readme_text
    assert "### Runtime environment" not in readme_en_text
    assert "python -c \"import sys, torch" not in readme_text
    assert "python -c \"import sys, torch" not in readme_en_text


def test_readmes_document_ocr_installation_after_local_cli_install() -> None:
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")
    paddle_url = "https://www.paddlepaddle.org.cn/install/quick"
    tesseract_url = "https://tesseract-ocr.github.io/tessdoc/Installation.html"
    commands = (
        "python -m pip install paddleocr paddlepaddle",
        "tesseract --version",
        "python -m pip install pytesseract",
        "\nimage2editable doctor\n",
    )

    for text in (readme_text, readme_en_text):
        assert text.index("pip install .") < text.index(commands[0])
        assert text.index(commands[0]) < text.index(commands[1])
        assert text.index(commands[1]) < text.index(commands[2])
        assert text.index(commands[2]) < text.index(commands[3])
        assert paddle_url in text
        assert tesseract_url in text
        assert '"ready": true' in text
    assert "Codex、Claude Code" in readme_text
    assert "Codex or Claude Code" in readme_en_text
    assert "#### 安装 OCR" in readme_text
    assert "##### 方案一：PaddleOCR（推荐）" in readme_text
    assert "##### 方案二：Tesseract" in readme_text
    assert readme_text.index("#### 检查环境 ✅") < readme_text.index(
        "#### 配置本地模型服务"
    )
    assert "#### Install OCR" in readme_en_text
    assert "##### Option 1: PaddleOCR (recommended)" in readme_en_text
    assert "##### Option 2: Tesseract" in readme_en_text
    assert readme_en_text.index("#### Check the environment ✅") < readme_en_text.index(
        "#### Configure the local model service"
    )


def test_readmes_describe_quality_completion_and_current_layout() -> None:
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")

    assert "只有通过质量门禁的重建结果才标记为可编辑转换完成" in readme_text
    assert "only reconstructed results that pass the quality gates are marked complete" in readme_en_text
    assert "无法通过质量检查时保留原内容并明确标记 warning" not in readme_text
    assert "retain their source content with a warning" not in readme_en_text
    assert "├── .github/" in readme_text
    assert "├── .github/" in readme_en_text
    assert "旧版图片专用技术路线，非当前推荐入口" in readme_text
    assert "Legacy image-only pipeline; not the recommended entry point" in readme_en_text


def test_requirements_keeps_pillow_floor() -> None:
    assert "Pillow>=9.0.0" in REQUIREMENTS.read_text(encoding="utf-8").splitlines()


def test_pyproject_keeps_supported_python_range() -> None:
    assert 'requires-python = ">=3.10,<3.13"' in PYPROJECT.read_text(
        encoding="utf-8"
    ).splitlines()


def test_claude_plugin_does_not_pin_a_version() -> None:
    manifest = json.loads(CLAUDE_PLUGIN.read_text(encoding="utf-8"))

    assert "version" not in manifest
    assert manifest["skills"] == [
        "./skills/image-to-ppt",
        "./skills/image-to-psd",
    ]


def test_github_ci_covers_supported_desktop_platforms() -> None:
    workflow = GITHUB_CI.read_text(encoding="utf-8")

    assert "ubuntu-latest" in workflow
    assert "windows-latest" in workflow
    assert "macos-latest" in workflow
