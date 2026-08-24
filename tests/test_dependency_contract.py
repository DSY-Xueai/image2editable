import json
import re
import types
from pathlib import Path

import pytest

from scripts import visual_segment

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


SAM_PIN = "SAM-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
IOPATH_PIN = "iopath @ git+https://github.com/facebookresearch/iopath.git@b3ea6da153ab61b3b8687544c0708a4234a8fb58"
ANTLR_SDIST_SHA256 = (
    "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"
)
ANTLR_SDIST_PIN = (
    "antlr4-python3-runtime @ "
    "https://files.pythonhosted.org/packages/3e/38/"
    "7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/"
    "antlr4-python3-runtime-4.9.3.tar.gz#sha256="
    f"{ANTLR_SDIST_SHA256}"
)
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
ROOT_LAMA_INPAINT = ROOT / "scripts" / "lama_inpaint.py"
STANDALONE_LAMA_INPAINT = (
    ROOT / "skills" / "image-to-ppt" / "scripts" / "lama_inpaint.py"
)
THIRD_PARTY_NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LAMA_LICENSE = ROOT / "third_party" / "licenses" / "LAMA-APACHE-2.0.txt"
PYPROJECT = ROOT / "pyproject.toml"
RUNTIME_CONSTRAINTS = ROOT / "constraints" / "runtime.txt"
CLAUDE_PLUGIN = ROOT / ".claude-plugin" / "plugin.json"
RUNTIME_REQUIREMENTS = [
    "python-pptx>=1.0.2,<2",
    "opencv-python>=4.10.0.84,<5",
    "Pillow>=10.4,<12",
    "numpy>=1.26.4,<2",
    "pypdfium2>=5.7.1,<6",
    "torch>=2.5.1,<3",
    "torchvision>=0.20.1,<1",
    SAM_PIN,
    "transformers>=4.57,<5",
    "accelerate>=1.8,<2",
]
DIRECT_RUNTIME_PINS = [
    "python-pptx==1.0.2",
    "opencv-python==4.10.0.84",
    "Pillow==10.4.0",
    "numpy==1.26.4",
    "pypdfium2==5.7.1",
    "torch==2.5.1",
    "torchvision==0.20.1",
    SAM_PIN,
    "transformers==4.57.1",
    "accelerate==1.8.0",
]
WINDOWS_COMMON_TRANSITIVE_PINS = [
    "certifi==2026.7.22",
    "charset-normalizer==3.5.0",
    'colorama==0.4.6; sys_platform == "win32"',
    "filelock==3.32.3",
    "fsspec==2026.7.0",
    "huggingface_hub==0.36.2",
    "hydra-core==1.3.5",
    "idna==3.18",
    "iniconfig==2.3.0",
    "Jinja2==3.1.6",
    "lxml==6.1.1",
    "MarkupSafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "omegaconf==2.3.1",
    "packaging==26.3",
    "pluggy==1.6.0",
    "portalocker==4.1.0",
    "psutil==7.2.2",
    "Pygments==2.20.0",
    "pypdf==6.16.1",
    "pytest==9.1.1",
    "PyYAML==6.0.2",
    "regex==2026.7.19",
    "reportlab==5.0.0",
    "requests==2.34.2",
    "safetensors==0.8.0",
    "sympy==1.13.1",
    "tokenizers==0.22.2",
    "tqdm==4.70.0",
    "typing_extensions==4.16.0",
    "urllib3==2.7.0",
    "xlsxwriter==3.2.9",
    'setuptools==84.0.0; python_version >= "3.12"',
]


def _dependency_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


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


def test_local_service_provider_docs_are_unambiguous() -> None:
    chinese = README.read_text(encoding="utf-8")
    english = README_EN.read_text(encoding="utf-8")
    skill = SKILL.read_text(encoding="utf-8")

    assert "可选：安装 Local Agent" not in chinese
    assert "Qwen" not in chinese
    assert "`local-service` 使用 OpenAI 兼容的本地服务" in chinese
    assert "--agent-provider local-service" in chinese
    assert "Optional: install the Local Agent" not in english
    assert "Qwen" not in english
    assert "`local-service` uses an OpenAI-compatible local service" in english
    assert "--agent-provider local-service" in english
    assert "`local` 由内置 Qwen" in skill
    assert "`local-service`" in skill
    assert "--agent-provider local-service" in skill


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


def test_runtime_dependency_ranges_match_product_and_standalone() -> None:
    assert _dependency_lines(REQUIREMENTS) == RUNTIME_REQUIREMENTS
    assert _dependency_lines(STANDALONE_REQUIREMENTS) == RUNTIME_REQUIREMENTS


def test_pyproject_reads_runtime_dependencies_and_does_not_relax_agent_extra() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert project["project"]["dynamic"] == ["dependencies"]
    assert project["tool"]["setuptools"]["dynamic"]["dependencies"]["file"] == [
        "requirements.txt"
    ]
    agent_local = project["project"]["optional-dependencies"]["agent-local"]
    assert [
        requirement
        for requirement in agent_local
        if requirement.split(">=", 1)[0]
        in {"torch", "transformers", "accelerate"}
    ] == [
        "torch>=2.5.1,<3",
        "transformers>=4.57,<5",
        "accelerate>=1.8,<2",
    ]


def test_runtime_constraints_are_explicitly_candidate_and_fully_pinned() -> None:
    text = RUNTIME_CONSTRAINTS.read_text(encoding="utf-8")
    lines = _dependency_lines(RUNTIME_CONSTRAINTS)

    assert "candidate release constraints" in text.casefold()
    assert "not release-final until task 7 ci matrix passes" in text.casefold()
    assert all(pin in lines for pin in DIRECT_RUNTIME_PINS)
    assert not any("@main" in line or "simple-lama" in line.casefold() for line in lines)
    for line in lines:
        if line in {SAM_PIN, IOPATH_PIN, ANTLR_SDIST_PIN}:
            continue
        requirement, separator, marker = line.partition(";")
        assert re.fullmatch(
            r"[A-Za-z0-9_.-]+==[^=<>!~;\s]+",
            requirement.strip(),
        )
        if separator:
            assert marker.strip()


def test_iopath_constraint_uses_official_v010_commit_not_a_branch() -> None:
    iopath_lines = [
        line
        for line in _dependency_lines(RUNTIME_CONSTRAINTS)
        if line.startswith("iopath @ git+")
    ]

    assert iopath_lines == [IOPATH_PIN]
    assert re.fullmatch(r"[0-9a-f]{40}", iopath_lines[0].rsplit("@", 1)[1])
    assert "@main" not in iopath_lines[0]


def test_constraints_bind_the_only_non_vcs_sdist_exception() -> None:
    text = RUNTIME_CONSTRAINTS.read_text(encoding="utf-8")

    assert ANTLR_SDIST_PIN in _dependency_lines(RUNTIME_CONSTRAINTS)
    assert text.count("Only non-VCS sdist exception:") == 1
    assert f"PyPI sdist SHA256: {ANTLR_SDIST_SHA256}" in text


def test_constraints_pin_windows_311_312_resolved_transitives() -> None:
    lines = _dependency_lines(RUNTIME_CONSTRAINTS)

    assert all(pin in lines for pin in WINDOWS_COMMON_TRANSITIVE_PINS)


def test_lama_dependencies_use_torch_without_the_old_wrapper() -> None:
    expected = "torch>=2.5.1,<3"

    for requirements in (REQUIREMENTS, STANDALONE_REQUIREMENTS):
        lines = requirements.read_text(encoding="utf-8").splitlines()
        assert lines.count(expected) == 1
        assert not any(
            "simple-lama-inpainting" in line.casefold()
            for line in lines
        )
    assert "simple-lama-inpainting" not in PYPROJECT.read_text(
        encoding="utf-8"
    ).casefold()


def _assert_standalone_skill_uses_the_local_lama_adapter(skill: str) -> None:
    assert "simple-lama-inpainting" not in skill.casefold()
    assert "上限来自 `simple-lama-inpainting" not in skill
    assert "wrapper 首次运行" not in skill
    assert "`torch>=2.5.1,<3`" in skill
    assert "本地 TorchScript adapter" in skill


def test_standalone_skill_uses_the_local_lama_adapter() -> None:
    _assert_standalone_skill_uses_the_local_lama_adapter(
        SKILL.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "old_requirement",
    [
        "simple-lama-inpainting",
        "simple-lama-inpainting>=0.1.2",
        "Simple-LaMa-InPainting >= 0.1.2",
    ],
)
def test_standalone_skill_rejects_old_lama_dependency_variants(
    old_requirement: str,
) -> None:
    skill = SKILL.read_text(encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_standalone_skill_uses_the_local_lama_adapter(
            f"{skill}\n- 安装 `{old_requirement}`。\n"
        )


def test_lama_adapter_product_and_skill_mirrors_match() -> None:
    assert ROOT_LAMA_INPAINT.read_bytes() == STANDALONE_LAMA_INPAINT.read_bytes()


def test_lama_notice_describes_adapter_reference_and_valid_license() -> None:
    notice = THIRD_PARTY_NOTICES.read_text(encoding="utf-8")
    license_text = LAMA_LICENSE.read_text(encoding="utf-8")

    assert "local TorchScript adapter" in notice
    assert "simple-lama-inpainting" in notice
    assert "LaMa" in notice
    assert "Apache License 2.0" in notice
    assert "third_party/licenses/LAMA-APACHE-2.0.txt" in notice
    assert "calls LaMa through the `simple-lama-inpainting` wrapper API" not in notice
    assert LAMA_LICENSE.is_file()
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text


def test_opencv_stays_within_the_verified_major_version() -> None:
    expected = "opencv-python>=4.10.0.84,<5"

    assert expected in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert expected in STANDALONE_REQUIREMENTS.read_text(
        encoding="utf-8"
    ).splitlines()


def test_standalone_declares_accelerate_used_by_visual_segmentation(
    monkeypatch,
) -> None:
    expected = "accelerate>=1.8,<2"
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


def test_readmes_document_offline_models_after_ocr_and_before_doctor() -> None:
    readme_text = README.read_text(encoding="utf-8")
    readme_en_text = README_EN.read_text(encoding="utf-8")
    paddle_url = "https://www.paddlepaddle.org.cn/install/quick"
    tesseract_url = "https://tesseract-ocr.github.io/tessdoc/Installation.html"
    commands = (
        'python -m pip install "paddleocr==3.7.0" "paddlepaddle==3.3.1" "PaddleX==3.7.2" "PyYAML==6.0.2"',
        "tesseract --version",
        "python -m pip install pytesseract",
        "image2editable models install runtime",
        "\nimage2editable doctor\n",
    )

    for text in (readme_text, readme_en_text):
        assert text.index("pip install .") < text.index(commands[0])
        for before, after in zip(commands, commands[1:]):
            assert text.index(before) < text.index(after)
        assert paddle_url in text
        assert tesseract_url in text
        assert '"ready": true' in text
    assert "需要确认" in readme_text
    assert "SAM 2.1 Large、Big-LaMa 和 Grounding DINO" in readme_text
    assert "校验下载结果、记录模型完整性" in readme_text
    assert "模型文件、完整性记录" in readme_text
    assert "asks for confirmation" in readme_en_text
    assert "SAM 2.1 Large, Big-LaMa, and Grounding DINO" in readme_en_text
    assert "receipt" in readme_en_text
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
        "#### Configure a local model service"
    )


def test_model_setup_docs_never_claim_first_conversion_downloads_models() -> None:
    documents = (
        README.read_text(encoding="utf-8"),
        README_EN.read_text(encoding="utf-8"),
        SKILL.read_text(encoding="utf-8"),
    )
    forbidden = (
        "首次转换自动下载",
        "首次运行自动下载",
        "first conversion automatically downloads",
        "first run automatically downloads",
    )

    for document in documents:
        assert all(text.casefold() not in document.casefold() for text in forbidden)


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


def test_requirements_keep_verified_pillow_range() -> None:
    assert "Pillow>=10.4,<12" in REQUIREMENTS.read_text(
        encoding="utf-8"
    ).splitlines()


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
