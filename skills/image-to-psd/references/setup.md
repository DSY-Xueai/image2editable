# 转换环境自动准备

首次使用或预检发现缺项时执行本流程，完整仓库与仅安装 Skill 使用相同步骤。准备完成后继续当前转换任务。缺少时直接安装，不把安装步骤转交给使用者；不得为依赖或模型安装向用户询问确认。宿主明确要求审批、网络阻断或安装校验失败时，遵循实际限制并报告原始原因，不把未安装状态当作成功。

## 存储与基础工具

先解析 `<skill-root>` 为当前 SKILL.md 所在绝对目录。Windows 新安装优先 `D:\image2editable`，再按盘符顺序选择其他可写的非 C 本地固定磁盘；仅在不存在非 C 本地固定磁盘时使用用户目录下的 `image2editable`。macOS 从 `/Volumes` 选择其他已挂载的本地数据卷，排除系统/用户目录所在卷、网络盘及安装镜像；Linux 用 `lsblk` 选择已挂载的其他本地磁盘，排除根文件系统所在磁盘、光盘和 loop 设备。都使用数据卷下的 `image2editable`，没有其他已挂载本地磁盘时才使用 `~/.local/share/image2editable`。其他盘存在但无法写入时修复权限或选择另一个数据盘，不静默回退系统盘。已有可用 Python、Git、转换环境和模型继续复用，不移动或重新下载；用户显式指定的模型路径优先。

1. 检查 Python 3.10–3.12 和 Git。Python 不存在时，先用系统工具按上述规则选定 `<root>`：Windows PowerShell 用 `Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3'` 枚举本地固定磁盘，再实际检查目标目录可写。下载 python.org 的 Python 3.12 Windows 安装器，验证 Authenticode 签名后以 `InstallAllUsers=0 TargetDir="<root>\tools\python" Include_launcher=0 PrependPath=0 Include_test=0` 安装；下载和临时文件同样放在 `<root>`。Git 缺失时从 Git for Windows 官方发布安装到 `<root>\tools\git`，使用安装器 `/DIR="<root>\tools\git"` 指定位置。后台安装窗口隐藏，等待退出码并执行 `--version` 验证。macOS/Linux 用 `diskutil`/`lsblk` 先选盘，缺少 Python 时下载与平台及 CPU 架构匹配的官方 uv 发行文件并校验发布的校验和，解压到 `<root>/tools/uv`，设置 `UV_PYTHON_INSTALL_DIR=<root>/tools/python`、`UV_CACHE_DIR=<root>/cache/uv`，执行 `uv python install 3.12`，用 `uv python find 3.12` 获取实际路径。Git 先复用平台自带版本；缺少时自动从官方源码构建并以 `prefix=<root>/tools/git` 安装，缺失的系统构建工具通过平台包管理器准备。系统包管理器有固定安装位置的基础工具遵循平台规则，转换 Python、依赖、模型和缓存仍放在选定数据盘。
2. 以该 Python 执行 `python "<skill-root>/scripts/skill_environment.py"`，读取 JSON 的 `root` 和 `environment`。后续准备及转换命令均继承这组环境变量，或通过 `python "<skill-root>/scripts/skill_environment.py" --run <命令及参数>` 执行。不要修改 `HOME`、`USERPROFILE` 或系统级 PATH。工具将 pip、Hugging Face、Torch、Paddle/OCR 缓存及 TEMP/TMP 指向所选盘；已有 runtime receipt 和显式模型缓存继续复用。
3. 复用已安装且可用的项目 Python 环境；没有时运行 `python -m venv "<root>/venv"`，之后使用其绝对 Python 路径 `<python>`。Windows 为 `venv/Scripts/python.exe`，Linux/macOS 为 `venv/bin/python`。依赖必须安装到该环境，不能因为当前系统 Python 在 C 盘就把新转换依赖装到 C 盘。

## 项目、依赖和模型

完整仓库使用已验证包含 `pyproject.toml`、`image2editable/` 和 `constraints/runtime.txt` 的当前仓库根目录作为 `<source>`，包括本地未提交修改；不要把调用者的任意当前目录当成项目源码。已有项目包必须核对实际代码，不能仅凭版本号或 `doctor` 通过就复用。

仅安装 Skill 时，在所选盘用自带工具获取 GitHub `main` 当前提交。工具在检出前设置浅层部分克隆（`--filter=blob:none`）与非 cone 稀疏文件清单，只获取运行模块、入口、依赖配置、包元数据必需的英文README及许可证；不获取 benchmarks、tests、docs图片、其他Skills、CI、基准生成/发布/开发脚本。即使已安装同版本项目包，也先检查远端当前提交：

```bash
<python> "<skill-root>/scripts/fetch_skill_source.py" "<root>/source/image2editable-runtime"
```

读取输出JSON的 `source` 和 `commit`，将完整SHA记入准备记录，后续安装和验收绑定此源码，不跟随中途变化的分支。工具只更新自己管理的干净克隆；遇到旧完整克隆或本地修改时保留它并另建专用目录，不执行 reset/clean，也不改用完整克隆或ZIP下载。项目主程序只能来自上述 `<source>`，不得运行 `pip install image2editable` 从 PyPI 取得可能滞后的代码。先执行 `<python> -I "<skill-root>/scripts/verify_skill_runtime.py" "<source>"`，逐文件检查已安装代码及实际导入位置；不一致或未安装时安装当前源码。第三方依赖仍可从 PyPI 安装。所有下列命令均继承上一步的缓存和临时目录设置：

```bash
<python> -m pip install --constraint "<source>/constraints/runtime.txt" torch torchvision setuptools==84.0.0
<python> -m pip install --constraint "<source>/constraints/runtime.txt" --no-build-isolation --no-binary antlr4-python3-runtime "<source>"
<python> -m pip install --no-deps --no-build-isolation --force-reinstall "<source>"
<python> -I "<skill-root>/scripts/verify_skill_runtime.py" "<source>"
<python> -m pip install --constraint "<source>/constraints/runtime.txt" "paddleocr==3.7.0" "paddlepaddle==3.3.1" "PaddleX==3.7.2" "PyYAML==6.0.2"
<python> -m image2editable models install runtime --yes
<python> -m image2editable doctor
```

只安装预检缺少的部分。源码核验一致时跳过项目重装；需要覆盖同版本旧代码时仅对项目使用 `--no-deps --force-reinstall`，保留已满足的第三方依赖、有效模型和缓存。代码核验、`doctor` 均通过才继续转换，不能退回旧 PyPI 包或仅记录 warning。已有可用 CUDA/ROCm PyTorch 时保留对应构建，不用 CPU wheel 覆盖；新 CPU 环境安装 torch/torchvision 时使用官方 CPU index。SAM 采用约束中的固定 Git commit。LaMa 使用本地 TorchScript adapter，依赖 `torch>=2.5.1,<3`。模型命令下载并校验 SAM 2.1 large、Big-LaMa 和 Grounding DINO，不要求使用者手工设置 `SAM2_MODEL`、`LAMA_MODEL`、`GROUNDING_DINO_MODEL`。已有显式路径需校验身份；未配置的模型由 runtime receipt 解析。

`doctor` 检查依赖导入但不代表 OCR 权重已下载。准备阶段以相同环境执行 `<python> -c "from scripts.text_detect import _get_paddleocr; _get_paddleocr('ch')"`，预热默认 OCR 并确认模型加载成功；其他语言按实际任务预热。不要从 Skill 的 `scripts/` 目录执行该命令，以免遮蔽已安装模块。推理不会下载 SAM/LaMa/DINO 模型或回退 Hugging Face cache。

PSD 任务额外安装 `aspose-psd>=26.5.0` 并预检已有 `ASPOSE_PSD_LICENSE`。商业授权不能通过下载安装取得，不伪造许可证或使用带水印试用输出。

## PPTX 实际渲染

原生 PDF 在交付前需要实际渲染。先复用 PowerPoint 或 LibreOffice。Windows 有 PowerPoint 时，在所选 Python 环境安装 `pywin32>=306`；没有可用渲染器时自动安装 LibreOffice：所选源码自带 `scripts/install_release_renderer.ps1`，其中下载 URL 和 SHA-256 已固定。

```powershell
$env:RUNNER_TEMP = "<root>/tools"
$env:GITHUB_ENV = "<root>/renderer.env"
& "<source>/scripts/install_release_renderer.ps1"
```

该命令须继承所选 TEMP/TMP。成功后验证 `<root>/tools/native-renderer/extracted/program/soffice.com --version`；路径工具会自动向后续子进程设置 `IMAGE2EDITABLE_LIBREOFFICE`。不得把 LibreOffice 的 program 目录前置到 PATH，其中的 Python 会干扰转换环境。macOS 下载匹配 Intel/Apple Silicon 的官方 LibreOffice DMG 并校验发布的 SHA-256，将应用复制到 `<root>/tools/LibreOffice.app`；Linux 下载对应平台官方 LibreOffice 归档并校验，以 `dpkg-deb -x` 或 `rpm2cpio` 解包到 `<root>/tools/libreoffice`，补齐实际缺失的系统共享库。均以实际 `soffice --version` 和试渲染验证，设置 `IMAGE2EDITABLE_LIBREOFFICE` 为该可执行文件的绝对路径并传给后续转换，不能仅以文件存在作为安装成功。

在同一环境继续原 Run，复用有效 OCR、分割及冻结组件。依赖准备通过不等于转换验收通过，最终仍执行原有完整质量门禁。
