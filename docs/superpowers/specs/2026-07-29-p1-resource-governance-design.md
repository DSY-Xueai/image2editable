# P1.1 资源治理与真实文件兼容设计

## 状态

本设计于 2026-07-29 经用户确认，用于完成 Unified Runtime P1 的真实文件验收收尾。P1.1 已于 2026-07-31 完成实现与真实文件验收；实测结果见文末。

P1.1 只解决三个问题：

1. 在不降低现有模型、阈值和 PDF 清晰度基线的前提下降低内存、显存和磁盘压力；
2. 兼容 `test1.pptx` 中合法的 ZIP 目录项，并识别整页背景图片候选；
3. 让被强制中断的执行任务可以安全恢复，不长期停留在 `running`。

P1.1 完成并通过真实文件验收后，才进入 P2 Agent 页面路由、截图终判和 OOXML 原位替换。

## 已确认事实

### 资源峰值

使用 `1-Embedding与向量数据库.pdf` 执行完整转换时：

- PDF 共 4 页，页面物理尺寸一致，标准渲染均为 2667×1500、200 DPI；
- Python 进程工作集峰值约 8.6 GB；
- RTX 4060 Laptop 8 GB 显存占用约 7.8 GB；
- 本机可见物理内存约 15.2 GB，转换前系统已使用约 8.6 GB；
- 内存总需求超过物理内存后触发 Windows 页面文件，磁盘活动率达到 100%；
- 当前代码在重型模型加载后才执行 OCR，Grounding DINO、SAM2 和后续模型可能同时驻留；
- `_try_paddleocr()` 每次调用都会新建 `PaddleOCR`，一页可能执行原图 OCR、重建检查和修复后复检三次；
- CUDA 下 SAM2 当前 `points_per_batch=16`；
- 中间文件不是本次磁盘满负载的主因：已完成页面的临时文件约 15.4 MB，PDF Run 约 1.9 MB。

### PPTX 兼容

`test1.pptx` 是有效的两页 16:9 图片版演示文稿：

- ZIP 包含 12 个规范、未加密、零字节目录项和 42 个普通文件 part；
- 当前 `_validate_archive()` 无条件拒绝目录项，首次遇到 `_rels/` 即失败；
- 仅移除目录项的诊断副本可被当前扫描器读取，42 个普通 part 内容 SHA-256 全部不变；
- 两页内容均由 `p:bg/p:bgPr/a:blipFill` 引用整页 PNG；
- 两页 `p:spTree` 中均没有普通 shape；
- 当前扫描结果因此为 2 页、0 对象、0 候选；
- 两张背景图均为 1672×941，与幻灯片宽高比一致。

### 中断状态

执行进程被用户结束后：

- OS 已释放 Python 进程和 GPU 资源；
- `run_state.json` 仍为 `running`；
- 页面仍可能停留在 `processing`；
- 当前 `retry` 只接受已记录失败或特定孤儿失败批次，不能恢复这种硬中断。

## 目标

### 资源目标

1. 保持 PDF 标准渲染策略不变：200 DPI、小页短边下限、6000 px 长边上限和 24 MP 上限均不修改。
2. 保持 Grounding DINO、SAM2 Large、LaMa、PaddleOCR server recognizer 和候选阈值；允许使用 mobile detector，并增强最终质量门禁。
3. CUDA 下 SAM2 保持相同采样点，只将 `points_per_batch` 从 16 降为 1。
4. OCR 检测与识别拆为顺序子进程，避免两个 PaddleOCR 模型同时驻留。
5. OCR 阶段和视觉模型阶段不同时持有不需要的重型模型。
6. 重型页面始终串行执行，不增加页面级并行。
7. 默认限制数值计算线程，最多使用 8 个逻辑线程，并尊重用户已设置的线程环境变量。
8. 在本机真实验收中，目标峰值为：
   - Python 工作集不高于 5.5 GB；
   - GPU 显存不高于 6.0 GB；
   - 不再因该进程触发持续页面文件换入换出。

资源目标是 P1.1 的验收门槛。若同模型、同阈值的实现仍无法达到门槛，应继续调整调度方式，不能通过默认降低 DPI、换小模型或减少 SAM 采样点来通过。

### 质量目标

1. 资源治理不能降低 PDF 清晰度、SAM2 Large、采样点或候选阈值；OCR 检测可使用已验收的 mobile detector，识别仍使用 server recognizer。
2. 每页仍执行现有全分辨率 `visual_difference`、残留文字复检和 `require_visual_quality`。
3. 低批次只改变推理分批方式，不改变采样点和候选阈值；最终重建出现可见伪影时允许保守回退为去文字底图。
4. 真实 PDF 必须完成 4 页输出，输出可打开、页序正确、页面比例正确。
5. 真实 PDF 的最终 PPTX 必须逐页渲染并人工检查文字、背景、图形和边界，无明显新增缺失、错位或拉伸。
6. 输入 PDF 和 PPTX 的 SHA-256 在验收前后保持不变。
7. 任何资源路径失败都必须显式失败，不能输出未通过质量门禁的结果。

## 非目标

P1.1 不实现：

- 根据页面语义自动选择轻量或重型视觉路径；
- Agent 对 PPTX 截图候选的最终 `replace` / `preserve` 决策；
- SAM2 Small/Tiny、低精度模型或新的 OCR 引擎切换；
- 默认降低 PDF DPI；
- 300 DPI detail 自动触发；
- OOXML 图片替换；
- 最终 PPTX 的自动 shadow render 评分与回滚；
- 任意用户可调的资源参数面板。

以上能力属于 P2 或后续独立优化。

## 方案选择

### 方案 A：只限制优先级和线程

只降低进程优先级、CPU 线程数和 SAM batch。

优点是改动最小，质量语义不变。缺点是 PaddleOCR 仍重复初始化，重型模型仍可能同时驻留，不能稳定解决 16 GB 内存机器的换页问题。

### 方案 B：默认降低 DPI 或更换小模型

直接减少输入像素或使用更小的视觉模型。

资源下降明显，但会改变 OCR 小字、细线和复杂元素的质量基线，不符合本轮“不以质量换资源”的要求。

### 方案 C：阶段化资源治理与质量门禁

保留现有模型和清晰度，将 OCR、视觉分解和组装拆成有边界的阶段；复用同阶段模型、降低批次峰值、限制并发，并保留现有全质量检查。

这是选定方案。它优先消除重复初始化和同时驻留，不改变算法能力；代价是执行时间可能增加。

## 架构

### 资源策略

新增一个很小的资源策略模块，负责：

- 在重型库导入前设置默认线程上限；
- 只在用户没有显式设置对应环境变量时写入默认值；
- 给 SAM2 构造器提供确定的 CUDA batch 值 1；
- 尽力将转换进程设置为不会抢占桌面交互的较低优先级；
- 记录本次执行采用的资源策略，供日志和 Run Summary 审计。

不增加通用配置系统。P1.1 只有一个默认安全策略。

进程优先级调整只影响调度，不改变计算内容。Windows 使用 Below Normal，
POSIX 使用正的 nice 增量；权限或平台不支持时记录 warning 并继续，不能因此中断转换。

### OCR 生命周期

每次 OCR 调用按“检测子进程 → 识别子进程”顺序执行。检测使用 mobile detector，识别继续使用 server recognizer；两个模型不同时驻留，子进程退出后由操作系统回收内存。

原图 OCR、重建残留检查和修复后复检仍沿用原有调用时机。OCR 检测结果、文字 mask、裁剪图和修复输入写入页面工作目录，不在父进程常驻。

### 视觉模型生命周期

每页视觉流程运行在独立 `visual_worker` 中；页面完成后整个进程退出。

约束：

- 页面仍然串行；
- Grounding DINO 和 SAM2 的模型、提示词、阈值不变；
- CUDA 与 CPU 均使用 `points_per_batch=1`；
- DINO proposal 使用独立 `object_worker`；
- SAM prompted、automatic 与最终 hole recheck 分别使用独立 `sam_worker`，mask 通过 packbits/Base64 传递；
- SAM2.1 Large 使用同一 FP32 checkpoint，并通过 mmap/空权重初始化减少加载峰值；CUDA 推理使用 BF16 autocast；
- LaMa 使用独立 `lama_worker`，不在视觉进程中长期驻留；
- 最终 hole recheck 不在父 `visual_worker` 创建 CUDA context，避免随后 LaMa 推理叠加显存占用。

进程退出是确定的资源释放边界，不使用小模型、低 DPI 或减少采样点作为默认补丁。

### 线程与并发

默认线程预算为：

```text
min(8, max(1, logical_cpu_count // 2))
```

只通过 `setdefault` 设置受支持的 OpenMP、MKL、OpenBLAS、NumExpr 和 Paddle 线程变量。用户显式设置的值优先。

P1.1 不启动多个重型页面 worker。轻量 JSON、哈希和 OOXML 扫描可以继续使用现有同步实现。

### Run 所有的临时目录

Unified Runtime 调用 Legacy adapter 时，把页面工作目录放到 Run 根目录下的
`work/`，不再为该路径使用无法追踪的系统临时目录。

- 正常完成并写入最终结果后删除 `work/`；
- 普通失败时保留本次失败页的诊断目录，并在失败摘要中记录位置；
- `recover` 在重置孤儿任务前删除 Run 所有的未完成 `work/`；
- Legacy 独立 CLI 保持现有行为，P1.1 不顺手改变其诊断目录契约。

组件 PNG、背景和质量诊断继续写磁盘，不改为常驻内存；这能避免用磁盘占用问题换成新的内存峰值。

### PPTX 安全目录项

归档扫描仍对所有成员应用数量上限。目录项只有同时满足以下条件才允许：

- ZIP API 将其识别为目录；
- 名称以 `/` 结尾；
- 去掉末尾 `/` 后是非空、相对、规范化的 POSIX 路径；
- 不含反斜杠、NUL、`.`、`..` 或向上逃逸；
- 未加密；
- `file_size == 0`；
- `compress_size == 0`；
- 不与已有目录或普通文件的规范化名称冲突。

安全目录项只作为容器元数据，不进入 OOXML part 名称集合，不参与 part 内容哈希、XML 解析或总解压字节累计。普通文件继续使用现有全部安全校验。

### PPTX 整页背景图候选

扫描每页 `p:cSld/p:bg/p:bgPr/a:blipFill`。

只有以下情况生成 `slide_background_image` 对象：

- 存在一个内部 `a:blip` 图片关系；
- 关系类型和 content type 是当前允许的图片类型；
- 媒体 part 存在、可安全解码并通过像素资源限制；
- 不存在外部关系或当前不支持的扩展效果。

该对象使用确定性身份：

```text
shape_id = "background"
name = "Slide Background Image"
z_order = -1
type = "slide_background_image"
slide_coverage = 1.0
```

符合安全条件时标记为 `candidate`；否则记录为 `preserve` 并写入具体安全原因。纯色、渐变、主题引用和无图片背景不生成该对象。

P1 执行仍逐字节复制原 PPTX，不修改背景关系、媒体或任何原生对象。

### 执行租约与硬中断恢复

每次 `execute` 持有 Run 目录内的 OS 级独占锁。进程退出或被强制结束时，操作系统自动释放锁。

Run 同时记录只用于审计的执行元数据：

- 随机 execution token；
- PID；
- 开始时间；
- 输入类型。

再次执行或调用恢复入口时：

1. 如果 Run 状态不是 `running` / `finalizing`，沿用现有状态机；
2. 如果独占锁仍被持有，拒绝并发恢复；
3. 如果状态为 `running` / `finalizing` 但锁可获得，判定为孤儿执行；
4. 对图片和 PDF，将 `processing` / `validated` 页面转换为 `failed` 后重置为 `pending`；
5. 删除 Run 目录内由该次执行拥有的未完成最终输出；
6. 删除 Run 目录内由该次执行拥有的未完成 `work/`；
7. 外部输出路径已出现时不自动覆盖，显式报告并阻止恢复；
8. Run 回到 `prepared` 后允许重新执行。

PPTX 继续沿用更严格的已发布输出保护，不能因为恢复功能放宽 byte-identical 和输出身份校验。

## 数据与接口

### Run Manifest

`options` 增加只读审计字段：

```json
{
  "resource_policy": {
    "name": "safe-default",
    "cpu_threads": 8,
    "heavy_page_concurrency": 1,
    "sam_points_per_batch": 1
  }
}
```

该字段由 Runtime 生成，用户不能通过 P1.1 CLI 传入任意值。

### 执行元数据

Run 根目录增加：

```text
execution.lock
execution.json
```

`execution.lock` 只用于 OS 锁；`execution.json` 使用原子 JSON 写入，不能作为锁是否存活的依据。

### CLI

保留现有：

```bash
image2editable run execute RUN_DIR
image2editable run retry RUN_DIR --page PAGE_ID
```

增加：

```bash
image2editable run recover RUN_DIR
```

`recover` 只恢复已失去执行锁的孤儿任务，不终止正在运行的进程。

## 错误处理

必须显式失败：

- 同一个 Run 已被另一个执行进程持锁；
- 目录项非空、加密、路径不规范或与普通成员冲突；
- 背景图片关系外部化、缺失、类型错误或媒体解码超限；
- 孤儿任务的外部输出路径已存在；
- 资源策略无法在重型库导入前生效；
- 质量门禁未通过。

可以保守继续：

- 页面没有背景图片；
- 背景使用纯色、渐变或主题引用；
- 背景图片结构完整但不满足候选安全条件：记录为 `preserve`；
- 用户已显式设置线程环境变量：保留用户值并记录。

## 测试

### 资源策略

- 默认线程预算在 32 逻辑处理器机器上为 8；
- 小于 16 个逻辑处理器时使用一半，至少为 1；
- 已存在的线程环境变量不被覆盖；
- 进程优先级调整失败只记录 warning；
- CUDA 和 CPU 的 SAM batch 都为 1，旧 manifest 的 4 仍可读取；
- 重型页面并发固定为 1；
- OCR 检测与识别由两个顺序子进程执行；
- OCR、LaMa、DINO、SAM 和整页视觉 worker 退出后不在父进程持有模型；
- 残留文字为空时不执行第三次 OCR；
- OCR 结果转换逻辑不变；检测改为 mobile detector，识别保持 server recognizer。

### PPTX

- 规范零字节目录项被接受；
- 目录项不进入 part 哈希集合；
- 非空、加密、绝对路径、反斜杠、`..`、重复和文件冲突目录项被拒绝；
- 普通 ZIP 文件成员的现有压缩炸弹和大小限制保持生效；
- `test1.pptx` 扫描得到 2 页、2 个 `slide_background_image` 对象和 2 个候选；
- 两个候选 coverage 均为 1.0，媒体尺寸均为 1672×941；
- 纯色或主题背景不生成图片对象；
- 普通整页 `p:pic` 候选行为不变；
- 混合原生对象和背景图片时，原生对象 inventory 保持完整；
- P1 输出与输入 `test1.pptx` SHA-256 完全一致。

### 恢复

- 正常执行持锁时第二个执行和 `recover` 都被拒绝；
- 模拟进程退出并释放锁后，`recover` 将孤儿图片/PDF Run 恢复为 `prepared`；
- 页面状态恢复为可重新执行的状态；
- Run 内未完成输出和 `work/` 被清理；
- 正常完成后 `work/` 被清理；
- 普通失败保留的诊断路径写入失败摘要；
- 外部输出存在时恢复被阻止；
- PPTX 已发布或身份不确定的输出继续阻止恢复；
- 正常失败、正常重试和完成态读取行为不回归。

### 回归

- 完整回归为 `772 passed, 13 skipped`；
- `python -m compileall -f -q image2editable scripts image_to_ppt.py` 通过；
- 主脚本与 Skill 镜像保持字节一致；
- CLI help、doctor、prepare、execute、retry 和 recover 冒烟通过。

## 真实文件验收

### `1-Embedding与向量数据库.pdf`

- 4 页标准渲染仍为 200 DPI、2667×1500。
- v36 四页完整转换耗时 1424.909 秒；进程树工作集峰值为 3.407 GB，GPU 峰值为 5.971 GB，系统最低可用内存为 2.096 GB，资源监控未触发上限。
- v35 曾因父 `visual_worker` 的最终 SAM CUDA context 与后续 LaMa 叠加达到 6.104 GB；将最终 hole recheck 移入独立 worker 后，v36 达到 6.0 GB 验收门槛。
- 最终验收稿为 `tmp/p11-acceptance/pdf-v36-final_original.pptx` 与 `pdf-v36-final_16x9.pptx`。
- 两份验收稿均为 4 页、可打开、无越界；所有页面已逐页渲染检查。
- 4 页均因连续伪影门禁触发去文字底图保真降级，分别保留 1、18、1、8 个可编辑文字框。
- 页面视觉无先前的大面积透明组件伪影；少量 OCR 标点误识别仍是 P1 已知限制。
- 输入 PDF SHA-256 保持为 `8ecf070cb6a0e0a0ff675c82360b25c335a0e5c9bae026b69ed923d2bcd00080`。

### `test1.pptx`

- 原文件直接 `prepare`，扫描得到 2 页、2 个背景图片对象和 2 个候选。
- P1 输出 `tmp/p11-acceptance/test1-preserved.pptx` 与输入 byte-identical。
- 输入和输出 SHA-256 均为 `03415ac5973a91e5b0d462a796690f618267ff1c05b4eb00d5f7ab20fa92ae80`。
- 输入和输出两页渲染图逐像素一致，原生对象未经过 CV。

## 文档与提交

实现完成时同步更新：

- `Course.md`：P1.1 状态、资源策略、PPTX 背景候选、恢复入口和真实验收结果；
- `README.md`：新增 `run recover`，说明默认资源策略不降低模型或 DPI；
- CLI help：说明 recover 只处理失去执行锁的孤儿 Run。

设计文档、测试素材和验收中间产物保持本地；产品代码提交到当前
`codex/agent-runtime-foundation` 分支，不推送、不合并到 `main`。
