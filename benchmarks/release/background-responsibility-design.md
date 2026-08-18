# 背景结构责任掩码设计

## 目标

图表中的网格线、坐标轴等细小背景结构可以保留，但不能因为计划中出现
`rebuild_background` 就把整页栅格内容视为可编辑转换成功。

## 已拒绝的实现

不再使用 `background_rebuild_approved: bool`。这个布尔值无法表达动作对象、
实际写入区域或产物身份，会把局部授权扩大为整页授权。

## 数据流

1. `_rebuild_canvas_background` 只负责生成背景。二值
   `background-responsibility.png` 延后到共用质量资产组装边界生成；此时当前 graph、
   text、语义掩码和 presentation ownership 都已固定。
2. 候选像素必须同时满足：
   - 位于固定的 `foreground_evidence` 内；
   - 不属于文本掩码；
   - 不属于当前非文本 `pending|pending_gate|frozen` 组件；
   - 本轮背景输出与 source 像素一致；
   - 满足下面“允许的细结构”定义之一。
3. 掩码总面积不得超过页面面积的 5%。超过预算时不发布责任掩码，质量门禁
   继续按未解释视觉内容失败，不做降级。
4. 掩码以 `{path, sha256}` 加入 `quality_input_refs`。新的背景重建以当前轮重新
   生成的掩码替换旧引用；没有背景重建时，按下面“跨轮迁移”规则安全收缩上一轮
   已绑定的掩码。
5. 质量端重新解码并验证二值、尺寸、面积、前景证据、文本排除和细结构约束，
   通过后才将其中像素计入 generated-underlay responsibility。

## 兼容性与失败方式

- 既有 execution/state 不含该引用时语义不变。
- 非法、损坏、超预算或不满足结构约束的掩码 fail closed。
- page-surface 的几何识别只影响背景重建，不再产生任何质量授权；即使识别错误，
  大面积内容仍会因未获得责任像素而被质量门禁拒绝。
- 不修改 Host plan schema，不上传模型或缓存。

## 允许的细结构

现有的 3×3 腐蚀边缘规则继续保留。候选像素只有在以下两类掩码的并集中才
能获得背景责任：

1. **细边缘**：`candidate & ~erode(candidate, 3×3)`。
2. **横向或纵向长直线段**：用于覆盖真实图表中宽度为 3 px 左右、腐蚀后仍有
   1 px 核心的网格线和坐标轴。

长直线段使用固定、与模型无关的几何判定：

- `short_side = min(page_width, page_height)`；
- `max_thickness = max(3, floor(short_side / 300 + 0.5))`；
- `min_length = max(32, floor(short_side * 0.10 + 0.5))`；
- 横向候选先用 `1 × min_length` 矩形核做 opening，再对结果做一次连通域统计；
  仅保留宽度不小于 `min_length`、高度不大于 `max_thickness`、宽高比不小于
  20 的连通域；
- 纵向候选使用转置后的同一规则；
- 最终长直线掩码还要与原始 `candidate` 相交，不能从形态学结果扩大像素集合。

实现只允许两次 opening、两次 `connectedComponentsWithStats` 和按 label 查表，
复杂度为 `O(page_pixels + component_count)`。禁止为每个连通域重新分配整页掩码，
避免噪声输入退化为 `O(component_count × page_pixels)`。

这项扩展有意只识别水平和垂直长线：斜线、曲线、短矩形、厚色块、照片和普通
图表数据线都不能通过长直线规则。恰好满足上述几何条件的细长水平或垂直内容仍
可能被视为背景结构；该边界由文本和组件排除、source/background 像素完全一致
以及页面 5% 总预算共同限制。

## 生成与质量复验

- 生成端和质量端可共用一个只负责上述几何运算的纯函数，避免规则漂移；两端
  必须分别从自己已验证的 source、background、foreground、text 和当前非文本
  `pending|pending_gate|frozen`
  掩码重新构造 `candidate`，质量端不能直接信任生成端的分类结果。
- 质量端继续独立验证责任掩码的路径、SHA-256、二值格式、尺寸、像素集合和 5%
  总预算。任一不一致都按未解释内容处理，不能 warning 后通过。
- 新规则不改变 OCR、SAM、LaMa、Host plan、模型安装或 release benchmark 的其他
  阈值，也不增加新的运行时依赖。

## 跨轮迁移

上一轮合法的背景责任像素可能在下一轮被新的非文本
`pending|pending_gate|frozen` 组件接管。旧引用不能
原样继承，也不能由质量端在检查时静默删掉冲突像素。execution 必须在发布当前轮
`quality_input_refs` 前显式派生当前责任：

```text
current_allowed = geometry(
    refined_foreground
    & ~current_text
    & ~current_visual_exclusion
    & (source == current_background)
)
next_responsibility = previous_responsibility & current_allowed
```

迁移规则固定为：

- `refined_foreground` 必须来自上一轮 `quality_input_refs` 中同一固定
  `foreground_evidence` 的已绑定 bytes；后续重建和迁移都不能重新信任
  `prepared_page.json` 里的裸 Path。它针对当前 background 重新 refinement，但不能
  更换或扩大原 foreground evidence。
- `current_text` 必须由当前 graph 经过 `_effective_text_context` 重新生成，不能沿用
  上一轮 text mask。
- `current_visual_exclusion` 是两个严格绑定集合的并集：当前 graph 中所有非文本
  `pending|pending_gate|frozen` 节点的语义掩码并集，以及当前 presentation assets
  中同一组节点的 ownership 掩码并集。presentation ownership 可能因文字 halo 或
  silhouette 分配越出原语义掩码，不能假定前者是后者的子集；两者都排除最保守，
  仍只需额外一次 `O(page_pixels)` 合并。
- 上一轮 artifact 必须先按原 `{path, sha256}` 从绑定 bytes 安全读取，并使用与质量
  端相同的共享严格 decoder。编码层先验证 PNG signature 和 IHDR，要求
  `bit_depth=8`、`color_type=0`、`compression=0`、`filter=0`、`interlace=0`；像素层
  再验证二维 `uint8`、仅 `{0,255}` 和尺寸。禁止用 PIL `convert("L")` 或只看
  OpenCV 解码结果，把 RGB、palette、1-bit、16-bit 或 `{0,1}` 输入洗白。非法、损坏
  或哈希不符立即失败；当前质量端也必须切换到这一共享 decoder。
- 安全读取使用词法 run-relative 路径，不在链接检查前调用 `resolve()`；要求 regular、
  non-link/reparse、单链接，并在单次 fd 读取期间验证身份稳定。这里的“路径换代”
  只承诺上述单次读取边界，不引入跨重启 inode 身份或新的 artifact schema。
- 原 artifact 永不修改。`next_responsibility` 与旧掩码相同时复用旧引用；只要减少
  了像素，就在当前 execution 目录写新 artifact 并记录新 SHA-256；结果为空时不再
  发布 `background_responsibility` 引用。
- 新 artifact 先在内存编码 PNG，再在已验证并持续持有的 parent directory
  inode/fd 内创建随机 O_EXCL/no-follow staging；通过同一 descriptor 完成写入、
  fsync、readback、单链接、内容和 SHA 验证，最后相对该 parent capability 原子
  no-replace 发布到固定名称。任何发布前失败只留下未引用 staging，不占正式路径，
  因而可重试；发布后失败不执行 unlink、reverse rename 或 replacement cleanup。
  不能用 `Image.save` 覆盖预置路径或硬链接。
- 用户已明确选择 capability-bound 安全语义：授权线性化点是 parent directory 在绑定
  时已经通过词法 Run 边界和目录身份验证，并在操作期间持续持有 inode/fd（Windows
  handle 不共享 DELETE）。绑定后同权限进程移动其祖先或该目录，不撤销对原 inode 的
  授权；后续 mutation 只能通过该 capability 作用于原目录，绝不能重新解析或跟随换代
  pathname、symlink、junction 或同名目录。此语义不承诺 POSIX 命名空间中目录在每一
  瞬间仍具有原词法路径；纯用户态没有将祖先 containment 与子项 mutation 原子绑定的
  通用原语。
- 迁移只能取交集，不能增加旧掩码中没有的责任像素。需要新增责任像素时，必须由
  当前轮实际 `rebuild_background` 重新生成完整掩码。
- 普通 component execution 与 parent fallback 使用同一迁移函数和相同输入边界，
  不能让 fallback 原样带入已与当前 graph 冲突的旧引用。
- 质量端仍从当前轮已绑定的 source、background、foreground、text 和 graph 独立
  重算 allowed 集合，并验证新 artifact 是其子集；质量端不负责修剪或改写输入。
- 迁移只增加一次线性掩码计算和必要时一次 PNG 写入，不调用 OCR、SAM、LaMa 或
  Agent，不改变模型轮数与质量阈值。

新的背景重建不执行旧掩码交集：它在与迁移相同的共用质量资产组装边界，使用
最终发布的同一组绑定 foreground、当前 background、text、graph semantic masks 和
presentation ownership 从零计算 `current_allowed`，新结果直接替换旧引用。超过
5% 或没有合法像素时仍不发布引用。禁止在 presentation assets 建立前由
`_rebuild_canvas_background` 提前生成责任，否则文字 halo/silhouette ownership 可能
越出当时可见的语义掩码。

分支优先级不可交换：

1. 有背景重建：不迁移旧责任；在共用质量资产边界使用完整
   `current_visual_exclusion` 和绑定 foreground evidence 生成当前掩码，结果为空或
   超预算则不发布。
2. 无背景重建且没有旧责任引用：不发布。
3. 无背景重建且有旧引用：严格读取并迁移；`next` 为空时优先不发布，否则相同则
   复用旧引用，不同才通过上述 staging-before-publication 边界写入新 artifact。

责任生成与迁移都放在普通 component execution 与 parent fallback 共用的当前质量
资产组装边界：此时当前 graph、有效 text、绑定 foreground、当前 background、语义
掩码并集和 presentation ownership 并集均已确定。两条路径以及 rebuild/migrate
分支不得各自实现一份近似逻辑。

## 验证

- RED：任意 `rebuild_background` 不能让组件外大面积 source-backed raster 通过。
- RED：伪造、非二值、越界、覆盖文本或超过面积预算的责任掩码必须失败。
- RED：长直线中的文字或非文本 `pending|pending_gate|frozen` 重叠像素不能获得责任。
- RED：短线、厚块、斜线、曲线和宽栅格内容不能通过长直线规则；宽内容最多保留
  原有的 1 px 细边缘，其内部仍必须被质量门禁识别为未解释内容。
- RED：包含大量噪声连通域的输入不能触发逐连通域整页扫描。
- RED：上一轮责任像素被本轮非文本 `pending|pending_gate|frozen` 组件接管后，
  不能继续以旧 SHA 原样
  发布，也不能在质量端静默忽略冲突。
- RED：跨轮迁移不能增加像素、修改旧 artifact，或让 component execution 与
  parent fallback 产生不同结果；非法旧 artifact 必须 fail closed。
- RED：换代 foreground、symlink/reparse、hardlink、RGB/palette/1-bit/16-bit 或
  `{0,1}` 责任图，以及预置的新 artifact 路径都必须在发布前失败。
- RED：迁移必须使用当前 graph 生成的 text，并同时排除非文本
  `pending|pending_gate|frozen` 的语义掩码并集与 presentation ownership 并集；质量
  端必须从各自已绑定的 graph masks 和 presentation assets 独立构造相同集合，不能
  事后修剪输入。
- GREEN：3 px 水平/垂直长网格线以及被图表元素分隔、仍达到长度阈值的线段获得
  完整像素责任；网格交点由 opening 后的线段掩码覆盖。
- GREEN：无冲突时复用旧引用；部分冲突时发布“旧掩码 ∩ 当前允许集合”的新
  artifact 与 SHA；全部冲突时移除引用，剩余未解释像素继续受严格质量门禁约束。
- GREEN：真实组合图中的细网格线获得有限责任，三轮固定计划可重新完成。
- 回归：必须重新生成组合图和流程图最新轮次的 request/graph hash，重新 author
  固定计划，并分别在全新 run root 严格完成。组件质量、runtime execution、release
  benchmark、候选 wheel 安装、PowerPoint 渲染和 `slides_test.py` 全部通过后才
  提交案例计划。
