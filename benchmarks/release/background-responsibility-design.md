# 背景结构责任掩码设计

## 目标

图表中的网格线、坐标轴等细小背景结构可以保留，但不能因为计划中出现
`rebuild_background` 就把整页栅格内容视为可编辑转换成功。

## 已拒绝的实现

不再使用 `background_rebuild_approved: bool`。这个布尔值无法表达动作对象、
实际写入区域或产物身份，会把局部授权扩大为整页授权。

## 数据流

1. `_rebuild_canvas_background` 仍负责生成背景，同时生成二值
   `background-responsibility.png`。
2. 候选像素必须同时满足：
   - 位于固定的 `foreground_evidence` 内；
   - 不属于文本掩码；
   - 不属于当前 active/frozen 组件；
   - 本轮背景输出与 source 像素一致；
   - 满足下面“允许的细结构”定义之一。
3. 掩码总面积不得超过页面面积的 5%。超过预算时不发布责任掩码，质量门禁
   继续按未解释视觉内容失败，不做降级。
4. 掩码以 `{path, sha256}` 加入 `quality_input_refs`。没有背景重建时不新增该
   引用；后续轮次只能继承上一轮已绑定的掩码，新的背景重建则与新产生的安全
   掩码合并。
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
  必须分别从自己已验证的 source、background、foreground、text 和 active/frozen
  掩码重新构造 `candidate`，质量端不能直接信任生成端的分类结果。
- 质量端继续独立验证责任掩码的路径、SHA-256、二值格式、尺寸、像素集合和 5%
  总预算。任一不一致都按未解释内容处理，不能 warning 后通过。
- 新规则不改变 OCR、SAM、LaMa、Host plan、模型安装或 release benchmark 的其他
  阈值，也不增加新的运行时依赖。

## 验证

- RED：任意 `rebuild_background` 不能让组件外大面积 source-backed raster 通过。
- RED：伪造、非二值、越界、覆盖文本或超过面积预算的责任掩码必须失败。
- RED：长直线中的文字/active/frozen 重叠像素不能获得责任。
- RED：短线、厚块、斜线、曲线和宽栅格内容不能通过长直线规则；宽内容最多保留
  原有的 1 px 细边缘，其内部仍必须被质量门禁识别为未解释内容。
- RED：包含大量噪声连通域的输入不能触发逐连通域整页扫描。
- GREEN：3 px 水平/垂直长网格线以及被图表元素分隔、仍达到长度阈值的线段获得
  完整像素责任；网格交点由 opening 后的线段掩码覆盖。
- GREEN：真实组合图中的细网格线获得有限责任，三轮固定计划可重新完成。
- 回归：必须重新生成组合图最新轮次的 request/graph hash，重新 author 固定计划，
  并在全新 run root 严格完成。组件质量、runtime execution、release benchmark、
  候选 wheel 安装、PowerPoint 渲染和 `slides_test.py` 全部通过后才提交案例计划。
