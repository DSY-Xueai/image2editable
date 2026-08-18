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
   - 是 3×3 形态学腐蚀后消失的细结构。
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

## 验证

- RED：任意 `rebuild_background` 不能让组件外大面积 source-backed raster 通过。
- RED：伪造、非二值、越界、覆盖文本或超过面积预算的责任掩码必须失败。
- GREEN：真实组合图中的细网格线获得有限责任，三轮固定计划可重新完成。
- 回归：组件质量、runtime execution、release benchmark、候选 wheel 安装、
  PowerPoint 渲染和 `slides_test.py` 全部通过后才提交案例计划。
