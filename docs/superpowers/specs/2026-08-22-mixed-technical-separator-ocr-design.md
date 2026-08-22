# 混合技术分隔文本 OCR 设计

## 背景

混合页尺寸 benchmark 的页脚被 PaddleOCR 以 `0.989` 置信度识别为：

```text
842 x 595 pt / project-generated / CC0-1.0
```

当前 `_is_spaced_semantic_separator_text()` 只要发现句点就判定整行无效，导致 `_filter_noise()` 丢弃这条可靠 OCR 文本。页脚随后既没有原生文本对象，也没有合法视觉归属，形成 5,518 个未解释像素。

## 目标

- 保留由带空格 `/` 或 `-` 分隔的合法混合文本。
- 允许分段内部出现嵌入字母或数字之间的 `.`、`-`，覆盖版本号、许可证和连字符单词。
- 保持现有噪声防线，不放宽 OCR 置信度或 release benchmark 门禁。
- 不增加 OCR 推理次数，不改变模型参数，避免性能回退。

## 方案

继续由 `_is_spaced_semantic_separator_text()` 负责判断整行，但把校验拆为两层：

1. 仅把两侧有空白的 `/` 或 `-` 当作段间分隔符；至少存在一个分隔符，且每段非空。
2. 每段至少包含两个字母、数字或 CJK 字符。段内只允许空白、字母数字、CJK，以及夹在两个字母或数字之间的 `.`、`-`。冒号、分号、反斜杠、边界标点和连续标点仍判为无效。

示例：

- 保留：`A4 LANDSCAPE / MIXED SIZE`
- 保留：`842 x 595 pt / project-generated / CC0-1.0`
- 保留：`ALPHA - BETA`
- 拒绝：`MCOULE ST:SETMP`
- 拒绝：`ALPHA / BETA. GAMMA`
- 拒绝：`A / B`
- 拒绝：`ALPHA / / BETA`

通用 `_filter_noise()` 调用顺序、技术标签校验、置信度阈值和后续字体下限保持不变。

## 数据流与失败处理

OCR worker 的检测与识别结果不变。原始文本先经过现有置信度和纯符号检查，再进入分隔文本语法判断。合法文本继续执行样式估算、原生文本生成和文字掩码构建；非法文本仍在噪声过滤阶段丢弃。

该判断是固定长度字符串扫描，不产生额外模型调用或图像处理。无法匹配明确语法的内容按现有保守策略拒绝。

## 修改范围

- `scripts/text_detect.py`
- `skills/image-to-ppt/scripts/text_detect.py` 镜像
- `tests/test_ocr_isolation.py`
- 根目录 `Course.md`

不修改 benchmark 阈值、组件计划协议、PaddleOCR 参数或模型文件。

## 验证

1. 先增加页脚合法文本和边界反例测试，确认旧实现失败。
2. 实现最小语法修正，确认新增测试通过。
3. 运行 OCR 相关完整测试、镜像一致性检查、语法编译和 lint。
4. 重建 wheel，在 E 盘 Python 3.12 验证环境重新安装。
5. 从新的 run 目录重新生成混合页尺寸样例，要求页脚成为原生文本、未解释像素归零，并继续满足严格 benchmark 门禁。
