# 项目状态

图像转可编辑 PPT 工具；运行入口为 `python -m image2editable`，核心 Local Agent 位于 `image2editable/local_agent.py`。

# 本轮变更

新增无内容 JSONL 性能记录与跨平台设备摘要；worker 和 Local Agent 可选接收 telemetry，未传入时保持原有子进程行为。产品与 `skills/image-to-ppt/scripts/` 镜像脚本保持一致。

# 注意事项

性能事件仅记录白名单字段，不记录路径、OCR、提示词或模型响应；CUDA 可用时选择 CUDA，MPS 仅报告可用性，不改变 CPU 默认选择。
