# 通用 PyTorch 视觉模型测速版本说明
2026.09.10：test 内部功能模块化。新增 `modules` 配置和
`--only/--enable/--disable/--list-modules`，可独立选择模型调用、pipeline、参数量、峰值显存、ONNX 导出、TensorRT 构建、engine 调用和一致性检查；关闭模块不会执行对应计算。旧的 `test_model_call`、`test_pipeline`、`test_pytorch` 选择字段已移除，避免和 YAML 模块开关重复。

v1.1，为一致性检查添加py文件构建api

v1.0：完成初版
