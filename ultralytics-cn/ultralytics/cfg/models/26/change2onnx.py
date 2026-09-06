from ultralytics import YOLO
model = YOLO("yolo26n.yaml")
model.export(format="onnx")
