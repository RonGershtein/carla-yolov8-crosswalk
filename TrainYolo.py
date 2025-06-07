from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # מודל בסיס
model.train(data="C:/Users/Ron/Desktop/YOLOTRAINER/data.yaml", epochs=50, imgsz=640)
