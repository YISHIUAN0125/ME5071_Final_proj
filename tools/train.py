from ultralytics import YOLO

model = YOLO("./models/yolo11n-seg.pt")
model.to('cuda')

results = model.train(data="data/data.yaml", batch=8, epochs=100, imgsz=1000, patience=20)