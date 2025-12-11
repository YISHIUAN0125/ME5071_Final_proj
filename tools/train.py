from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./models/yolo11n-seg.pt")
    model.to('cuda')

    results = model.train(data="data/domain_a/data.yaml", batch=4, epochs=100, imgsz=1000, patience=20)