from ultralytics import YOLO

# Load pose model
model = YOLO("yolov8n-pose.pt")


model.train(
    data="coco8-pose.yaml",   
    epochs=10,
    imgsz=640
)

model = YOLO("runs/pose/train2/weights/best.pt")

model.predict(
    source="runs/pose/train2",   
    save=True
)