from ultralytics import YOLO

# Load pose model
model = YOLO("yolov8n-pose.pt")

# Train using built-in dataset (AUTO DOWNLOAD)
model.train(
    data="coco8-pose.yaml",   # ✅ THIS WILL AUTO DOWNLOAD DATASET
    epochs=10,
    imgsz=640
)

model = YOLO("runs/pose/train2/weights/best.pt")

model.predict(
    source="runs/pose/train2",   # ✅ guaranteed exists
    save=True
)