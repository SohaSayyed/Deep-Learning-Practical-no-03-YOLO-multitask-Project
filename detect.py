from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")   # nano model

# Train
model.train(
    data="datasets/detection/data.yaml",
    epochs=10,
    imgsz=640
) 

# Predict
model.predict(
    source="datasets/detection/test/images",
    save=True
)