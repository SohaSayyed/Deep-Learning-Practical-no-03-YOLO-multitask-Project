from ultralytics import YOLO

# ---------------------------
# Step 1: Load OBB model
# ---------------------------
model = YOLO("yolov8n-obb.pt")   # Pretrained YOLOv8 OBB model

# ---------------------------
# Step 2: Train the model
# ---------------------------
model.train(
    data="datasets/roboflow/data.yaml",   # Your YAML file path
    epochs=10,
    imgsz=640
)

# ---------------------------
# Step 3: Predict on test images
# ---------------------------
model.predict(
    source="datasets/roboflow/test/images",  # Test images path
    save=True
)

print("✅ OBB Training and Prediction Completed!")