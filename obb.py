from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")  


model.train(
    data="datasets/roboflow/data.yaml",  
    epochs=10,
    imgsz=640
)


model.predict(
    source="datasets/roboflow/test/images",
    save=True
)

print(" OBB Training and Prediction Completed!")