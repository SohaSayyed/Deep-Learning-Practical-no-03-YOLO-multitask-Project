from ultralytics import YOLO
import glob
import os

# ---------------------------
# Step 1: Load the model
# ---------------------------
model = YOLO("yolov8n-cls.pt")  # pretrained YOLOv8 classification model

# ---------------------------
# Step 2: Train the model
# ---------------------------
model.train(
    data="datasets/classification",  # dataset folder
    epochs=10,
    imgsz=224,
    save=True  # saves weights automatically
)

# ---------------------------
# Step 3: Predict on test images
# ---------------------------

# Collect all images from test folder and subfolders
test_images = glob.glob("datasets/classification/test/**/*.*", recursive=True)

# Filter for supported image formats
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
test_images = [img for img in test_images if img.lower().endswith(supported_formats)]

if len(test_images) == 0:
    print("No images found in test folder!")
else:
    # Run prediction
    model.predict(
        source=test_images,  # list of image paths
        save=True,           # saves predicted images
        show=True            # shows images in a window (optional)
    )

    print(f"Predictions completed. Check the 'runs/classify/predict' folder.")