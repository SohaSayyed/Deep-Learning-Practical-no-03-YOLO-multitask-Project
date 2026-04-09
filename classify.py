from ultralytics import YOLO
import glob
import os


model = YOLO("yolov8n-cls.pt") 


model.train(
    data="datasets/classification", 
    epochs=10,
    imgsz=224,
    save=True 
)




test_images = glob.glob("datasets/classification/test/**/*.*", recursive=True)


supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
test_images = [img for img in test_images if img.lower().endswith(supported_formats)]

if len(test_images) == 0:
    print("No images found in test folder!")
else:
   
    model.predict(
        source=test_images,  
        save=True,           
        show=True            
    )

    print(f"Predictions completed. Check the 'runs/classify/predict' folder.")