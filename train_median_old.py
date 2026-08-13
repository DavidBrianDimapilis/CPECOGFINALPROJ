import cv2
import numpy as np
import os
from ultralytics import YOLO

def preprocess_dataset():
    INPUT_FOLDER = "train/images"
    OUTPUT_FOLDER = "train/preproc/images"
    
    print("Starting preprocessing...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        imagePath = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(imagePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        
        outputPath = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(outputPath, blurred)

    print("Preprocessed inside ", OUTPUT_FOLDER)
if  __name__ == '__main__':
    preprocess_dataset()
    
    model = YOLO("yolo26n.pt") 
    
    results = model.train(
        data="data.yaml", 
        epochs=100,
        imgsz=640,
        batch=-1,
        patience=5,
        name="m4_median"
    )