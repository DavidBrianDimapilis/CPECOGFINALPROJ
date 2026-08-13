import cv2
import numpy as np
import os
from ultralytics import YOLO

def preprocess_dataset():
    INPUT_FOLDER = "train/images"
    OUTPUT_FOLDER = "train/preproc"
    
    print("Starting preprocessing...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        imagePath = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(imagePath)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.BilateralBlur(gray, 9, 75, 75)
        
        outputPath = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(outputPath, blurred)

    print("Preprocessed inside ", OUTPUT_FOLDER)

if _name_ == '_main_':
    preprocess_dataset()
    
    model = YOLO("yolo26n.pt") 
    
    results = model.train(
        data="data.yaml", 
        epochs=100,
        imgsz=640,
        batch=-1,
        patience=5,
        name="04_150_epochs_15_patience"
    )