import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

def preprocess_dataset():
    INPUT_FOLDER = "train/images"
    OUTPUT_FOLDER = "train/preproc/images"

    LABEL_FOLDER = "train/labels"
    OUTPUT_LABEL_FOLDER = "train/preproc/labels"
    
    print("Starting preprocessing...")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)
    
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        imagePath = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(imagePath)

        if image is None:
            print("Could not read:", imagePath)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        
        # Save processed image
        outputPath = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(outputPath, blurred)

        # Copy matching label
        label_filename = os.path.splitext(filename)[0] + ".txt"
        labelPath = os.path.join(LABEL_FOLDER, label_filename)
        outputLabelPath = os.path.join(OUTPUT_LABEL_FOLDER, label_filename)

        if os.path.exists(labelPath):
            shutil.copy2(labelPath, outputLabelPath)
        else:
            print("WARNING: Label not found:", label_filename)

    print("Preprocessed inside:", OUTPUT_FOLDER)
    print("Labels copied inside:", OUTPUT_LABEL_FOLDER)


if __name__ == '__main__':
    preprocess_dataset()
    
    model = YOLO("yolo26n.pt") 
    
    results = model.train(
        data="data.yaml", 
        epochs=50,
        imgsz=640,
        batch=-1,
        patience=12,
        name="m4_median"
    )