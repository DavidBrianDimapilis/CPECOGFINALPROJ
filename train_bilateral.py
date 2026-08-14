import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

def preprocess_split(split_name):
    """Applies bilateral filter to images and copies labels for a given split."""
    
    INPUT_FOLDER = f"{split_name}/images"
    OUTPUT_FOLDER = f"{split_name}/preproc_bilateral/images"

    LABEL_FOLDER = f"{split_name}/labels"
    OUTPUT_LABEL_FOLDER = f"{split_name}/preproc_bilateral/labels"
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Skipping {split_name}: Folder not found ({INPUT_FOLDER})")
        return

    print(f"Starting bilateral preprocessing for {split_name}...")
    
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
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        outputPath = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(outputPath, blurred)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        labelPath = os.path.join(LABEL_FOLDER, label_filename)
        outputLabelPath = os.path.join(OUTPUT_LABEL_FOLDER, label_filename)

        if os.path.exists(labelPath):
            shutil.copy2(labelPath, outputLabelPath)
        else:
            print(f"WARNING: Label not found for {filename}")

    print(f"Finished {split_name}! Images in {OUTPUT_FOLDER}, Labels in {OUTPUT_LABEL_FOLDER}\n")


if __name__ == '__main__':
   
    for split in ["train", "valid", "test"]:
        preprocess_split(split)
    
    model = YOLO("yolo26n.pt") 
    
    results = model.train(
        data="data.yaml", 
        epochs=50,
        imgsz=640,
        batch=-1,
        patience=12,
        name="m4_bilateral"
    )