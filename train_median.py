import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

def preprocess_dataset():
    INPUT_FOLDER = "train/images"
    LABEL_FOLDER = "train/labels"
    OUTPUT_FOLDER = "train/preproc/images"
    OUTPUT_LABEL_FOLDER = "train/preproc/labels"

    print("Starting preprocessing...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        imagePath = os.path.join(INPUT_FOLDER, filename)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)

        outputPath = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(outputPath, blurred)

        label_name = os.path.splitext(filename)[0] + ".txt"
        src_label = os.path.join(LABEL_FOLDER, label_name)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(OUTPUT_LABEL_FOLDER, label_name))
    print("Preprocessed inside ", OUTPUT_FOLDER)

def preprocess_valid():
	INPUT_FOLDER = "valid/images"
	LABEL_FOLDER = "valid/labels"
	OUTPUT_FOLDER = "valid/preproc/images"
	OUTPUT_LABEL_FOLDER = "valid/preproc/labels"

	print("Starting preprocessing...")
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)
	os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

	for filename in os.listdir(INPUT_FOLDER):
		if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
			continue
		imagePath = os.path.join(INPUT_FOLDER, filename)
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.medianBlur(gray, 3)

		outputPath = os.path.join(OUTPUT_FOLDER, filename)
		cv2.imwrite(outputPath, blurred)

		label_name = os.path.splitext(filename)[0] + ".txt"
		src_label = os.path.join(LABEL_FOLDER, label_name)
		if os.path.exists(src_label):
			shutil.copy(src_label, os.path.join(OUTPUT_LABEL_FOLDER, label_name))
	print("Preprocessed inside ", OUTPUT_FOLDER)

def preprocess_test():
	INPUT_FOLDER = "test/images"
	LABEL_FOLDER = "test/labels"
	OUTPUT_FOLDER = "test/preproc/images"
	OUTPUT_LABEL_FOLDER = "test/preproc/labels"

	print("Starting preprocessing...")
	os.makedirs(OUTPUT_FOLDER, exist_ok=True)
	os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

	for filename in os.listdir(INPUT_FOLDER):
		if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
			continue
		imagePath = os.path.join(INPUT_FOLDER, filename)
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.medianBlur(gray, 3)

		outputPath = os.path.join(OUTPUT_FOLDER, filename)
		cv2.imwrite(outputPath, blurred)

		label_name = os.path.splitext(filename)[0] + ".txt"
		src_label = os.path.join(LABEL_FOLDER, label_name)
		if os.path.exists(src_label):
			shutil.copy(src_label, os.path.join(OUTPUT_LABEL_FOLDER, label_name))
	print("Preprocessed inside ", OUTPUT_FOLDER)

if __name__ == '__main__':
	preprocess_dataset()
	preprocess_valid()
	preprocess_test()

	model = YOLO("yolo26n.pt") 

	results = model.train(
        data="data.yaml", 
        epochs=50,
        imgsz=640,
        batch=-1,
        patience=12,
        name="06_Median_50epochs_12patience"
    )


