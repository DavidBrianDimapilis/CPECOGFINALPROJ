from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# NOTE: Please rename name to 03_xxxxx, 04_xxxxxxx
results = model.train(
	data = "data.yaml",
	epochs = 50,
	imgsz = 640,
	batch = -1,
	patience = 12,
	name = "05_No_Preprocess_50epochs_12patience"

		)

