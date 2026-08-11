from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# NOTE: Please rename name to 03_xxxxx, 04_xxxxxxx
results = model.train(
	data = "data.yaml",
	epochs = 100,
	imgsz = 640,
	batch = 16,
	name = "02_realRun"

		)

