from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
	data = "data.yaml",
	epochs = 10,
	imgsz = 320,
	batch = 4,
	name = "01_firstRun"

		)
