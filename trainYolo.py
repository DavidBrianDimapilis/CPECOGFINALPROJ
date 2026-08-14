from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n.pt")

    # NOTE: Please rename name to 03_xxxxx, 04_xxxxxxx
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,
        patience=12,
        name="04_nonpreprocessed"
    )