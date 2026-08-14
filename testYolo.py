from ultralytics import YOLO

MODEL_PATH = "runs/detect/m4_median-6/weights/best.pt"
DATA_YAML = "data.yaml"


if __name__ == "__main__":

    model = YOLO(MODEL_PATH)

    print("\nTESTING against annotations")

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        name="test_eval",
        workers=0
    )

    print("\n--- Test set results ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"Precision:{metrics.box.mp:.4f}")
    print(f"Recall:   {metrics.box.mr:.4f}")

    print("\nPer-class mAP50-95:")
    for i, class_name in model.names.items():
        print(f"  {class_name}: {metrics.box.maps[i]:.4f}")

    print("\nPREDICTING raw images\n")

    results = model.predict(
        source="test/images",
        save=True,
        conf=0.25,
        name="test_predictions",
        workers=0
    )

    print("\nDone. Annotated test images saved in:")
    print("runs/detect/test_predictions/")

    print("\nTest metrics/plots saved in:")
    print("runs/detect/test_eval/")