from ultralytics import YOLO

MODEL_PATH = "models/02_realRun/best.pt"   # CHANGE 02_realRun to other folder names
DATA_YAML = "data.yaml"          

model = YOLO(MODEL_PATH)

print("\n TESTING against annotations ")
metrics = model.val(data=DATA_YAML, split="test", name="test_eval")

print("\n--- Test set results ---")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"Precision:{metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")
print("\nPer-class mAP50-95:")
for i, class_name in model.names.items():
    print(f"  {class_name}: {metrics.box.maps[i]:.4f}")

print("\n PREDICTING raw images\n")
results = model.predict(
    source="test/images",   
    save=True,
    conf=0.25,
    name="test_predictions"
)

print(f"\nDone. Annotated test images saved in: runs/detect/test_predictions/")
print(f"Test metrics/plots saved in: runs/detect/test_eval/")
