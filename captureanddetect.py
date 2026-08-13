import cv2
import os
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIG ----------------
# Pointing directly to your newly trained 2-class model
MODEL_PATH = "runs/detect/04_150_epochs_15_patience/weights/best.pt"   
VIDEO_SOURCE = "C:/Users/David D/Documents/GitHub/CPECOGFINALPROJ/test/videos/videotest.mp4"      
SAVE_DIR = "captures"
# -----------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: could not open video source '{VIDEO_SOURCE}'")
        return

    print("\nControls:")
    print("  [SPACE] = play / pause")
    print("  [n]     = step forward one frame (while paused)")
    print("  [c]     = capture current frame and save")
    print("  [q]     = quit\n")

    paused = False
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                paused = True 
                continue

        # ---------------- THE FIX ----------------
        # 1. Isolate ROI: Crop top 35% of sky/buildings
        h, w = frame.shape[:2]
        start_y = int(h * 0.35)
        roi_frame = frame[start_y:h, 0:w]

        # 2. Live YOLO Inference with Max Recall Settings
        results = model.predict(
            source=roi_frame, 
            imgsz=1280, 
            conf=0.05,          # Dropped to 5% to catch the blurriest background riders
            iou=0.70,           # Increased to allow boxes to heavily overlap
            augment=True,       # Forces YOLO to double-check the image at multiple scales
            agnostic_nms=True,  # Stops bounding boxes from deleting each other
            verbose=False
        )

        # 3. Draw bounding boxes on the cropped ROI
        annotated_roi = results[0].plot()

        # 4. Paste the annotated road back onto the original frame for a seamless display
        display_frame = frame.copy()
        display_frame[start_y:h, 0:w] = annotated_roi
        # -----------------------------------------

        # Resize display window so it fits on your laptop screen
        preview = cv2.resize(display_frame, (960, int(960 * (h / w))))
        
        status = "PAUSED" if paused else "PLAYING (LIVE AI DETECTION)"
        cv2.putText(preview, f"[{status}]  SPACE: play/pause  N: step  C: capture  Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Motorcycle Helmet Live Stream", preview)

        # waitKey(1) allows the video to play smoothly in real-time
        key = cv2.waitKey(0 if paused else 1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused

        elif key == ord('n') and paused:
            ret, stepped = cap.read()
            if ret:
                frame = stepped
            else:
                print("End of video reached.")

        elif key == ord('c'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"capture_{timestamp}.jpg")
            
            # Save the frame that already has the boxes drawn on it!
            cv2.imwrite(save_path, display_frame)
            print(f"\nCaptured frame saved -> {save_path}")

            if len(results[0].boxes) == 0:
                print("  No objects detected above confidence threshold.")
            else:
                print(f"  Detected {len(results[0].boxes)} objects in this frame:")
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    print(f"  - {cls_name} ({conf:.2f})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()