
import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/site_test.mp4"

CONFIDENCE = 0.35
IMGSZ = 640

def main():

    # Load custom YOLO model
    model = YOLO(MODEL_PATH)

    # Open recorded video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    print("Video opened successfully.")

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # Run object detection
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMGSZ,
            verbose=False
        )

        # Draw detection boxes
        annotated_frame = results[0].plot()

        # Display output
        cv2.imshow(
            "Forklift Safety Detection",
            annotated_frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()