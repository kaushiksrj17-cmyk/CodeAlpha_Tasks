from ultralytics import YOLO
import cv2
import time

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

previous_time = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    current_time = time.time()

    if previous_time:
        fps = 1 / (current_time - previous_time)
    else:
        fps = 0

    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "CodeAlpha - Real-Time Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam detection stopped.")