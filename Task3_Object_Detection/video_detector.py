from ultralytics import YOLO
import cv2
import os
import time

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

input_path = "assets/test_video.mp4"
output_path = "outputs/detected_video.mp4"

if not os.path.exists(input_path):
    print(f"Error: Video not found: {input_path}")
    exit()

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video resolution: {width}x{height}")
print(f"Video FPS: {fps:.1f}")

os.makedirs("outputs", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0
start_time = time.time()

print("Starting video detection...")

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed frames: {frame_count}")

cap.release()
out.release()

elapsed = time.time() - start_time

print("\nVideo detection complete!")
print(f"Frames processed: {frame_count}")
print(f"Processing time: {elapsed:.1f} seconds")
print(f"Output saved to: {output_path}")