import cv2
import os
import time
from ultralytics import YOLO

# ============================================================
# PROJECT 3 - PHASE 3
# PROFESSIONAL OBJECT TRACKING
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/tracked_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)

# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3 OBJECT TRACKING")
print("=" * 70)

print("\nLoading YOLOv8n model...")

try:
    model = YOLO(MODEL_PATH)
except Exception:
    print("Local model not found.")
    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")

print("YOLOv8n loaded successfully!")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("\nERROR: Could not open input video.")
    print(f"Expected file: {INPUT_VIDEO}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo Information")
print("-" * 50)
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")

# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# TRACKING VARIABLES
# ============================================================

unique_ids = set()

object_counts = {}

frame_count = 0

start_time = time.time()

# ============================================================
# TRACKING LOOP
# ============================================================

print("\nStarting ByteTrack tracking...\n")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    # --------------------------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

            class_ids = boxes.cls.int().cpu().tolist()

            confidences = boxes.conf.cpu().tolist()

            coordinates = boxes.xyxy.int().cpu().tolist()

            # ------------------------------------------------
            # DRAW TRACKING RESULTS
            # ------------------------------------------------

            for track_id, class_id, confidence, box in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates
            ):

                x1, y1, x2, y2 = box

                class_name = model.names[class_id]

                # Save unique ID
                unique_ids.add(track_id)

                # Count classes
                if class_name not in object_counts:
                    object_counts[class_name] = 0

                object_counts[class_name] += 1

                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (x1, y1 - 30),
                    (x1 + 220, y1),
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2
                )

                # ------------------------------------------------
                # TRACKING CENTER
                # ------------------------------------------------

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 255, 255),
                    -1
                )

# ============================================================
# REAL-TIME FPS
# ============================================================

    elapsed_time = time.time() - start_time

    current_fps = (
        frame_count / elapsed_time
        if elapsed_time > 0
        else 0
    )

# ============================================================
# STATISTICS PANEL
# ============================================================

    panel_height = 120

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (420, panel_height),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    cv2.putText(
        frame,
        "SMART OBJECT TRACKING",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Current Objects: {len(track_ids) if result.boxes is not None and result.boxes.id is not None else 0}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique Objects: {len(unique_ids)}",
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {current_fps:.1f}",
        (15, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

# ============================================================
# SAVE FRAME
# ============================================================

    out.write(frame)

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

# ============================================================
# FINAL REPORT
# ============================================================

total_time = time.time() - start_time

print("\n")
print("=" * 70)
print("              PHASE 3A COMPLETE")
print("=" * 70)

print(f"\nProcessed Frames : {frame_count}")

print(f"Unique Objects   : {len(unique_ids)}")

print(f"Processing Time  : {total_time:.2f} seconds")

if total_time > 0:
    print(
        f"Average FPS     : "
        f"{frame_count / total_time:.2f}"
    )

print("\nObject Detection Counts:")

for class_name, count in object_counts.items():

    print(
        f"  {class_name:<15} : {count}"
    )

print("\nOutput:")
print(f"  {OUTPUT_VIDEO}")

print("\nPhase 3A successfully completed!")
print("=" * 70)