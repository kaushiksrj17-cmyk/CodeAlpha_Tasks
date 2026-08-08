import cv2
import os
import time
import csv
from collections import defaultdict

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - PHASE 3B
# INTELLIGENT OBJECT COUNTING
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/counting_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

CSV_FILE = "output/object_counting.csv"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3B INTELLIGENT OBJECT COUNTING")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

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

    print("\nERROR: Could not open video.")
    print(f"Expected: {INPUT_VIDEO}")
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
# VIDEO WRITER
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

class_unique_ids = defaultdict(set)

frame_count = 0

start_time = time.time()


# ============================================================
# CSV REPORT
# ============================================================

csv_file = open(
    CSV_FILE,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Track_ID",
    "Class",
    "Confidence"
])


# ============================================================
# MAIN LOOP
# ============================================================

print("\nStarting object counting...\n")


while True:

    success, frame = cap.read()

    if not success:
        break


    frame_count += 1


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    result = results[0]


    current_objects = 0

    current_class_counts = defaultdict(int)


    # ========================================================
    # PROCESS TRACKED OBJECTS
    # ========================================================

    if result.boxes is not None:

        boxes = result.boxes


        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

            class_ids = boxes.cls.int().cpu().tolist()

            confidences = boxes.conf.cpu().tolist()

            coordinates = boxes.xyxy.int().cpu().tolist()


            current_objects = len(track_ids)


            for track_id, class_id, confidence, box in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates
            ):

                x1, y1, x2, y2 = box

                class_name = model.names[class_id]


                # ============================================
                # UNIQUE OBJECT COUNT
                # ============================================

                unique_ids.add(track_id)

                class_unique_ids[class_name].add(track_id)


                # ============================================
                # CURRENT CLASS COUNT
                # ============================================

                current_class_counts[class_name] += 1


                # ============================================
                # CSV
                # ============================================

                csv_writer.writerow([
                    frame_count,
                    track_id,
                    class_name,
                    round(confidence, 3)
                ])


                # ============================================
                # DRAW BOUNDING BOX
                # ============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # ============================================
                # LABEL
                # ============================================

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


                # ============================================
                # CENTER POINT
                # ============================================

                center_x = int((x1 + x2) / 2)

                center_y = int((y1 + y2) / 2)


                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 255, 255),
                    -1
                )


    # ========================================================
    # FPS
    # ========================================================

    elapsed = time.time() - start_time


    current_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0
    )


    # ========================================================
    # STATISTICS PANEL
    # ========================================================

    panel_height = 210


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (370, panel_height),
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


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "SMART OBJECT COUNTING",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # GENERAL STATISTICS
    # ========================================================

    cv2.putText(
        frame,
        f"Current Objects : {current_objects}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Unique Objects  : {len(unique_ids)}",
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS             : {current_fps:.1f}",
        (15, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CLASS STATISTICS
    # ========================================================

    y_position = 135


    for class_name in sorted(class_unique_ids):

        unique_count = len(
            class_unique_ids[class_name]
        )


        current_count = current_class_counts.get(
            class_name,
            0
        )


        text = (
            f"{class_name}: "
            f"{current_count} current / "
            f"{unique_count} total"
        )


        cv2.putText(
            frame,
            text,
            (15, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        y_position += 23


        # Prevent panel overflow

        if y_position > panel_height - 5:

            break


    # ========================================================
    # SAVE FRAME
    # ========================================================

    out.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

out.release()

csv_file.close()


# ============================================================
# FINAL STATISTICS
# ============================================================

total_time = time.time() - start_time


print("\n")
print("=" * 70)
print("              PHASE 3B COMPLETE")
print("=" * 70)


print(f"\nProcessed Frames : {frame_count}")

print(
    f"Unique Objects  : "
    f"{len(unique_ids)}"
)


print(
    f"Processing Time  : "
    f"{total_time:.2f} seconds"
)


if total_time > 0:

    average_fps = frame_count / total_time

else:

    average_fps = 0


print(
    f"Average FPS     : "
    f"{average_fps:.2f}"
)


print("\nUnique Objects By Class")
print("-" * 40)


for class_name in sorted(class_unique_ids):

    count = len(
        class_unique_ids[class_name]
    )

    print(
        f"{class_name:<15} : {count}"
    )


print("\nGenerated Files")

print(f"  [OK] {OUTPUT_VIDEO}")

print(f"  [OK] {CSV_FILE}")


print("\nPhase 3B successfully completed!")

print("=" * 70)