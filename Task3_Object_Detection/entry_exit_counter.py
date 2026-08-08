import cv2
import os
import time
import csv
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - PHASE 3C
# ENTRY / EXIT COUNTING
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/entry_exit_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

CSV_FILE = "output/entry_exit_events.csv"


# ============================================================
# SETTINGS
# ============================================================

# Counting line position as a percentage of video height.
# 0.50 means the line is at the center.
LINE_POSITION = 0.50

# Minimum movement required before deciding a crossing.
MIN_MOVEMENT = 5


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3C ENTRY / EXIT COUNTING")
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


# ============================================================
# COUNTING LINE
# ============================================================

COUNT_LINE_Y = int(height * LINE_POSITION)


print("\nVideo Information")
print("-" * 50)

print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Counting Line: Y = {COUNT_LINE_Y}")


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
# TRACKING DATA
# ============================================================

# Track ID -> recent center positions
track_history = defaultdict(
    lambda: deque(maxlen=10)
)

# Track IDs that have already crossed
counted_ids = set()

# Track ID -> object class
track_classes = {}


# ============================================================
# COUNTERS
# ============================================================

entry_count = 0
exit_count = 0

class_entries = defaultdict(int)
class_exits = defaultdict(int)

frame_count = 0

start_time = time.time()


# ============================================================
# CSV
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
    "Direction",
    "Center_X",
    "Center_Y"
])


# ============================================================
# MAIN LOOP
# ============================================================

print("\nStarting Entry / Exit Tracking...\n")


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


    # ========================================================
    # DRAW COUNTING LINE
    # ========================================================

    cv2.line(
        frame,
        (0, COUNT_LINE_Y),
        (width, COUNT_LINE_Y),
        (0, 255, 255),
        3
    )


    cv2.putText(
        frame,
        "COUNTING LINE",
        (10, COUNT_LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


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

                track_classes[track_id] = class_name


                # ====================================================
                # CENTER POINT
                # ====================================================

                center_x = int((x1 + x2) / 2)

                center_y = int((y1 + y2) / 2)


                # ====================================================
                # SAVE POSITION HISTORY
                # ====================================================

                history = track_history[track_id]

                previous_y = (
                    history[-1][1]
                    if len(history) > 0
                    else None
                )

                history.append(
                    (center_x, center_y)
                )


                # ====================================================
                # DETERMINE LINE CROSSING
                # ====================================================

                if (
                    previous_y is not None
                    and track_id not in counted_ids
                ):

                    movement = center_y - previous_y


                    # ------------------------------------------------
                    # TOP -> BOTTOM
                    # ------------------------------------------------

                    if (
                        previous_y < COUNT_LINE_Y
                        and center_y >= COUNT_LINE_Y
                        and abs(movement) >= MIN_MOVEMENT
                    ):

                        entry_count += 1

                        class_entries[class_name] += 1

                        counted_ids.add(track_id)


                        csv_writer.writerow([
                            frame_count,
                            track_id,
                            class_name,
                            "IN",
                            center_x,
                            center_y
                        ])


                        print(
                            f"[ENTRY] "
                            f"{class_name} "
                            f"ID:{track_id}"
                        )


                    # ------------------------------------------------
                    # BOTTOM -> TOP
                    # ------------------------------------------------

                    elif (
                        previous_y > COUNT_LINE_Y
                        and center_y <= COUNT_LINE_Y
                        and abs(movement) >= MIN_MOVEMENT
                    ):

                        exit_count += 1

                        class_exits[class_name] += 1

                        counted_ids.add(track_id)


                        csv_writer.writerow([
                            frame_count,
                            track_id,
                            class_name,
                            "OUT",
                            center_x,
                            center_y
                        ])


                        print(
                            f"[EXIT] "
                            f"{class_name} "
                            f"ID:{track_id}"
                        )


                # ====================================================
                # BOUNDING BOX
                # ====================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # ====================================================
                # LABEL
                # ====================================================

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


                # ====================================================
                # CENTER POINT
                # ====================================================

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 0, 255),
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

    panel_height = 170

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (360, panel_height),
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
        "ENTRY / EXIT MONITOR",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Current Objects : {current_objects}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ENTERED (IN)    : {entry_count}",
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"EXITED (OUT)    : {exit_count}",
        (15, 109),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS              : {current_fps:.1f}",
        (15, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


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
# FINAL REPORT
# ============================================================

total_time = time.time() - start_time


print("\n")
print("=" * 70)
print("              PHASE 3C COMPLETE")
print("=" * 70)


print(f"\nProcessed Frames : {frame_count}")

print(f"Entered (IN)     : {entry_count}")

print(f"Exited (OUT)     : {exit_count}")

print(
    f"Processing Time  : "
    f"{total_time:.2f} seconds"
)


if total_time > 0:

    print(
        f"Average FPS     : "
        f"{frame_count / total_time:.2f}"
    )


print("\nEntry Counts By Class")
print("-" * 40)

for class_name in sorted(class_entries):

    print(
        f"{class_name:<15} : "
        f"{class_entries[class_name]}"
    )


print("\nExit Counts By Class")
print("-" * 40)

for class_name in sorted(class_exits):

    print(
        f"{class_name:<15} : "
        f"{class_exits[class_name]}"
    )


print("\nGenerated Files")

print(f"  [OK] {OUTPUT_VIDEO}")

print(f"  [OK] {CSV_FILE}")


print("\nPhase 3C successfully completed!")

print("=" * 70)