import cv2
import os
import time
import csv
import math
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - PHASE 3D
# OBJECT DIRECTION DETECTION
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/direction_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

CSV_FILE = "output/direction_events.csv"


# ============================================================
# SETTINGS
# ============================================================

# Number of previous positions used for movement analysis
HISTORY_LENGTH = 8

# Minimum pixel movement before considering an object moving
MOVEMENT_THRESHOLD = 4

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3D OBJECT DIRECTION DETECTION")
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
# TRACK HISTORY
# ============================================================

track_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


# ============================================================
# TRACK INFORMATION
# ============================================================

track_classes = {}

track_directions = {}

# Direction statistics
direction_counts = defaultdict(set)

frame_count = 0

start_time = time.time()


# ============================================================
# CSV FILE
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
    "Center_X",
    "Center_Y",
    "Direction",
    "Movement_X",
    "Movement_Y",
    "Speed_Pixels"
])


# ============================================================
# DIRECTION FUNCTION
# ============================================================

def calculate_direction(history):

    if len(history) < 2:

        return "STATIONARY", 0, 0, 0


    # Oldest position
    old_x, old_y = history[0]

    # Newest position
    new_x, new_y = history[-1]


    dx = new_x - old_x

    dy = new_y - old_y


    distance = math.sqrt(
        dx * dx + dy * dy
    )


    # --------------------------------------------------------
    # Stationary
    # --------------------------------------------------------

    if distance < MOVEMENT_THRESHOLD:

        return "STATIONARY", dx, dy, distance


    # --------------------------------------------------------
    # Determine dominant axis
    # --------------------------------------------------------

    if abs(dx) > abs(dy):

        if dx > 0:

            direction = "RIGHT"

        else:

            direction = "LEFT"

    else:

        if dy > 0:

            direction = "DOWN"

        else:

            direction = "UP"


    return direction, dx, dy, distance


# ============================================================
# MAIN LOOP
# ============================================================

print("\nStarting direction detection...\n")


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

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ====================================================
                # UPDATE HISTORY
                # ====================================================

                history = track_history[track_id]

                history.append(
                    (center_x, center_y)
                )


                # ====================================================
                # CALCULATE DIRECTION
                # ====================================================

                direction, dx, dy, distance = (
                    calculate_direction(history)
                )


                track_directions[track_id] = direction


                # ====================================================
                # DIRECTION STATISTICS
                # ====================================================

                direction_counts[direction].add(
                    track_id
                )


                # ====================================================
                # CSV
                # ====================================================

                csv_writer.writerow([
                    frame_count,
                    track_id,
                    class_name,
                    center_x,
                    center_y,
                    direction,
                    dx,
                    dy,
                    round(distance, 2)
                ])


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
                # CENTER POINT
                # ====================================================

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 0, 255),
                    -1
                )


                # ====================================================
                # DRAW MOVEMENT TRAIL
                # ====================================================

                points = list(history)


                for i in range(
                    1,
                    len(points)
                ):

                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        (255, 0, 255),
                        2
                    )


                # ====================================================
                # DRAW DIRECTION ARROW
                # ====================================================

                if direction == "RIGHT":

                    arrow_end = (
                        center_x + 45,
                        center_y
                    )

                elif direction == "LEFT":

                    arrow_end = (
                        center_x - 45,
                        center_y
                    )

                elif direction == "DOWN":

                    arrow_end = (
                        center_x,
                        center_y + 45
                    )

                elif direction == "UP":

                    arrow_end = (
                        center_x,
                        center_y - 45
                    )

                else:

                    arrow_end = (
                        center_x,
                        center_y
                    )


                if direction != "STATIONARY":

                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        arrow_end,
                        (0, 255, 255),
                        3,
                        tipLength=0.3
                    )


                # ====================================================
                # LABEL
                # ====================================================

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{direction}"
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

    panel_height = 225

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (380, panel_height),
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
    # PANEL TITLE
    # ========================================================

    cv2.putText(
        frame,
        "OBJECT DIRECTION MONITOR",
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
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RIGHT : {len(direction_counts['RIGHT'])}",
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"LEFT  : {len(direction_counts['LEFT'])}",
        (15, 109),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"UP    : {len(direction_counts['UP'])}",
        (15, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"DOWN  : {len(direction_counts['DOWN'])}",
        (15, 163),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS   : {current_fps:.1f}",
        (15, 190),
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
print("              PHASE 3D COMPLETE")
print("=" * 70)


print(f"\nProcessed Frames : {frame_count}")


print(
    f"Processing Time  : "
    f"{total_time:.2f} seconds"
)


if total_time > 0:

    print(
        f"Average FPS     : "
        f"{frame_count / total_time:.2f}"
    )


print("\nDirection Statistics")
print("-" * 40)


directions = [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "STATIONARY"
]


for direction in directions:

    count = len(
        direction_counts[direction]
    )

    print(
        f"{direction:<15} : {count}"
    )


print("\nGenerated Files")

print(f"  [OK] {OUTPUT_VIDEO}")

print(f"  [OK] {CSV_FILE}")


print("\nPhase 3D successfully completed!")

print("=" * 70)