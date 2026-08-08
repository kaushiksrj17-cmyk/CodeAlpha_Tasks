import cv2
import os
import time
import csv
import math
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - PHASE 3F
# REAL-TIME STATISTICS & PERFORMANCE MONITORING
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/realtime_monitor.mp4"

MODEL_PATH = "models/yolov8n.pt"

CSV_FILE = "output/realtime_statistics.csv"


# ============================================================
# SETTINGS
# ============================================================

HISTORY_LENGTH = 8

MOVEMENT_THRESHOLD = 4

# Counting line at 50% of video height
LINE_POSITION = 0.50


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3F REAL-TIME MONITORING")
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

    print(
        f"Expected: {INPUT_VIDEO}"
    )

    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


# ============================================================
# COUNTING LINE
# ============================================================

COUNT_LINE_Y = int(
    height * LINE_POSITION
)


print("\nVideo Information")
print("-" * 50)

print(
    f"Resolution   : "
    f"{width} x {height}"
)

print(
    f"FPS          : "
    f"{fps:.2f}"
)

print(
    f"Total Frames : "
    f"{total_frames}"
)

print(
    f"Counting Line: "
    f"Y = {COUNT_LINE_Y}"
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKING DATA
# ============================================================

track_history = defaultdict(
    lambda: deque(
        maxlen=HISTORY_LENGTH
    )
)


track_start_frame = {}

track_last_frame = {}

track_class = {}

track_direction = {}

counted_ids = set()


# ============================================================
# STATISTICS
# ============================================================

unique_ids = set()

entry_count = 0

exit_count = 0

class_counts = defaultdict(int)

direction_counts = defaultdict(set)

dwell_times = defaultdict(float)


# ============================================================
# PERFORMANCE
# ============================================================

frame_count = 0

start_time = time.time()

processing_times = deque(
    maxlen=30
)


# ============================================================
# CSV
# ============================================================

csv_file = open(
    CSV_FILE,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "Frame",
    "Current_Objects",
    "Unique_Objects",
    "Entered",
    "Exited",
    "FPS",
    "Avg_Dwell",
    "Processing_Time_ms"
])


# ============================================================
# DIRECTION FUNCTION
# ============================================================

def calculate_direction(history):

    if len(history) < 2:

        return "STATIONARY", 0, 0


    old_x, old_y = history[0]

    new_x, new_y = history[-1]


    dx = new_x - old_x

    dy = new_y - old_y


    distance = math.sqrt(
        dx * dx +
        dy * dy
    )


    if distance < MOVEMENT_THRESHOLD:

        return "STATIONARY", dx, dy


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


    return direction, dx, dy


# ============================================================
# MAIN LOOP
# ============================================================

print(
    "\nStarting real-time monitoring...\n"
)


while True:

    frame_start = time.time()


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
    # DRAW COUNTING LINE
    # ========================================================

    cv2.line(
        frame,
        (0, COUNT_LINE_Y),
        (width, COUNT_LINE_Y),
        (0, 255, 255),
        2
    )


    # ========================================================
    # PROCESS TRACKED OBJECTS
    # ========================================================

    if result.boxes is not None:

        boxes = result.boxes


        if boxes.id is not None:

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )


            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )


            confidences = (
                boxes.conf
                .cpu()
                .tolist()
            )


            coordinates = (
                boxes.xyxy
                .int()
                .cpu()
                .tolist()
            )


            current_objects = len(
                track_ids
            )


            # =================================================
            # EACH OBJECT
            # =================================================

            for (
                track_id,
                class_id,
                confidence,
                box
            ) in zip(
                track_ids,
                class_ids,
                confidences,
                coordinates
            ):

                x1, y1, x2, y2 = box


                class_name = (
                    model.names[class_id]
                )


                unique_ids.add(
                    track_id
                )


                class_counts[
                    class_name
                ] += 1


                current_class_counts[
                    class_name
                ] += 1


                track_class[
                    track_id
                ] = class_name


                # =============================================
                # CENTER
                # =============================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                history = (
                    track_history[
                        track_id
                    ]
                )


                previous_y = (
                    history[-1][1]
                    if len(history) > 0
                    else None
                )


                history.append(
                    (
                        center_x,
                        center_y
                    )
                )


                # =============================================
                # DIRECTION
                # =============================================

                direction, dx, dy = (
                    calculate_direction(
                        history
                    )
                )


                track_direction[
                    track_id
                ] = direction


                direction_counts[
                    direction
                ].add(
                    track_id
                )


                # =============================================
                # DWELL TIME
                # =============================================

                if track_id not in track_start_frame:

                    track_start_frame[
                        track_id
                    ] = frame_count


                track_last_frame[
                    track_id
                ] = frame_count


                dwell_seconds = (
                    (
                        frame_count -
                        track_start_frame[
                            track_id
                        ] +
                        1
                    )
                    / fps
                    if fps > 0
                    else 0
                )


                dwell_times[
                    track_id
                ] = dwell_seconds


                # =============================================
                # ENTRY / EXIT
                # =============================================

                if (
                    previous_y is not None
                    and track_id not in counted_ids
                ):

                    if (
                        previous_y < COUNT_LINE_Y
                        and center_y >= COUNT_LINE_Y
                    ):

                        entry_count += 1

                        counted_ids.add(
                            track_id
                        )


                    elif (
                        previous_y > COUNT_LINE_Y
                        and center_y <= COUNT_LINE_Y
                    ):

                        exit_count += 1

                        counted_ids.add(
                            track_id
                        )


                # =============================================
                # BOUNDING BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # =============================================
                # LABEL
                # =============================================

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{direction} "
                    f"{dwell_seconds:.1f}s"
                )


                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2
                )


                # =============================================
                # CENTER
                # =============================================

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    4,
                    (255, 0, 255),
                    -1
                )


                # =============================================
                # TRAIL
                # =============================================

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


    # ========================================================
    # PERFORMANCE
    # ========================================================

    frame_processing_time = (
        time.time() -
        frame_start
    )


    processing_times.append(
        frame_processing_time
    )


    total_elapsed = (
        time.time() -
        start_time
    )


    current_fps = (
        frame_count /
        total_elapsed
        if total_elapsed > 0
        else 0
    )


    avg_processing_ms = (
        sum(processing_times)
        /
        len(processing_times)
        *
        1000
        if processing_times
        else 0
    )


    # ========================================================
    # AVERAGE DWELL
    # ========================================================

    if dwell_times:

        average_dwell = (
            sum(
                dwell_times.values()
            )
            /
            len(
                dwell_times
            )
        )

    else:

        average_dwell = 0


    # ========================================================
    # STATISTICS PANEL
    # ========================================================

    panel_width = 395

    panel_height = 300


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (
            panel_width,
            panel_height
        ),
        (0, 0, 0),
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "SMART OBJECT ANALYTICS",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CORE METRICS
    # ========================================================

    cv2.putText(
        frame,
        f"Current Objects : {current_objects}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Unique Objects  : {len(unique_ids)}",
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"IN              : {entry_count}",
        (15, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"OUT             : {exit_count}",
        (15, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 165, 255),
        2
    )


    cv2.putText(
        frame,
        f"Average Dwell   : {average_dwell:.1f}s",
        (15, 154),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DIRECTION METRICS
    # ========================================================

    cv2.putText(
        frame,
        "DIRECTION",
        (15, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RIGHT : {len(direction_counts['RIGHT'])}",
        (15, 207),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"LEFT  : {len(direction_counts['LEFT'])}",
        (110, 207),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"UP    : {len(direction_counts['UP'])}",
        (205, 207),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"DOWN  : {len(direction_counts['DOWN'])}",
        (295, 207),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    cv2.putText(
        frame,
        "PERFORMANCE",
        (15, 237),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS             : {current_fps:.2f}",
        (15, 262),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame Time      : {avg_processing_ms:.1f} ms",
        (15, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
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

total_time = (
    time.time() -
    start_time
)


print("\n")

print("=" * 70)

print(
    "              PHASE 3F COMPLETE"
)

print("=" * 70)


print(
    f"\nProcessed Frames : "
    f"{frame_count}"
)


print(
    f"Unique Objects  : "
    f"{len(unique_ids)}"
)


print(
    f"Entered (IN)     : "
    f"{entry_count}"
)


print(
    f"Exited (OUT)     : "
    f"{exit_count}"
)


print(
    f"Average Dwell    : "
    f"{average_dwell:.2f} seconds"
)


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


for direction in [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "STATIONARY"
]:

    print(
        f"{direction:<15} : "
        f"{len(direction_counts[direction])}"
    )


print("\nGenerated Files")

print(
    f"  [OK] {OUTPUT_VIDEO}"
)

print(
    f"  [OK] {CSV_FILE}"
)


print(
    "\nPhase 3F successfully completed!"
)

print(
    "=" * 70
)