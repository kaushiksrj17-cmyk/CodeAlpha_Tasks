import cv2
import os
import time
import csv
import math
from collections import defaultdict, deque

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - CODEALPHA
# FINAL PROFESSIONAL OBJECT DETECTION & TRACKING SYSTEM
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

OUTPUT_VIDEO = "output/final_tracking_system.mp4"

CSV_FILE = "output/final_tracking_events.csv"


# ============================================================
# SYSTEM SETTINGS
# ============================================================

HISTORY_LENGTH = 8

MOVEMENT_THRESHOLD = 4

COUNT_LINE_POSITION = 0.50


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)

print(
    "        CODEALPHA PROJECT 3"
)

print(
    "   FINAL AI OBJECT DETECTION & TRACKING SYSTEM"
)

print("=" * 75)


print("\nFeatures Enabled:")

print("  [✓] YOLOv8 Object Detection")
print("  [✓] ByteTrack Multi-Object Tracking")
print("  [✓] Persistent Object IDs")
print("  [✓] Object Counting")
print("  [✓] Entry / Exit Detection")
print("  [✓] Direction Detection")
print("  [✓] Dwell Time Tracking")
print("  [✓] FPS Monitoring")
print("  [✓] Performance Monitoring")
print("  [✓] CSV Event Logging")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLOv8n model...")

try:

    model = YOLO(MODEL_PATH)

except Exception:

    print(
        "Local model not found."
    )

    print(
        "Downloading YOLOv8n..."
    )

    model = YOLO(
        "yolov8n.pt"
    )


print(
    "YOLOv8n loaded successfully!"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    INPUT_VIDEO
)


if not cap.isOpened():

    print(
        "\nERROR: Could not open input video."
    )

    print(
        f"Expected file: {INPUT_VIDEO}"
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
    height *
    COUNT_LINE_POSITION
)


# ============================================================
# VIDEO INFORMATION
# ============================================================

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
    f"Count Line   : "
    f"{COUNT_LINE_Y}"
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


# ============================================================
# OBJECT STATISTICS
# ============================================================

unique_ids = set()

counted_ids = set()

entry_ids = set()

exit_ids = set()


entry_count = 0

exit_count = 0


# ============================================================
# CLASS STATISTICS
# ============================================================

class_unique_ids = defaultdict(set)

class_entries = defaultdict(int)

class_exits = defaultdict(int)


# ============================================================
# DIRECTION STATISTICS
# ============================================================

direction_ids = defaultdict(set)


# ============================================================
# DWELL TIME
# ============================================================

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
# CSV EVENT LOGGER
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
    "Track_ID",
    "Class",
    "Center_X",
    "Center_Y",
    "Direction",
    "Dwell_Time",
    "Event"
])


# ============================================================
# DIRECTION FUNCTION
# ============================================================

def calculate_direction(history):

    if len(history) < 2:

        return (
            "STATIONARY",
            0,
            0
        )


    old_x, old_y = history[0]

    new_x, new_y = history[-1]


    dx = new_x - old_x

    dy = new_y - old_y


    distance = math.sqrt(
        dx * dx +
        dy * dy
    )


    if distance < MOVEMENT_THRESHOLD:

        return (
            "STATIONARY",
            dx,
            dy
        )


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


    return (
        direction,
        dx,
        dy
    )


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

print(
    "\nStarting final AI tracking system..."
)

print(
    "Processing video...\n"
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


    current_classes = defaultdict(int)


    # ========================================================
    # COUNTING LINE
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
        "ENTRY / EXIT LINE",
        (
            10,
            COUNT_LINE_Y - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )


    # ========================================================
    # TRACKED OBJECTS
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
            # PROCESS EVERY OBJECT
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


                # =============================================
                # UNIQUE ID
                # =============================================

                unique_ids.add(
                    track_id
                )


                class_unique_ids[
                    class_name
                ].add(
                    track_id
                )


                current_classes[
                    class_name
                ] += 1


                track_class[
                    track_id
                ] = class_name


                # =============================================
                # CENTER POINT
                # =============================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # =============================================
                # POSITION HISTORY
                # =============================================

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

                (
                    direction,
                    dx,
                    dy
                ) = calculate_direction(
                    history
                )


                track_direction[
                    track_id
                ] = direction


                direction_ids[
                    direction
                ].add(
                    track_id
                )


                # =============================================
                # DWELL TIME
                # =============================================

                if (
                    track_id
                    not in
                    track_start_frame
                ):

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

                    /

                    fps

                    if fps > 0

                    else 0
                )


                dwell_times[
                    track_id
                ] = dwell_seconds


                # =============================================
                # ENTRY / EXIT DETECTION
                # =============================================

                event = ""


                if (
                    previous_y is not None
                    and track_id
                    not in counted_ids
                ):


                    # -----------------------------------------
                    # TOP -> BOTTOM
                    # -----------------------------------------

                    if (
                        previous_y < COUNT_LINE_Y
                        and center_y >= COUNT_LINE_Y
                    ):

                        entry_count += 1

                        entry_ids.add(
                            track_id
                        )

                        counted_ids.add(
                            track_id
                        )

                        class_entries[
                            class_name
                        ] += 1

                        event = "ENTRY"


                    # -----------------------------------------
                    # BOTTOM -> TOP
                    # -----------------------------------------

                    elif (
                        previous_y > COUNT_LINE_Y
                        and center_y <= COUNT_LINE_Y
                    ):

                        exit_count += 1

                        exit_ids.add(
                            track_id
                        )

                        counted_ids.add(
                            track_id
                        )

                        class_exits[
                            class_name
                        ] += 1

                        event = "EXIT"


                # =============================================
                # CSV EVENT LOG
                # =============================================

                csv_writer.writerow([
                    frame_count,
                    track_id,
                    class_name,
                    center_x,
                    center_y,
                    direction,
                    round(
                        dwell_seconds,
                        2
                    ),
                    event
                ])


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
                # CENTER POINT
                # =============================================

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    5,
                    (255, 0, 255),
                    -1
                )


                # =============================================
                # TRACK TRAIL
                # =============================================

                points = list(
                    history
                )


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


                # =============================================
                # DIRECTION ARROW
                # =============================================

                if direction == "RIGHT":

                    arrow_end = (
                        center_x + 40,
                        center_y
                    )

                elif direction == "LEFT":

                    arrow_end = (
                        center_x - 40,
                        center_y
                    )

                elif direction == "DOWN":

                    arrow_end = (
                        center_x,
                        center_y + 40
                    )

                elif direction == "UP":

                    arrow_end = (
                        center_x,
                        center_y - 40
                    )

                else:

                    arrow_end = (
                        center_x,
                        center_y
                    )


                if (
                    direction !=
                    "STATIONARY"
                ):

                    cv2.arrowedLine(
                        frame,
                        (
                            center_x,
                            center_y
                        ),
                        arrow_end,
                        (0, 255, 255),
                        3,
                        tipLength=0.3
                    )


                # =============================================
                # OBJECT LABEL
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


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    frame_processing_time = (
        time.time() -
        frame_start
    )


    processing_times.append(
        frame_processing_time
    )


    elapsed = (
        time.time() -
        start_time
    )


    current_fps = (
        frame_count /
        elapsed
        if elapsed > 0
        else 0
    )


    average_frame_time = (

        sum(
            processing_times
        )

        /

        len(
            processing_times
        )

        *

        1000

        if processing_times

        else 0
    )


    # ========================================================
    # AVERAGE DWELL TIME
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

    panel_width = 410

    panel_height = 330


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
        0.70,
        frame,
        0.30,
        0
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AI OBJECT TRACKING ANALYTICS",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # OBJECT METRICS
    # ========================================================

    cv2.putText(
        frame,
        f"Current Objects : {current_objects}",
        (15, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Unique Objects  : {len(unique_ids)}",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ENTERED (IN)    : {entry_count}",
        (15, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"EXITED (OUT)    : {exit_count}",
        (15, 126),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 165, 255),
        2
    )


    cv2.putText(
        frame,
        f"AVG DWELL       : {average_dwell:.1f}s",
        (15, 149),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DIRECTION
    # ========================================================

    cv2.putText(
        frame,
        "DIRECTION",
        (15, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RIGHT : {len(direction_ids['RIGHT'])}",
        (15, 203),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"LEFT  : {len(direction_ids['LEFT'])}",
        (110, 203),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"UP    : {len(direction_ids['UP'])}",
        (205, 203),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"DOWN  : {len(direction_ids['DOWN'])}",
        (295, 203),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    cv2.putText(
        frame,
        "OBJECT CLASSES",
        (15, 233),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )


    class_y = 256


    for class_name in sorted(
        class_unique_ids
    ):

        total_class_objects = len(
            class_unique_ids[
                class_name
            ]
        )


        current_class_objects = (
            current_classes.get(
                class_name,
                0
            )
        )


        text = (
            f"{class_name}: "
            f"{current_class_objects} "
            f"current / "
            f"{total_class_objects} "
            f"total"
        )


        cv2.putText(
            frame,
            text,
            (
                15,
                class_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            2
        )


        class_y += 20


        if class_y > 300:

            break


    # ========================================================
    # PERFORMANCE
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {current_fps:.2f}",
        (
            width - 145,
            height - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"{average_frame_time:.1f} ms/frame",
        (
            width - 175,
            height - 12
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    out.write(
        frame
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

out.release()

csv_file.close()


# ============================================================
# FINAL REPORT
# ============================================================

total_processing_time = (
    time.time() -
    start_time
)


average_fps = (

    frame_count /
    total_processing_time

    if total_processing_time > 0

    else 0
)


print("\n")

print("=" * 75)

print(
    "              FINAL SYSTEM COMPLETE"
)

print("=" * 75)


print(
    "\nPERFORMANCE SUMMARY"
)

print("-" * 50)


print(
    f"Processed Frames : "
    f"{frame_count}"
)


print(
    f"Unique Objects   : "
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
    f"{total_processing_time:.2f} seconds"
)


print(
    f"Average FPS      : "
    f"{average_fps:.2f}"
)


# ============================================================
# CLASS REPORT
# ============================================================

print(
    "\nOBJECT CLASS SUMMARY"
)

print("-" * 50)


for class_name in sorted(
    class_unique_ids
):

    total_count = len(
        class_unique_ids[
            class_name
        ]
    )


    print(
        f"{class_name:<15} : "
        f"{total_count}"
    )


# ============================================================
# DIRECTION REPORT
# ============================================================

print(
    "\nDIRECTION SUMMARY"
)

print("-" * 50)


for direction in [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "STATIONARY"
]:

    count = len(
        direction_ids[
            direction
        ]
    )


    print(
        f"{direction:<15} : "
        f"{count}"
    )


# ============================================================
# FILES
# ============================================================

print(
    "\nGENERATED FILES"
)

print("-" * 50)


print(
    f"  [OK] {OUTPUT_VIDEO}"
)


print(
    f"  [OK] {CSV_FILE}"
)


print(
    "\nAll Phase 3 tracking features "
    "successfully integrated!"
)


print("=" * 75)