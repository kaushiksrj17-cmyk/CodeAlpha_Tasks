import cv2
import os
import time
import csv
from collections import defaultdict

from ultralytics import YOLO


# ============================================================
# PROJECT 3 - PHASE 3E
# OBJECT DWELL TIME TRACKING
# ============================================================

INPUT_VIDEO = "assets/test_video.mp4"
OUTPUT_VIDEO = "output/dwell_time_video.mp4"

MODEL_PATH = "models/yolov8n.pt"

CSV_FILE = "output/dwell_time_report.csv"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("output", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       PROJECT 3 - PHASE 3E DWELL TIME TRACKING")
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
# TRACK DATA
# ============================================================

# First frame where object was detected
track_start_frame = {}

# Last frame where object was detected
track_last_frame = {}

# Object class
track_class = {}

# Maximum dwell time recorded
track_dwell_time = defaultdict(float)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

unique_ids = set()

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


csv_writer = csv.writer(
    csv_file
)


csv_writer.writerow([
    "Track_ID",
    "Class",
    "Start_Frame",
    "End_Frame",
    "Dwell_Time_Seconds"
])


# ============================================================
# MAIN LOOP
# ============================================================

print(
    "\nStarting dwell time tracking...\n"
)


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
    # PROCESS OBJECTS
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
            # PROCESS EACH TRACK
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
                # REGISTER NEW OBJECT
                # =============================================

                if track_id not in track_start_frame:

                    track_start_frame[
                        track_id
                    ] = frame_count


                # =============================================
                # UPDATE LAST FRAME
                # =============================================

                track_last_frame[
                    track_id
                ] = frame_count


                track_class[
                    track_id
                ] = class_name


                unique_ids.add(
                    track_id
                )


                # =============================================
                # CALCULATE DWELL TIME
                # =============================================

                start_frame = (
                    track_start_frame[
                        track_id
                    ]
                )


                frames_visible = (
                    frame_count -
                    start_frame +
                    1
                )


                dwell_seconds = (
                    frames_visible / fps
                    if fps > 0
                    else 0
                )


                track_dwell_time[
                    track_id
                ] = dwell_seconds


                # =============================================
                # DRAW BOUNDING BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # =============================================
                # DWELL LABEL
                # =============================================

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
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
                    0.5,
                    (0, 255, 0),
                    2
                )


                # =============================================
                # CENTER POINT
                # =============================================

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


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


    # ========================================================
    # FPS
    # ========================================================

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


    # ========================================================
    # CURRENT AVERAGE DWELL
    # ========================================================

    if track_dwell_time:

        average_dwell = (
            sum(
                track_dwell_time.values()
            )
            /
            len(
                track_dwell_time
            )
        )

    else:

        average_dwell = 0


    # ========================================================
    # STATISTICS PANEL
    # ========================================================

    panel_height = 165


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
        "DWELL TIME MONITOR",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # STATISTICS
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
        f"Tracked Objects : {len(unique_ids)}",
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Avg Dwell Time  : {average_dwell:.1f}s",
        (15, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS             : {current_fps:.1f}",
        (15, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE FRAME
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
# WRITE FINAL REPORT
# ============================================================

with open(
    CSV_FILE,
    mode="a",
    newline="",
    encoding="utf-8"
) as report_file:

    report_writer = csv.writer(
        report_file
    )


    for track_id in sorted(
        unique_ids
    ):

        start_frame = (
            track_start_frame[
                track_id
            ]
        )


        end_frame = (
            track_last_frame[
                track_id
            ]
        )


        frames_visible = (
            end_frame -
            start_frame +
            1
        )


        dwell_seconds = (
            frames_visible /
            fps
            if fps > 0
            else 0
        )


        report_writer.writerow([
            track_id,
            track_class.get(
                track_id,
                "unknown"
            ),
            start_frame,
            end_frame,
            round(
                dwell_seconds,
                2
            )
        ])


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
    "              PHASE 3E COMPLETE"
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
    f"Processing Time  : "
    f"{total_time:.2f} seconds"
)


if total_time > 0:

    print(
        f"Average FPS     : "
        f"{frame_count / total_time:.2f}"
    )


print(
    "\nObject Dwell Times"
)

print(
    "-" * 50
)


for track_id in sorted(
    unique_ids
):

    start_frame = (
        track_start_frame[
            track_id
        ]
    )


    end_frame = (
        track_last_frame[
            track_id
        ]
    )


    frames_visible = (
        end_frame -
        start_frame +
        1
    )


    dwell_seconds = (
        frames_visible /
        fps
        if fps > 0
        else 0
    )


    class_name = (
        track_class.get(
            track_id,
            "unknown"
        )
    )


    print(
        f"ID {track_id:<4} "
        f"{class_name:<12} "
        f"{dwell_seconds:.2f} seconds"
    )


print(
    "\nGenerated Files"
)


print(
    f"  [OK] {OUTPUT_VIDEO}"
)


print(
    f"  [OK] {CSV_FILE}"
)


print(
    "\nPhase 3E successfully completed!"
)


print(
    "=" * 70
)