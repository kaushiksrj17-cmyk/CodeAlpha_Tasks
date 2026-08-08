"""
Core AI object detection and tracking engine.

CodeAlpha Project 3
Uses YOLOv8n + ByteTrack.
"""

import cv2
import os
import time
import csv

from collections import defaultdict, deque

from ultralytics import YOLO

from utils import (
    calculate_direction,
    calculate_center,
    calculate_fps,
    calculate_average_dwell,
    draw_tracking_line,
    draw_track_trail,
    draw_direction_arrow
)


HISTORY_LENGTH = 8

MOVEMENT_THRESHOLD = 4

COUNT_LINE_POSITION = 0.50


def run_tracking(
    input_video,
    model_path,
    output_video,
    csv_file
):
    """
    Run the complete object detection and tracking pipeline.
    """

    os.makedirs(
        os.path.dirname(output_video),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(csv_file),
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading YOLOv8n model...")

    try:
        model = YOLO(model_path)

    except Exception:

        print(
            "Local model unavailable."
        )

        print(
            "Downloading YOLOv8n..."
        )

        model = YOLO("yolov8n.pt")

    print(
        "YOLOv8n loaded successfully!"
    )

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open video."
        )

        return

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

    count_line_y = int(
        height *
        COUNT_LINE_POSITION
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

    # --------------------------------------------------------
    # VIDEO OUTPUT
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    # --------------------------------------------------------
    # TRACKING DATA
    # --------------------------------------------------------

    track_history = defaultdict(
        lambda: deque(
            maxlen=HISTORY_LENGTH
        )
    )

    track_start_frame = {}

    track_class = {}

    track_direction = {}

    dwell_times = {}

    # --------------------------------------------------------
    # OBJECT STATISTICS
    # --------------------------------------------------------

    unique_ids = set()

    counted_ids = set()

    entry_ids = set()

    exit_ids = set()

    entry_count = 0

    exit_count = 0

    class_unique_ids = defaultdict(set)

    class_entries = defaultdict(int)

    class_exits = defaultdict(int)

    direction_ids = defaultdict(set)

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    frame_count = 0

    start_time = time.time()

    processing_times = deque(
        maxlen=30
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_output = open(
        csv_file,
        mode="w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_output
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

    print(
        "\nStarting ByteTrack tracking...\n"
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        frame_start = time.time()

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        current_objects = 0

        current_classes = defaultdict(int)

        # ----------------------------------------------------
        # COUNTING LINE
        # ----------------------------------------------------

        draw_tracking_line(
            frame,
            count_line_y,
            width
        )

        # ----------------------------------------------------
        # TRACKED OBJECTS
        # ----------------------------------------------------

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            coordinates = (
                result.boxes.xyxy
                .int()
                .cpu()
                .tolist()
            )

            current_objects = len(
                track_ids
            )

            # ------------------------------------------------
            # EACH OBJECT
            # ------------------------------------------------

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

                # --------------------------------------------
                # CENTER
                # --------------------------------------------

                center_x, center_y = (
                    calculate_center(
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                # --------------------------------------------
                # HISTORY
                # --------------------------------------------

                history = (
                    track_history[
                        track_id
                    ]
                )

                previous_y = (
                    history[-1][1]
                    if history
                    else None
                )

                history.append(
                    (
                        center_x,
                        center_y
                    )
                )

                # --------------------------------------------
                # DIRECTION
                # --------------------------------------------

                direction, dx, dy = (
                    calculate_direction(
                        history,
                        MOVEMENT_THRESHOLD
                    )
                )

                track_direction[
                    track_id
                ] = direction

                direction_ids[
                    direction
                ].add(
                    track_id
                )

                # --------------------------------------------
                # DWELL TIME
                # --------------------------------------------

                if (
                    track_id
                    not in track_start_frame
                ):

                    track_start_frame[
                        track_id
                    ] = frame_count

                if fps > 0:

                    dwell_seconds = (
                        (
                            frame_count
                            -
                            track_start_frame[
                                track_id
                            ]
                            + 1
                        )
                        /
                        fps
                    )

                else:

                    dwell_seconds = 0

                dwell_times[
                    track_id
                ] = dwell_seconds

                # --------------------------------------------
                # ENTRY / EXIT
                # --------------------------------------------

                event = ""

                if (
                    previous_y is not None
                    and track_id
                    not in counted_ids
                ):

                    # Top -> Bottom
                    if (
                        previous_y < count_line_y
                        and center_y >= count_line_y
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

                    # Bottom -> Top
                    elif (
                        previous_y > count_line_y
                        and center_y <= count_line_y
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

                # --------------------------------------------
                # CSV
                # --------------------------------------------

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

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # --------------------------------------------
                # CENTER
                # --------------------------------------------

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

                # --------------------------------------------
                # TRAIL
                # --------------------------------------------

                draw_track_trail(
                    frame,
                    list(history)
                )

                # --------------------------------------------
                # DIRECTION ARROW
                # --------------------------------------------

                draw_direction_arrow(
                    frame,
                    center_x,
                    center_y,
                    direction
                )

                # --------------------------------------------
                # LABEL
                # --------------------------------------------

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

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        processing_time = (
            time.time()
            -
            frame_start
        )

        processing_times.append(
            processing_time
        )

        current_fps = calculate_fps(
            frame_count,
            start_time
        )

        average_frame_time = (
            sum(processing_times)
            /
            len(processing_times)
            *
            1000
            if processing_times
            else 0
        )

        average_dwell = (
            calculate_average_dwell(
                dwell_times
            )
        )

        # ----------------------------------------------------
        # STATISTICS PANEL
        # ----------------------------------------------------

        panel_width = 410
        panel_height = 315

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

        cv2.putText(
            frame,
            "AI OBJECT TRACKING ANALYTICS",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Current Objects : {current_objects}",
            (15, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Unique Objects  : {len(unique_ids)}",
            (15, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"ENTERED (IN)    : {entry_count}",
            (15, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"EXITED (OUT)    : {exit_count}",
            (15, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            f"AVG DWELL       : {average_dwell:.1f}s",
            (15, 149),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

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
            f"LEFT : {len(direction_ids['LEFT'])}",
            (110, 203),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"UP : {len(direction_ids['UP'])}",
            (205, 203),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"DOWN : {len(direction_ids['DOWN'])}",
            (295, 203),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # OBJECT CLASSES
        # ----------------------------------------------------

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
                f"{total_class_objects} total"
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

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE FRAME
        # ----------------------------------------------------

        out.write(frame)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    cap.release()

    out.release()

    csv_output.close()

    total_time = (
        time.time()
        -
        start_time
    )

    average_fps = (
        frame_count / total_time
        if total_time > 0
        else 0
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("              FINAL SYSTEM COMPLETE")
    print("=" * 75)

    print("\nPERFORMANCE SUMMARY")
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
        f"{total_time:.2f} seconds"
    )

    print(
        f"Average FPS      : "
        f"{average_fps:.2f}"
    )

    print("\nOBJECT CLASS SUMMARY")
    print("-" * 50)

    for class_name in sorted(
        class_unique_ids
    ):

        print(
            f"{class_name:<15} : "
            f"{len(class_unique_ids[class_name])}"
        )

    print("\nDIRECTION SUMMARY")
    print("-" * 50)

    for direction in [
        "RIGHT",
        "LEFT",
        "UP",
        "DOWN",
        "STATIONARY"
    ]:

        print(
            f"{direction:<15} : "
            f"{len(direction_ids[direction])}"
        )

    print("\nGENERATED FILES")
    print("-" * 50)

    print(
        f"  [OK] {output_video}"
    )

    print(
        f"  [OK] {csv_file}"
    )

    print(
        "\nAll Project 3 tracking features "
        "successfully integrated!"
    )

    print("=" * 75)


if __name__ == "__main__":

    run_tracking(
        input_video="assets/test_video.mp4",
        model_path="models/yolov8n.pt",
        output_video="output/final_tracking_system.mp4",
        csv_file="output/final_tracking_events.csv"
    )