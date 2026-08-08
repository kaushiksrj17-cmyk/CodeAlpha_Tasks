"""
Utility functions for CodeAlpha Project 3.
"""

import math
import time


def calculate_direction(history, movement_threshold=4):
    """
    Calculate movement direction from object position history.
    """

    if len(history) < 2:
        return "STATIONARY", 0, 0

    old_x, old_y = history[0]
    new_x, new_y = history[-1]

    dx = new_x - old_x
    dy = new_y - old_y

    distance = math.sqrt(
        dx * dx + dy * dy
    )

    if distance < movement_threshold:
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


def calculate_center(x1, y1, x2, y2):
    """
    Calculate the center point of a bounding box.
    """

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y


def calculate_fps(frame_count, start_time):
    """
    Calculate average processing FPS.
    """

    elapsed = time.time() - start_time

    if elapsed <= 0:
        return 0.0

    return frame_count / elapsed


def calculate_average_dwell(dwell_times):
    """
    Calculate average dwell time.
    """

    if not dwell_times:
        return 0.0

    return sum(
        dwell_times.values()
    ) / len(dwell_times)


def draw_tracking_line(
    frame,
    line_y,
    width,
    color=(0, 255, 255)
):
    """
    Draw the entry/exit counting line.
    """

    import cv2

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        color,
        3
    )

    cv2.putText(
        frame,
        "ENTRY / EXIT LINE",
        (10, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_track_trail(
    frame,
    points,
    color=(255, 0, 255)
):
    """
    Draw the movement trail of an object.
    """

    import cv2

    for i in range(1, len(points)):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            2
        )


def draw_direction_arrow(
    frame,
    center_x,
    center_y,
    direction
):
    """
    Draw an arrow representing object movement.
    """

    import cv2

    arrow_length = 40

    if direction == "RIGHT":

        end_point = (
            center_x + arrow_length,
            center_y
        )

    elif direction == "LEFT":

        end_point = (
            center_x - arrow_length,
            center_y
        )

    elif direction == "DOWN":

        end_point = (
            center_x,
            center_y + arrow_length
        )

    elif direction == "UP":

        end_point = (
            center_x,
            center_y - arrow_length
        )

    else:
        return

    cv2.arrowedLine(
        frame,
        (center_x, center_y),
        end_point,
        (0, 255, 255),
        3,
        tipLength=0.3
    )