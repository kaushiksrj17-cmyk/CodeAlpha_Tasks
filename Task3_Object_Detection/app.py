"""
CodeAlpha Project 3
AI Object Detection and Multi-Object Tracking

Main application entry point.
"""

import os
from tracker import run_tracking


def main():
    print("=" * 75)
    print("        CODEALPHA PROJECT 3")
    print("   AI OBJECT DETECTION & TRACKING SYSTEM")
    print("=" * 75)

    input_video = "assets/test_video.mp4"
    model_path = "models/yolov8n.pt"
    output_video = "output/final_tracking_system.mp4"
    csv_file = "output/final_tracking_events.csv"

    if not os.path.exists(input_video):
        print("\nERROR: Input video not found.")
        print(f"Expected: {input_video}")
        return

    if not os.path.exists(model_path):
        print("\nERROR: YOLOv8 model not found.")
        print(f"Expected: {model_path}")
        return

    print("\nStarting final tracking system...\n")

    run_tracking(
        input_video=input_video,
        model_path=model_path,
        output_video=output_video,
        csv_file=csv_file
    )


if __name__ == "__main__":
    main()