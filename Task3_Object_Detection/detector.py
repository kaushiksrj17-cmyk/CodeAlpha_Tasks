from ultralytics import YOLO
import cv2
import os

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Input image
image_path = "assets/test.png"

# Check image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found: {image_path}")
    exit()

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

print("Running object detection...")

# Run YOLO detection
results = model(image)

# Draw detections
annotated_image = results[0].plot()

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Save result
output_path = "outputs/detected_image.jpg"
cv2.imwrite(output_path, annotated_image)

# Print detected objects
print("\nDetected Objects:")

for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    print(f"- {class_name}: {confidence * 100:.1f}%")

print("\nDetection complete!")
print(f"Result saved to: {output_path}")