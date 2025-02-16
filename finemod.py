import cv2
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
import os
from inference_sdk import InferenceHTTPClient

# Convert SVG to PNG
def convert_svg_to_png(svg_path, png_path="converted.png"):
    """ Converts SVG to PNG and returns PNG path """
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    return png_path

# Initialize inference client (Ensure API key is correct)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2mJz8J9Jx7NWoco65LK3"  # Replace with your actual API key
)

# Set image path (Replace with your actual file)
image_path = "test.jpg"  # Replace with the actual file

# Validate image format
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File not found: {image_path}")

# If the file is SVG, convert it to PNG
if image_path.lower().endswith(".svg"):
    image_path = convert_svg_to_png(image_path)

# Load and resize image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image: {image_path}")

image = cv2.resize(image, (1024, 1024))  # Resize to 1024x1024
cv2.imwrite("resized.jpg", image)  # Save resized image

# Perform inference with confidence parameter in the model_id URL
result = CLIENT.infer("resized.jpg", model_id="builderformer-4/2?confidence=0")

# Print detection results
print(result)

# Extract detected objects from response
detections = []
if "predictions" in result:
    for pred in result["predictions"]:
        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        detections.append({
            "label": pred["class"],
            "confidence": int(pred["confidence"] * 100),  # Convert to percentage
            "box": (x1, y1, x2, y2)
        })

# Define unique colors for each label
label_colors = {}
for detection in detections:
    if detection["label"] not in label_colors:
        label_colors[detection["label"]] = tuple(np.random.randint(0, 255, 3).tolist())

# Draw bounding boxes and labels on the image
for detection in detections:
    x1, y1, x2, y2 = detection["box"]
    label = detection["label"]
    confidence = detection["confidence"]
    color = label_colors[label]

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    # Create label text
    text = f"{label} {confidence}%"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Add semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Put label text on image
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Display the improved image
plt.figure(figsize=(12, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
