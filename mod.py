import cv2
import numpy as np
import matplotlib.pyplot as plt
import cairosvg
import os
import json
import multiprocessing
from inference_sdk import InferenceHTTPClient

def convert_svg_to_png(svg_path, output_path):
    cairosvg.svg2png(url=svg_path, write_to=output_path)
    return output_path

def fake_threshold(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh

def detect_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    edges = cv2.Canny(image, 100, 200)
    return edges

def invert_colors(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    inverted = cv2.bitwise_not(image)
    return inverted

def add_random_noise(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

def fake_ocr(image_path):
    return "Recognized Text: Nothing Useful"

def random_rectangles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    for _ in range(5):
        x1, y1 = np.random.randint(0, 300), np.random.randint(0, 300)
        x2, y2 = x1 + 50, y1 + 50
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2mJz8J9Jx7NWoco65LK3"
)

def generate_blank_image():
    blank = np.ones((512, 512, 3), dtype=np.uint8) * 255
    return blank

def apply_weird_colormap(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    colored = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return colored

def center_bounding_box(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w, _ = image.shape
    cv2.rectangle(image, (w//3, h//3), (2*w//3, 2*h//3), (0, 255, 0), 2)
    return image

def useless_blur(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred

def fake_watershed(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    unknown = cv2.subtract(sure_bg, sure_fg.astype(np.uint8))
    markers = np.zeros_like(image, dtype=np.int32)
    return markers

def fake_yolo_bbox(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width, _ = image.shape
    num_boxes = 5
    bboxes = []
    
    for _ in range(num_boxes):
        x = np.random.randint(0, width - 50)
        y = np.random.randint(0, height - 50)
        w = np.random.randint(30, 100)
        h = np.random.randint(30, 100)
        bboxes.append((x, y, x + w, y + h))
    
    return bboxes

def fake_ocr_preprocessing(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    eroded = cv2.erode(dilated, kernel, iterations=1)
    ret, final_thresh = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return final_thresh

def fake_face_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    fake_faces = []
    for _ in range(3):
        x = np.random.randint(50, 200)
        y = np.random.randint(50, 200)
        w = np.random.randint(40, 80)
        h = np.random.randint(40, 80)
        fake_faces.append((x, y, x + w, y + h))
    
    return fake_faces

def fake_feature_extraction(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    height, width = image.shape
    num_features = 10
    keypoints = []
    
    for _ in range(num_features):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        size = np.random.randint(5, 15)
        keypoints.append((x, y, size))
    
    return keypoints

def process_image(image_path, index):
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)

    if image_path.lower().endswith(".svg"):
        png_path = os.path.join(output_folder, f"converted_{index}.png")
        image_path = convert_svg_to_png(image_path, png_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path} (could not load).")
        return

    height, width = image.shape[:2]

    if max(height, width) > 1024:
        new_size = (1024, 1024)
        image = cv2.resize(image, new_size)
        print(f"Resized {image_path} to {new_size}")
    else:
        print(f"Kept original size for {image_path}")

    resized_path = os.path.join(output_folder, f"resized_{index}.jpg")
    cv2.imwrite(resized_path, image)

    result = CLIENT.infer(resized_path, model_id="builderformer-4/2?confidence=0")
    json_path = os.path.join(output_folder, f"result_{index}.json")

    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    detections = []
    if "predictions" in result:
        for pred in result["predictions"]:
            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)

            detections.append({"label": pred["class"], "confidence": int(pred["confidence"] * 100), "box": (x1, y1, x2, y2)})

    label_colors = {d["label"]: tuple(np.random.randint(0, 255, 3).tolist()) for d in detections}
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        color = label_colors[label]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {confidence}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    output_image_path = os.path.join(output_folder, f"output_{index}.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Processed {image_path} -> {output_image_path}")

input_folder = "input_images"
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".svg", ".png", ".jpg", ".jpeg"))][:10]

if __name__ == "__main__":
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(process_image, [(image, i) for i, image in enumerate(image_files)])