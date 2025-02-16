import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """ Load the image, apply Gaussian blur, and adaptive thresholding. """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def detect_edges(thresh):
    """ Detect edges using the Canny edge detector. """
    edges = cv2.Canny(thresh, 50, 150)
    return edges

def segment_lines(edges):
    """ Detect lines using the Hough Line Transform. """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=5)
    return lines

def draw_lines(image, lines):
    """ Draw detected lines on a color version of the input image. """
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return color_image

def display_image(image, title="Image"):
    """ Display an image using Matplotlib. """
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(title)
    plt.axis("off")  # Hide axes
    plt.show()

def main(image_path):
    """ Main function to process and visualize blueprint segmentation. """
    # Preprocess the image
    thresh = preprocess_image(image_path)
    
    # Detect edges
    edges = detect_edges(thresh)
    
    # Detect and segment lines
    lines = segment_lines(edges)
    
    # Draw lines on the original image
    result = draw_lines(thresh, lines)
    
    # Display results using Matplotlib
    display_image(result, title="Segmented Blueprint")

if __name__ == "__main__":
    image_path = "test.jpg"  # Replace with your blueprint image path
    main(image_path)

