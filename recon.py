import bpy
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Ensure Tesseract is installed and set path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_blueprint(image_path):
    """Preprocess the blueprint image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)
    processed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return processed_img

def detect_walls(processed_img):
    """Detect walls using contour detection."""
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls = []
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        walls.append(approx)
    
    return walls

def create_3d_walls(walls, wall_height=3):
    """Convert 2D detected walls into 3D objects in Blender."""
    
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  # Clear previous objects

    for wall in walls:
        vertices = [(pt[0][0] / 100, pt[0][1] / 100, 0) for pt in wall]  # Scale to fit Blender
        faces = [(i, (i + 1) % len(vertices), (i + 1) % len(vertices), i) for i in range(len(vertices))]

        mesh = bpy.data.meshes.new("WallMesh")
        obj = bpy.data.objects.new("Wall", mesh)
        bpy.context.collection.objects.link(obj)
        
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Select & Extrude to create height
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, wall_height)})
        bpy.ops.object.mode_set(mode='OBJECT')

if __name__ == "__main__":
    image_path = "test.jpg"

    processed_img = preprocess_blueprint(image_path)
    walls = detect_walls(processed_img)

    create_3d_walls(walls)
    print("3D Floorplan generated in Blender")
