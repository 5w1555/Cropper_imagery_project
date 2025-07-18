import os
from PIL import Image

def create_required_folders(base_path):
    folders = ['originals', 'face_detector', 'cropped']
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)

def read_image(image_path):
    pil_image = Image.open(image_path)
    cv_image = None  # Placeholder for OpenCV image if needed
    metadata = {"icc_profile": pil_image.info.get("icc_profile")}
    return cv_image, pil_image, metadata

def save_image(image, path, metadata=None, output_format="PNG"):
    image.save(path, format=output_format, **(metadata or {}))

def save_as_heic_fallback(image, path):
    try:
        image.save(path, format='HEIC')
    except Exception:
        save_image(image, path.replace('.heic', '.png'))  # Fallback to PNG if HEIC fails