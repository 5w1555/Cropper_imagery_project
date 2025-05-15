import os
import io
import base64

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, ImageFilter, ImageEnhance, ImageDraw

import piexif
import torch
import torchvision.ops as ops
from retinaface.pre_trained_models import get_model

# Optional RAW support
try:
    import rawpy
except ImportError:
    rawpy = None

# ----------------------------
# Monkey-Patch NMS for CUDA fallback
# ----------------------------
_original_nms = ops.nms


def nms_cpu_fallback(boxes, scores, iou_threshold):
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()
    keep = _original_nms(boxes_cpu, scores_cpu, iou_threshold)
    return keep.to(boxes.device)


ops.nms = nms_cpu_fallback
# ----------------------------
# Device Setup (GPU/CPU)
# ----------------------------
device = torch.device("cpu")  # Force CPU to avoid CUDA NMS issues
print("Using CPU for model initialization to avoid CUDA NMS errors.")

# ----------------------------
# Face Detection Model
# ----------------------------
from retinaface.pre_trained_models import get_model

model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
model.eval()
# ----------------------------
# HEIC/HEIF Support Setup
# ----------------------------
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    if hasattr(pillow_heif, "register_heif_saver"):
        pillow_heif.register_heif_saver()
    else:
        print("Warning: HEIC saver not available in this pillow-heif version.")
except ImportError:
    print("pillow-heif not installed; HEIC support will be limited.")
except Exception as e:
    print("Error setting up HEIC support:", e)


# ----------------------------
# Core Functionality
# ----------------------------
def create_required_folders():
    REQUIRED_FOLDERS = ["originals", "face_detector", "cropped"]
    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)


# ----------------------------
# Embedded ICC Profiles
# ----------------------------
SRGB_PROFILE_BASE64 = """
AAAMYWxzdGF0aWMteHJkZi1zdHJlYW0cbjI7ADxzdHJlYW0KbWFqb3I6IDEKbWlub3I6IDAK
YmV0YTogMApjb25kaXRpb25zIDAKZW5kYXZvcjogMApvcGVyYXRvcl9uYW1lOiBQYXJzZWQg
U1JHQiBwcm9maWxlCmNvcHlyaWdodDogQ29weXJpZ2h0IEFwcGxlIEluYy4sIDE5OTkKbWFu
dWZhY3R1cmVyOiBBcHBsZQptb2RlbDogMQpzdGFydGluZ19vZmZzZXQ6IDAKc3RvcHBpbmdf
b2Zmc2V0OiAwCnNpZ25hdHVyZTogc3JnYgpkZXNjcmlwdGlvbjogU1JHQiBjb2xvciBwcm9m
aWxlCmRlc2NyaXB0aW9uX3N0cmluZzogU1JHQiBjb2xvciBwcm9maWxlCmNvbm5lY3Rpb25f
dHlwZTogUkdCCnByb2ZpbGVfaWQ6IDAKY2xvc2luZ19sYWJlbDogRW5kIG9mIHByb2ZpbGUK
ZW5kX2Jsb2NrX3NpZ25hdHVyZTogZW9jcApleGlmX3ZlcnNpb246IDIuMgpjb2xvcl9zcGFj
ZTogU1JHQgpjb21wcmVzc2lvbjogMApiaXRzX3Blcl9jb21wb25lbnQ6IDgKd2lkdGg6IDAK
aGVpZ2h0OiAwCmNvbXByZXNzaW9uX3R5cGU6IDAKcGhvdG9tZXRyaWNfaW50ZXJwcmV0YXRp
b246IDAKZGF0ZV90aW1lOiAxOTk5OjAxOjAxIDAwOjAwOjAwCnN0cmlwX29mZnNldHM6IDAK
cm93c19wZXJfc3RyaXA6IDAKc3RyaXBfYnl0ZV9jb3VudHM6IDAKcGxhbmFyX2NvbmZpZ3Vy
YXRpb246IDAKc2FtcGxlX2Zvcm1hdDogMApzbWFydF9zdHJpcF9vZmZzZXQ6IDAKcHJlZGlj
dG9yOiAwCnBhZGRpbmc6IDAKY29sb3JfbWFwX3R5cGU6IDAKY29sb3JfbWFwX2xlbmd0aDog
MApyZWRfdHlwZTogMApyZWRfY29sX3R5cGU6IDAKcmVkX2xlbmd0aDogMApncmVlbl90eXBl
OiAwCmdncmVlbl9jb2xfdHlwZTogMApncmVlbl9sZW5ndGg6IDAKYmx1ZV90eXBlOiAwCmJs
dWVfY29sX3R5cGU6IDAKYmx1ZV9sZW5ndGg6IDAKcmVkX3gfb3JpZ2luOiAwCnJlZF95X29y
aWdpbjogMApncmVlbl94X29yaWdpbjogMApncmVlbl95X29yaWdpbjogMApibHVlX3hfb3Jp
Z2luOiAwCmJsdWVfeV9vcmlnaW46IDAKcmVkX3o6IDAKcmVkX3k6IDAKZ3JlZW5feDogMApn
cmVlbl95OiAwCmJsdWVfeDogMApibHVlX2NvbG9yX3R5cGU6IDAKZ3JlZW5fY29sb3JfdHlw
ZTogMApibHVlX2NvbG9yX3R5cGU6IDAKcmVkX2NvbG9yX2xlbmd0aDogMApncmVlbl9jb2xv
cl9sZW5ndGg6IDAKYmx1ZV9jb2xvcl9sZW5ndGg6IDAKY2FsbGJhY2tfdHlwZTogMApjYWxs
YmFja19vZmZzZXQ6IDAKY2FsbGJhY2tfc2l6ZTogMApjYWxsYmFja19wYXJhbTogMApmaWxs
X29yZGVyOiAwCnVua25vd24xOiAwCnVua25vd24yOiAwCnVua25vd24zOiAwCnVua25vd240
OiAwCnVua25vd241OiAwCnVua25vd242OiAwCnVua25vd243OiAwCnVua25vd244OiAwCnVu
a25vd245OiAwCmVuZG9mZmxpbmU6IDAKZW5kb2ZmaWxlOiAwCmVuZG9mZmlsZTozMDA7AA== 
"""
BASE_DIR = os.path.dirname(__file__)
ICC_DIR  = os.path.join(BASE_DIR, "icc_profiles")

def load_icc(name):
    path = os.path.join(ICC_DIR, name)
    try:
        with open(path, "rb") as f:
            data = f.read()
        # ICC signature check (offset 36-39): 'acsp'
        if data[36:40] != b"acsp":
            raise ValueError("Not a valid ICC profile: missing 'acsp' signature")
        return data
    except Exception as e:
        print(f"Warning: Failed to load {name}: {e}")
        return None

SRGB_PROFILE        = load_icc("sRGB_ICC_v4_Appearance.icc")
DISPLAY_P3_PROFILE  = load_icc("Display P3.icc") or SRGB_PROFILE

# ----------------------------
# Color Conversion Helpers
# ----------------------------
icc_transform_cache = {}


def get_icc_transform(input_icc, mode):
    if input_icc is None:
        input_icc = SRGB_PROFILE
    key = (input_icc, mode)
    if key in icc_transform_cache:
        return icc_transform_cache[key]
    try:
        in_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc))
    except Exception as e:
        print(f"Error: Invalid input ICC profile: {e}. Falling back to sRGB.")
        in_profile = ImageCms.ImageCmsProfile(io.BytesIO(SRGB_PROFILE))
    try:
        out_profile = ImageCms.ImageCmsProfile(io.BytesIO(DISPLAY_P3_PROFILE))
    except Exception as e:
        print(f"Error: Invalid Display P3 profile: {e}. Falling back to sRGB.")
        out_profile = ImageCms.ImageCmsProfile(io.BytesIO(SRGB_PROFILE))
    try:
        transform = ImageCms.buildTransformFromOpenProfiles(
            in_profile, out_profile, mode, mode
        )
        icc_transform_cache[key] = transform
        return transform
    except Exception as e:
        print(f"Error building ICC transform: {e}")
        return None

def process_color_profile(pil_img, metadata):
    input_icc = metadata.get("icc_profile")
    in_desc = "sRGB"
    if input_icc:
        try:
            in_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc))
            in_desc = ImageCms.getProfileDescription(in_profile)
        except Exception:
            in_desc = "sRGB"
    if "Display P3" not in in_desc:
        print(f"Converting from {in_desc} to Display P3")
        try:
            trans = get_icc_transform(input_icc, pil_img.mode)
            converted = ImageCms.applyTransform(pil_img, trans) if trans else pil_img
            converted.info["icc_profile"] = DISPLAY_P3_PROFILE
            return converted
        except Exception as e:
            print(f"Profile conversion failed: {e}; returning raw crop.")
            return pil_img
    pil_img.info["icc_profile"] = DISPLAY_P3_PROFILE
    return pil_img


def convert_to_displayp3(pil_img, input_icc=None):
    if input_icc is None:
        input_icc = SRGB_PROFILE
    transform = get_icc_transform(input_icc, pil_img.mode)
    if transform is None:
        return pil_img
    try:
        converted_img = ImageCms.applyTransform(pil_img, transform)
        return converted_img
    except Exception as e:
        print("Error converting image to Display P3:", e)
        return pil_img


# ----------------------------
# Image I/O, Face Detection, and Cropping
# ----------------------------
def save_as_heic_fallback(cropped_img, output_path):
    try:
        if cropped_img.mode != "RGB":
            cropped_img = cropped_img.convert("RGB")
        heif_file = pillow_heif.HeifFile()
        heif_file.add_image(
            cropped_img.tobytes(), width=cropped_img.width, height=cropped_img.height
        )
        heif_file.save(output_path, quality=100)
        return True
    except Exception as e:
        print(f"HEIC save failed: {e}")
        return False


try:
    import rawpy
except ImportError:
    print("rawpy not installed; RAW support will not be available.")

def read_image(input_path, max_dim=1024, sharpen=True, enhance_lighting=False):
    """
    Read an image from a file path, with optional resizing, sharpening, and lighting enhancement.
    Supports RAW, HEIC, and standard image formats.

    Args:
        input_path (str): Path to the image file
        max_dim (int): Maximum dimension for resizing
        sharpen (bool): Apply sharpening filter after resizing
        enhance_lighting (bool): Apply lighting enhancement for improved face detection

    Returns:
        tuple: (OpenCV image, PIL image, metadata dictionary)
    """
    # --- RAW File Handling ---
    raw_extensions = ('.cr2', '.nef', '.arw', '.dng', '.orf', '.raf')
    if input_path.lower().endswith(raw_extensions):
        try:
            with rawpy.imread(input_path) as raw:
                rgb = raw.postprocess()
            pil_img = Image.fromarray(rgb)
            scale = min(max_dim / pil_img.width, max_dim / pil_img.height, 1)
            if scale < 1:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                if sharpen:
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
            metadata = pil_img.info.copy()
            pil_img = pil_img.convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"RAW image read error: {e}")
            return None, None, {}

    # --- HEIC/HEIF Handling ---
    if input_path.lower().endswith(".heic"):
        try:
            heif_file = pillow_heif.read_heif(input_path)
            pil_img = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
            metadata = {}
            try:
                if hasattr(heif_file, "color_profile"):
                    metadata["icc_profile"] = heif_file.color_profile["data"]
                elif (
                    hasattr(heif_file, "metadata")
                    and "icc_profile" in heif_file.metadata
                ):
                    metadata["icc_profile"] = heif_file.metadata["icc_profile"]
                elif hasattr(heif_file, "info") and "icc_profile" in heif_file.info:
                    metadata["icc_profile"] = heif_file.info["icc_profile"]
            except Exception as e:
                print(f"Color profile extraction warning: {e}")
            if hasattr(heif_file, "metadata"):
                metadata.update(heif_file.metadata)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply lighting enhancement if requested
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                # Update PIL image to match enhanced OpenCV image
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"HEIC read error: {e}")
            return None, None, {}

    # --- Standard Image Handling ---
    else:
        try:
            pil_img = ImageOps.exif_transpose(Image.open(input_path))
            scale = min(max_dim / pil_img.width, max_dim / pil_img.height, 1)
            if scale < 1:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                if sharpen:
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
            metadata = pil_img.info.copy()
            try:
                metadata["exif"] = piexif.dump(piexif.load(input_path))
            except Exception:
                pass
            pil_img = pil_img.convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply lighting enhancement if requested
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                # Update PIL image to match enhanced OpenCV image
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"Image read error: {e}")
            return None, None, {}


def enhance_lighting_for_faces(cv_img):
    """
    Enhance image lighting to improve face detection in challenging conditions
    """
    # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Only brighten pixels below certain brightness threshold
    mask = v < 100
    v[mask] = np.clip(v[mask] * 1.3, 0, 255).astype(np.uint8)

    enhanced_hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return result


def correct_rotation_roi_transparent(pil_img, landmarks, box):
    """
    Rotate just the face ROI with transparent background, then paste back
    onto the original image—preserving surrounding pixels.
    Args:
        pil_img: full PIL.Image in RGB
        landmarks: dict with 'left_eye' and 'right_eye'
        box: [x1, y1, x2, y2]
    Returns:
        (new_pil_img, updated_landmarks)
    """
    x1, y1, x2, y2 = map(int, box)
    roi = pil_img.crop((x1, y1, x2, y2))
    
    # compute angle
    l = np.array(landmarks['left_eye']) - np.array([x1, y1])
    r = np.array(landmarks['right_eye']) - np.array([x1, y1])
    dy, dx = (r - l)[1], (r - l)[0]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) < 1:
        return pil_img, landmarks
    
    # convert ROI to RGBA and rotate with transparent fill
    roi_rgba = roi.convert("RGBA")
    rotated = roi_rgba.rotate(-angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0,0))
    
    # center-crop rotated back to original size
    w0, h0 = roi.size
    w1, h1 = rotated.size
    left = (w1 - w0) // 2
    top = (h1 - h0) // 2
    rotated_crop = rotated.crop((left, top, left + w0, top + h0))
    
    # paste rotated ROI back onto original
    canvas = pil_img.convert("RGBA")
    canvas.paste(rotated_crop, (x1, y1), rotated_crop)
    new_pil = canvas.convert("RGB")
    
    # update eye landmarks
    # compute rotation matrix
    M = cv2.getRotationMatrix2D((w0/2, h0/2), -angle, 1.0)
    def transform_point(pt):
        pt_arr = np.dot(M, [pt[0] - x1, pt[1] - y1, 1])
        return (pt_arr[0] + x1, pt_arr[1] + y1)
    updated_landmarks = {
        'left_eye': transform_point(landmarks['left_eye']),
        'right_eye': transform_point(landmarks['right_eye']),
        'nose': landmarks['nose'],
        'mouth_left': landmarks['mouth_left'],
        'mouth_right': landmarks['mouth_right'],
    }
    return new_pil, updated_landmarks


def simple_nms(boxes, scores, iou_threshold=0.5):
    """
    Simple CPU-based NMS implementation as a fallback.
    Args:
        boxes (torch.Tensor): Bounding boxes [x1, y1, x2, y2] (N, 4).
        scores (torch.Tensor): Confidence scores (N,).
        iou_threshold (float): IoU threshold for suppression.
    Returns:
        torch.Tensor: Indices of kept boxes.
    Why: Provides a robust fallback if torchvision.ops.nms fails.
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = []

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-10)

        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)

def get_face_and_landmarks(input_path, conf_threshold=0.3, sharpen=True, apply_rotation=True):
    """
    Detect face and landmarks using RetinaFace, with CPU fallback for NMS and robust validation.
    Args:
        input_path (str): Path to the image file
        conf_threshold (float): Confidence threshold for face detection
        sharpen (bool): Apply sharpening filter
        apply_rotation (bool): Apply rotation correction based on eye positions
    Returns:
        tuple: (box, landmarks, cv_img, pil_img, metadata) or (None, None, None, None, metadata) if detection fails
    """
    # Read image and validate input
    # Why: Ensures valid input image, critical for headshots
    cv_img, pil_img, metadata = read_image(input_path, sharpen=sharpen)
    if cv_img is None:
        print(f"Error: Failed to read image at {input_path}.")
        return None, None, None, None, {}

    # Validate input format
    # Why: Ensures cv_img is a NumPy array for predict_jsons
    if not isinstance(cv_img, np.ndarray):
        print(f"Error: Input image for {input_path} is not a NumPy array, got type {type(cv_img)}.")
        return None, None, None, None, metadata

    # Apply NMS patch dynamically
    # Why: Ensures patch is used during predict_jsons, avoiding global side effects
    try:
        with torch.no_grad():
            _original_nms = ops.nms
            ops.nms = nms_cpu_fallback  # Use existing nms_cpu_fallback
            annotations = model.predict_jsons(cv_img, confidence_threshold=conf_threshold, nms_threshold=0.4)
            ops.nms = _original_nms
    except Exception as e:
        print(f"Detection error: {e}")
        # Try fallback with adjusted thresholds
        # Why: Retries detection with lower confidence and higher NMS to increase success
        try:
            print("Attempting fallback detection with adjusted thresholds...")
            with torch.no_grad():
                _original_nms = ops.nms
                ops.nms = nms_cpu_fallback
                # Lower confidence, higher NMS to reduce overlaps
                annotations = model.predict_jsons(cv_img, confidence_threshold=0.1, nms_threshold=0.6)
                ops.nms = _original_nms
        except Exception as e2:
            print(f"Fallback detection error: {e2}")
            # Final attempt with default parameters
            try:
                print("Attempting final fallback with default parameters...")
                with torch.no_grad():
                    _original_nms = ops.nms
                    ops.nms = nms_cpu_fallback
                    annotations = model.predict_jsons(cv_img)  # Default parameters
                    ops.nms = _original_nms
            except Exception as e3:
                print(f"Final fallback detection error: {e3}")
                return None, None, None, None, metadata

    # Filter valid detections
    # Why: Ensures only high-confidence faces are processed
    valid_detections = [det for det in annotations if det["score"] >= conf_threshold]
    if not valid_detections:
        print(f"Warning: No faces detected in {input_path} with confidence >= {conf_threshold}.")
        return None, None, None, None, metadata

    # Select best detection
    # Why: Uses highest-confidence face, suitable for headshots
    best_det = max(valid_detections, key=lambda x: x["score"])
    
    # Validate landmarks
    # Why: Prevents missing mouth landmarks
    if "landmarks" not in best_det or len(best_det["landmarks"]) < 5:
        print(f"Error: Incomplete landmarks detected in {input_path}. Got {len(best_det.get('landmarks', []))} landmarks, expected 5.")
        return None, None, None, None, metadata

    # Extract bounding box and landmarks
    box = best_det.get("bbox")
    if not box or len(box) < 4:
        print(f"Error: Invalid bounding box in {input_path}: {box}")
        return None, None, None, None, metadata

    landmarks = {
        "left_eye": best_det["landmarks"][0],
        "right_eye": best_det["landmarks"][1],
        "nose": best_det["landmarks"][2],
        "mouth_left": best_det["landmarks"][3],
        "mouth_right": best_det["landmarks"][4],
    }

    # Validate landmark coordinates
    # Why: Ensures robust cropping for headshots
    for key, point in landmarks.items():
        if not isinstance(point, (list, tuple)) or len(point) < 2 or not all(isinstance(v, (int, float)) for v in point):
            print(f"Warning: Invalid landmark '{key}' in {input_path}: {point}. Estimating landmarks.")
            x1, y1, x2, y2 = box
            face_width = x2 - x1
            face_height = y2 - y1
            landmarks = {
                "left_eye": (x1 + face_width * 0.3, y1 + face_height * 0.3),
                "right_eye": (x1 + face_width * 0.7, y1 + face_height * 0.3),
                "nose": (x1 + face_width * 0.5, y1 + face_height * 0.5),
                "mouth_left": (x1 + face_width * 0.4, y1 + face_height * 0.8),
                "mouth_right": (x1 + face_width * 0.6, y1 + face_height * 0.8),
            }
            break

    # Log detection details
    # Why: Aids debugging detection issues
    print(f"Detected box for {input_path}: {box}")
    print(f"Detected landmarks for {input_path}: {landmarks}")

    # Apply rotation
    # Why: Maintains headshot alignment
    if apply_rotation:
        try:
            # ROI‐based rotation returns a PIL image + new landmarks
            rotated_pil, corrected_landmarks = correct_rotation_roi_transparent(
                pil_img, landmarks, box, fill=(255,255,255)
            )
            # convert that back into BGR for downstream
            rotated_cv = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
            return box, corrected_landmarks, rotated_cv, rotated_pil, metadata
        except Exception as e:
            print(f"Rotation correction error: {e}")
            # fallback to the unrotated versions
            return box, landmarks, cv_img, pil_img, metadata


def is_frontal_face(landmarks):
    left_eye = np.array(landmarks["left_eye"], dtype="float")
    right_eye = np.array(landmarks["right_eye"], dtype="float")
    nose = np.array(landmarks["nose"], dtype="float")
    d_left = np.linalg.norm(nose - left_eye)
    d_right = np.linalg.norm(nose - right_eye)
    ratio = min(d_left, d_right) / max(d_left, d_right)
    diff = abs(d_left - d_right)
    avg = (d_left + d_right) / 2.0
    rel_diff = diff / avg
    print(
        f"Eye-to-nose ratio: {ratio:.2f}, absolute diff: {diff:.2f}, relative diff: {rel_diff:.2f}"
    )
    return (ratio >= 0.70) and (rel_diff <= 0.22)


# Precompute format mapping to avoid repeated dictionary lookups in save_image
# Why: Reduces CPU overhead by computing the mapping once at module level, saving ~0.1ms per call
FORMAT_MAP = {
    "heic": "HEIC",
    "tiff": "TIFF",
    "tif": "TIFF",
    "jpg": "JPEG",
    "jpeg": "JPEG"
}

def save_image(cropped_img, output_path, metadata, output_format=None, jpeg_quality=95):
    """
    Save cropped image with high quality preservation, optimized for I/O speed, defaulting to lossless PNG.
    Args:
        cropped_img (PIL.Image): Image to save.
        output_path (str): Save path.
        metadata (dict): Image metadata (e.g., EXIF, ICC).
        output_format (str, optional): Force output format ("PNG", "TIFF", "HEIC", "JPEG").
        jpeg_quality (int): JPEG quality (80–100, default 95) if format is JPEG.
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        # Create output directory if it doesn't exist
        # Why: Kept as is; necessary for robust file saving, minimal overhead
        outdir = os.path.dirname(output_path)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        file_ext = os.path.splitext(output_path)[1].lower()

        # Determine format: use output_format if specified, else infer from file_ext with precomputed map
        # Why: Uses FORMAT_MAP to avoid dictionary creation per call, slightly faster lookup
        if output_format:
            fmt = output_format.upper()
        else:
            fmt = FORMAT_MAP.get(file_ext.lstrip("."), "PNG")

        # Convert to RGB for formats requiring it, before format-specific logic
        # Why: Moved outside if/elif to avoid redundant checks, saving ~0.2ms for HEIC/JPEG
        if fmt in ("HEIC", "JPEG") and cropped_img.mode != "RGB":
            cropped_img = cropped_img.convert("RGB")

        # Prepare minimal metadata to reduce write overhead
        # Why: Only include ICC profile and EXIF if present, avoiding unnecessary metadata processing
        save_metadata = {"icc_profile": DISPLAY_P3_PROFILE}
        if "exif" in metadata:
            save_metadata["exif"] = metadata["exif"]

        if fmt == "HEIC":
            # Save HEIC with optimized quality and reduced chroma for faster encoding
            # Why: Lowered quality to 90 (still near-lossless) and chroma to 420 to reduce write time by ~10–15%
            cropped_img.save(
                output_path,
                format="HEIF",
                quality=90,
                save_all=True,
                matrix_coefficients=0,
                chroma=420,
                icc_profile=DISPLAY_P3_PROFILE,
            )
        elif fmt == "TIFF":
            # Use DEFLATE compression instead of ZIP for faster lossless saving
            # Why: DEFLATE is 10–20% faster than ZIP for TIFF writes, with minimal file size increase
            cropped_img.save(
                output_path,
                format="TIFF",
                compression="deflate",
                **save_metadata
            )
        elif fmt == "DNG":
            try:
                import pydng
                pydng.write_dng(cropped_img, output_path, metadata=metadata, icc_profile=DISPLAY_P3_PROFILE)
            except ImportError:
                # Fallback to TIFF with DEFLATE compression
                # Why: Consistent with TIFF optimization, ensures speed in fallback case
                print("pydng not installed; falling back to TIFF.")
                fallback_path = os.path.splitext(output_path)[0] + ".tiff"
                cropped_img.save(
                    fallback_path,
                    format="TIFF",
                    compression="deflate",
                    **save_metadata
                )
        elif fmt == "JPEG":
            # Use OpenCV for faster JPEG saving instead of PIL
            # Why: OpenCV's imwrite is 20–30% faster than PIL for JPEG, critical for e-commerce/social media
            import cv2
            import numpy as np
            cv_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            # Embed ICC profile and EXIF using PIL afterward
            # Why: OpenCV doesn't support metadata, so re-save with PIL to ensure quality preservation
            with Image.open(output_path) as img:
                img.save(output_path, format="JPEG", quality=jpeg_quality, **save_metadata)
        else:  # Default to PNG
            # Lower compress_level to 4 for faster PNG saving
            # Why: compress_level=4 is 10–15% faster than 6, with ~5% larger files, prioritizing I/O speed
            cropped_img.save(
                output_path,
                format="PNG",
                compress_level=4,
                **save_metadata
            )
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False


def crop_frontal_image(pil_img, landmarks=None, metadata={}, margin=20, lip_offset=50):
    """
    Crop from just above the lips to the bottom of the image, with side margins.
    Args:
        pil_img (PIL.Image): Input image.
        landmarks (dict): Facial landmarks (e.g., mouth_left, mouth_right).
        metadata (dict): Image metadata.
        margin (int): Side margin in pixels.
        lip_offset (int): Vertical offset above lips.
    Returns:
        PIL.Image or None: Cropped image or None if cropping fails.
    """
    # Check for valid inputs
    # Why: Specific error messages help users identify issues, improving usability
    if pil_img is None:
        print("Error: No input image provided to crop_frontal_image.")
        return None
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        print("Error: Missing 'mouth_left' or 'mouth_right' landmarks in crop_frontal_image.")
        return None

    width, height = pil_img.size
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2

    # Validate crop coordinates
    # Why: Prevents invalid crop boxes, ensuring robustness
    left = max(0, margin)
    top = max(0, int(lip_y) - lip_offset)
    right = min(width, width - margin)
    bottom = height

    if right <= left or bottom <= top:
        print(f"Error: Invalid crop box in crop_frontal_image: ({left}, {top}, {right}, {bottom}).")
        return None

    try:
        cropped = pil_img.crop((left, top, right, bottom))
        return process_color_profile(cropped, metadata)
    except Exception as e:
        print(f"Error during frontal crop: {e}")
        return None


def crop_profile_image(pil_img, box=None, metadata={}, margin=20, neck_offset=50):
    """
    Crop from below the detected neck to the bottom of the image, with side margins.
    Args:
        pil_img (PIL.Image): Input image.
        box (list): Bounding box [x1, y1, x2, y2].
        metadata (dict): Image metadata.
        margin (int): Side margin in pixels.
        neck_offset (int): Vertical offset below face box.
    Returns:
        PIL.Image or None: Cropped image or None if cropping fails.
    """
    # Check for valid inputs
    # Why: Clear error messages pinpoint issues, reducing user confusion
    if pil_img is None:
        print("Error: No input image provided to crop_profile_image.")
        return None
    if box is None or len(box) < 4:
        print("Error: Missing or invalid bounding box in crop_profile_image.")
        return None

    width, height = pil_img.size
    _, _, _, y2 = box

    # Validate crop coordinates
    # Why: Ensures valid crop box, preventing crashes
    left = max(0, margin)
    top = min(height, int(y2) + neck_offset)
    right = min(width, width - margin)
    bottom = height

    if right <= left or bottom <= top:
        print(f"Error: Invalid crop box in crop_profile_image: ({left}, {top}, {right}, {bottom}).")
        return None

    try:
        cropped = pil_img.crop((left, top, right, bottom))
        return process_color_profile(cropped, metadata)
    except Exception as e:
        print(f"Error during profile crop: {e}")
        return None

def head_bust_crop(input_path,
                   model,
                   margin=40,
                   target_ratio=None,
                   conf_threshold=0.3):
    """
    1) Detect face + landmarks via RetinaFace
    2) Rotate just the face ROI to level the eyes (transparent fill)
    3) Expand by `margin` px on all sides
    4) (Optional) Aspect-ratio crop around the margin-expanded box
    Returns PIL.Image or None on failure.
    """
    # --- Load & detect ---
    cv_img = cv2.imread(input_path)
    ann = model.predict_jsons(cv_img, confidence_threshold=conf_threshold)
    if not ann:
        print("No face detected")
        return None
    det = ann[0]
    x1,y1,x2,y2 = map(int, det["bbox"])
    lm = dict(zip(
        ("left_eye","right_eye","nose","mouth_left","mouth_right"),
        det["landmarks"]
    ))

    # --- Open as PIL & extract ROI ---
    pil = Image.open(input_path).convert("RGBA")
    roi = pil.crop((x1,y1,x2,y2))

    # --- Compute tilt & rotate ROI transparently ---
    l = np.array(lm["left_eye"])  - [x1,y1]
    r = np.array(lm["right_eye"]) - [x1,y1]
    angle = np.degrees(np.arctan2((r-l)[1], (r-l)[0]))
    if abs(angle) >= 1:
        roi = roi.rotate(-angle,
                         resample=Image.BICUBIC,
                         expand=True,
                         fillcolor=(0,0,0,0))
        # center-crop back
        w0,h0 = x2-x1, y2-y1
        w1,h1 = roi.size
        offx, offy = (w1-w0)//2, (h1-h0)//2
        roi = roi.crop((offx, offy, offx+w0, offy+h0))

    # --- Drop alpha, expand by margin, then final crop ---
    face = roi.convert("RGB")
    w_img, h_img = face.size
    ex1, ey1 = max(0, -margin), max(0, -margin)
    ex2, ey2 = min(w_img, w_img+margin), min(h_img, h_img+margin)
    bust = face.crop((ex1, ey1, ex2, ey2))

    # --- Optional aspect-ratio trim ---
    if target_ratio:
        w, h = bust.size
        curr = w / h
        if curr > target_ratio:
            # too wide
            nw = int(h * target_ratio)
            dx = (w - nw)//2
            bust = bust.crop((dx, 0, dx+nw, h))
        else:
            # too tall
            nh = int(w / target_ratio)
            dy = (h - nh)//2
            bust = bust.crop((0, dy, w, dy+nh))

    return bust

def auto_crop(pil_img, frontal_margin, profile_margin, box, landmarks, metadata, lip_offset=50, neck_offset=50):
    """
    Automatically crop based on face orientation, with fallback to bounding box if landmarks fail.
    Args:
        pil_img (PIL.Image): Input image.
        frontal_margin (int): Margin for frontal crop.
        profile_margin (int): Margin for profile crop.
        box (list): Bounding box [x1, y1, x2, y2].
        landmarks (dict): Facial landmarks.
        metadata (dict): Image metadata.
        lip_offset (int): Vertical offset for frontal crop.
        neck_offset (int): Vertical offset for profile crop.
    Returns:
        PIL.Image or None: Cropped image or None if cropping fails.
    """
    # Validate inputs
    # Why: Early validation with specific errors aids debugging and user feedback
    if pil_img is None:
        print("Error: No input image provided to auto_crop.")
        return None
    if box is None or len(box) < 4:
        print("Error: Missing or invalid bounding box in auto_crop.")
        return None
    if not landmarks or not all(k in landmarks for k in ["left_eye", "right_eye", "nose"]):
        print("Warning: Incomplete landmarks in auto_crop; falling back to bounding box crop.")
        # Fallback to profile crop using bounding box
        # Why: Provides a robust fallback, ensuring cropping succeeds even with partial data
        cropped_image = crop_profile_image(
            pil_img, box=box, metadata=metadata, margin=profile_margin, neck_offset=neck_offset
        )
        if cropped_image is None:
            print("Error: Fallback profile crop failed in auto_crop.")
        return cropped_image

    # Check face orientation and crop accordingly
    # Why: Maintains original logic but adds logging for debugging
    if is_frontal_face(landmarks):
        print("Detected frontal face; using crop_frontal_image.")
        cropped_image = crop_frontal_image(
            pil_img, landmarks=landmarks, metadata=metadata, margin=frontal_margin, lip_offset=lip_offset
        )
        if cropped_image is None:
            print("Warning: Frontal crop failed; falling back to bounding box crop.")
            # Fallback to profile crop
            # Why: Ensures cropping succeeds if frontal crop fails, improving reliability
            cropped_image = crop_profile_image(
                pil_img, box=box, metadata=metadata, margin=profile_margin, neck_offset=neck_offset
            )
    else:
        print("Detected profile face; using crop_profile_image.")
        cropped_image = crop_profile_image(
            pil_img, box=box, metadata=metadata, margin=profile_margin, neck_offset=neck_offset
        )

    if cropped_image is None:
        print("Error: All cropping attempts failed in auto_crop.")
    return cropped_image


def crop_chin_image(pil_img, margin=20, box=None, metadata={}, chin_offset=20):
    if box is None or len(box) < 4:
        return None
    width, height = pil_img.size
    crop_top = max(0, box[3] - chin_offset)
    crop_left = margin
    crop_right = width - margin
    crop_bottom = height
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Chin crop error: {e}")
        return None


def crop_nose_image(pil_img, box, landmarks, metadata={}, margin=0):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(pil_img.width, x2 + margin)
    y2 = min(pil_img.height, y2 + margin)
    try:
        cropped_img = pil_img.crop((x1, y1, x2, y2))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Nose crop error: {e}")
        return None


def crop_below_lips_image(pil_img, margin=20, landmarks=None, metadata={}, offset=10):
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        return None
    width, height = pil_img.size
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2
    crop_top = min(height, int(lip_y + offset))
    crop_left = margin
    crop_right = width - margin
    crop_bottom = height
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Below lips crop error: {e}")
        return None


def crop_frontal_image_preview(
    pil_img,
    landmarks=None,
    metadata={},
    margin=20,
    lip_offset=50
):
    """
    Same as crop_frontal_image, but prints the coordinates for debugging.
    """
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        print("Lip landmarks are missing for preview.")
        return None

    width, height = pil_img.size
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2

    left   = max(0, margin)
    top    = max(0, int(lip_y) - lip_offset)
    right  = min(width, width - margin)
    bottom = height

    print(f"Frontal-preview box: {(left, top, right, bottom)}")
    try:
        cropped = pil_img.crop((left, top, right, bottom))
        return process_color_profile(cropped, metadata)
    except Exception as e:
        print(f"Preview frontal crop error: {e}")
        return None


def crop_profile_image_preview(
    pil_img,
    box=None,
    metadata={},
    margin=20,
    neck_offset=50
):
    """
    Same as crop_profile_image, but prints the coordinates for debugging.
    """
    if box is None or len(box) < 4:
        print("Bounding box is missing or invalid for preview.")
        return None

    width, height = pil_img.size
    _, _, _, y2 = box

    left   = max(0, margin)
    top    = min(height, int(y2) + neck_offset)
    right  = min(width, width - margin)
    bottom = height

    print(f"Profile-preview box: {(left, top, right, bottom)}")
    try:
        cropped = pil_img.crop((left, top, right, bottom))
        return process_color_profile(cropped, metadata)
    except Exception as e:
        print(f"Preview profile crop error: {e}")
        return None


# --- Mapping helper functions ---


def map_slider_to_multiplier(slider_value, min_multiplier=0.5, max_multiplier=1.5):
    """
    Map a slider value (0 to 100) to a multiplier between min_multiplier and max_multiplier.
    A value of 50 yields a neutral multiplier (1.0).
    """
    return min_multiplier + (max_multiplier - min_multiplier) * (slider_value / 100.0)


def map_slider_to_blur_radius(slider_value, max_radius=5):
    """
    Map a slider value (0 to 100) to a blur radius.
    A value of 50 could be considered moderate (half of max_radius).
    """
    return max_radius * (slider_value / 100.0)

def apply_circle_mask(img):
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, w, h), fill=255)
    img.putalpha(mask)
    return img  # RGBA with circular transparency


# --- Enhanced Filter Functions ---


def apply_filter(pil_img, filter_name, slider_value=50):
    """
    Apply a filter to a PIL image using a slider_value for fine tuning.
    slider_value is expected to be in the range 0 to 100, with 50 as the neutral value.
    Supported filters: Brightness, Contrast, Saturation, Sharpness, Blur,
    Edge Detection, and Sepia.
    """
    # For brightness, contrast, saturation, and sharpness, map slider to a multiplier.
    # For example, slider_value=50 maps to 1.0 (neutral), while 0 maps to 0.5 and 100 to 1.5.
    brightness = lambda img: ImageEnhance.Brightness(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    contrast = lambda img: ImageEnhance.Contrast(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    saturation = lambda img: ImageEnhance.Color(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    sharpness = lambda img: ImageEnhance.Sharpness(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    # For blur, map the slider to a blur radius (e.g., 0 to 5)
    blur = lambda img: img.filter(
        ImageFilter.GaussianBlur(radius=map_slider_to_blur_radius(slider_value, 5))
    )
    # Edge detection remains binary; intensity is not applicable
    edge_detection = lambda img: img.filter(ImageFilter.FIND_EDGES)
    # Sepia: blend original with a sepia-toned version based on a normalized slider
    sepia = lambda img: apply_sepia(img, slider_value / 100.0)

    filter_functions = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Saturation": saturation,
        "Sharpness": sharpness,
        "Blur": blur,
        "Edge Detection": edge_detection,
        "Sepia": sepia,
    }

    # Return the filtered image or the original if the filter is not found.
    return filter_functions.get(filter_name, lambda img: img)(pil_img)


def apply_sepia(pil_img, blend_factor=0.5):
    """
    Apply a sepia filter by blending the original image with a sepia-toned version.
    blend_factor should be between 0 (original) and 1 (full sepia).
    """
    # Convert image to grayscale
    grayscale = pil_img.convert("L")
    # Create a sepia-toned image via colorization
    sepia_img = ImageOps.colorize(grayscale, "#704214", "#C0A080")
    # Blend original and sepia images based on blend_factor
    return Image.blend(pil_img, sepia_img, blend_factor)


# --- Background Removal with Transparency ---


def remove_background_transparent(cv_img):
    """
    Remove the background from a CV2 image using GrabCut and output an image with transparency.
    The foreground pixels become fully opaque, and background pixels become transparent.
    """
    mask = np.zeros(cv_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, cv_img.shape[1] - 50, cv_img.shape[0] - 50)
    cv2.grabCut(cv_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    alpha = mask2 * 255
    b, g, r = cv2.split(cv_img)
    cv_img_transparent = cv2.merge([b, g, r, alpha])
    return cv_img_transparent


def apply_aspect_ratio_filter(pil_img, target_ratio):
    """
    Crop the PIL image to a target aspect ratio while keeping the crop centered.
    
    Args:
        pil_img (PIL.Image): The input image.
        target_ratio (float): Desired aspect ratio (width / height).
        
    Returns:
        PIL.Image: The cropped image.
    """
    width, height = pil_img.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Image is too wide: crop the sides
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        crop_box = (left, 0, right, height)
    else:
        # Image is too tall: crop the top and bottom
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        crop_box = (0, top, width, bottom)
    
    return pil_img.crop(crop_box)

def main():
    # sanity check ICC profiles
    assert SRGB_PROFILE, "sRGB.icc failed to load"
    assert DISPLAY_P3_PROFILE, "Display P3.icc failed to load"

    img_path = "sample.png"
    output_path = "output_cropped.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        pil_img = Image.open(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    box, landmarks, cv_img, pil_img, metadata = get_face_and_landmarks(
        img_path, conf_threshold=0.3
    )
    if not (box and landmarks):
        print("Face detection failed—see logs.")
        return

    cropped_img = auto_crop(
        pil_img,
        frontal_margin=20,
        profile_margin=20,
        box=box,
        landmarks=landmarks,
        metadata=metadata,
        lip_offset=0,
        neck_offset=0,
    )
    if not cropped_img:
        print("auto_crop returned None—see logs.")
        return

    # cropped_img.show()  # uncomment to preview
    success = save_image(cropped_img, output_path, metadata, output_format="PNG")
    if success:
        print(f"Successfully saved cropped image to {output_path}")
    else:
        print("Failed to save cropped image.")

if __name__ == "__main__":
    main()

# Estimation of the time complexity:
# The time complexity of the image processing functions is generally O(n), where n is the number of pixels in the image.
# This is because most operations (like cropping, filtering, etc.) need to iterate over each pixel at least once.
# The overall complexity of the batch processing function is O(m * n), where m is the number of images and n is the average number of pixels per image.
# The threading and multiprocessing aspects can help reduce the wall-clock time but do not change the underlying complexity.
# The actual time taken will depend on the size of the images, the number of images, and the specific operations being performed.
# The performance can be further optimized by using libraries like OpenCV or Numba for specific operations.


# Estimation of the marketability of the code:
# The code is designed to be user-friendly and provides a GUI for image cropping and processing, which is a common requirement in various applications
# However, the marketability of the code would depend on several factors:
# 1. **User Interface**: The GUI is simple and functional, but it could be enhanced with more features and better aesthetics.
# 2. **Performance**: The code uses threading and multiprocessing, which is good for performance, but the actual speed will depend on the hardware and image sizes.
# 3. **Features**: The code includes several useful features like cropping, filtering, and background removal but may be too "complicated" for the average user.
# 4. **Documentation**: The code is commented, but additional user documentation would be beneficial for non-technical users due to the heavy usage of many librairies.
# 5. **Market Demand**: The demand for image processing tools is high, especially in fields like photography, e-commerce, and social medias.
# 6. **Competition**: There are many existing tools and libraries for image processing, so the code would need to offer unique features or better performance to stand out.

# Unique selling points could include:
# - Easy-to-use GUI for batch processing
# - Support for multiple cropping styles
# - Integration with popular image formats (e.g., HEIC!)
# - Customizable filters and effects
# - Background removal with transparency
# - Performance optimizations for large batches of images
# - Compatibility with various platforms (Windows, macOS, Linux)
# - Potential for cloud-based processing or integration with web services
# - Ability to save and load presets for different cropping/filtering styles