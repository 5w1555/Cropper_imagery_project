import pytest
from unittest import mock
from PIL import Image
import numpy as np
import os
import torch

# test__test_cropper.py

from cropper import (
    create_required_folders,
    load_icc,
    get_icc_transform,
    process_color_profile,
    convert_to_displayp3,
    save_as_heic_fallback,
    read_image,
    enhance_lighting_for_faces,
    correct_rotation_roi_transparent,
    simple_nms,
    get_face_and_landmarks,
    is_frontal_face,
    save_image,
    crop_frontal_image,
    crop_profile_image,
    head_bust_crop,
    auto_crop,
    crop_chin_image,
    crop_nose_image,
    crop_below_lips_image,
    crop_frontal_image_preview,
    crop_profile_image_preview,
    map_slider_to_multiplier,
    map_slider_to_blur_radius,
    apply_circle_mask,
    apply_filter,
    apply_sepia,
    remove_background_transparent,
    apply_aspect_ratio_filter,
)

@pytest.fixture
def dummy_pil_img():
    return Image.new("RGB", (100, 200), color="white")

@pytest.fixture
def dummy_landmarks():
    return {
        "left_eye": (30, 60),
        "right_eye": (70, 60),
        "nose": (50, 100),
        "mouth_left": (35, 150),
        "mouth_right": (65, 150),
    }

@pytest.fixture
def dummy_box():
    return [20, 40, 80, 120]

@pytest.fixture
def dummy_metadata():
    return {"icc_profile": None}

def test_create_required_folders(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(os, "makedirs", lambda p, exist_ok: created.append(p))
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    create_required_folders()
    assert any("originals" in f or "face_detector" in f or "cropped" in f for f in created)

def test_load_icc_handles_missing(monkeypatch):
    monkeypatch.setattr("os.path.join", lambda *a: "notfound.icc")
    with pytest.raises(Exception):
        load_icc("notfound.icc")

def test_get_icc_transform_cache(dummy_pil_img):
    # Should return None or a transform object; test cache miss/hit
    assert get_icc_transform(None, "RGB") is None

def test_process_color_profile(dummy_pil_img, dummy_metadata):
    out = process_color_profile(dummy_pil_img, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_convert_to_displayp3(dummy_pil_img):
    out = convert_to_displayp3(dummy_pil_img)
    assert isinstance(out, Image.Image)

def test_save_as_heic_fallback(dummy_pil_img, tmp_path):
    path = tmp_path / "test.heic"
    # Should not raise
    save_as_heic_fallback(dummy_pil_img, str(path))

def test_read_image_jpg(tmp_path, monkeypatch):
    path = tmp_path / "test.jpg"
    Image.new("RGB", (10, 10)).save(path)
    cv_img, pil_img, meta = read_image(str(path))
    assert pil_img.size == (10, 10)
    assert isinstance(meta, dict)

def test_enhance_lighting_for_faces(dummy_pil_img):
    arr = np.array(dummy_pil_img)
    arr = arr[..., ::-1]  # RGB to BGR
    out = enhance_lighting_for_faces(arr)
    assert out.shape == arr.shape

def test_correct_rotation_roi_transparent(dummy_pil_img, dummy_landmarks, dummy_box):
    # Should return rotated image and updated landmarks
    with mock.patch("cv2.getRotationMatrix2D", return_value=np.eye(2, 3)):
        out = correct_rotation_roi_transparent(dummy_pil_img, dummy_landmarks, dummy_box)
        assert isinstance(out, tuple)

def test_simple_nms_empty():
    boxes = torch.empty((0, 4))
    scores = torch.empty((0,))
    out = simple_nms(boxes, scores)
    assert out.numel() == 0

def test_is_frontal_face(dummy_landmarks):
    assert is_frontal_face(dummy_landmarks) is True

def test_save_image(dummy_pil_img, tmp_path, dummy_metadata):
    path = tmp_path / "test.png"
    assert save_image(dummy_pil_img, str(path), dummy_metadata, output_format="PNG")

def test_crop_frontal_image(dummy_pil_img, dummy_landmarks, dummy_metadata):
    out = crop_frontal_image(dummy_pil_img, dummy_landmarks, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_crop_profile_image(dummy_pil_img, dummy_box, dummy_metadata):
    out = crop_profile_image(dummy_pil_img, dummy_box, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_head_bust_crop(monkeypatch, tmp_path):
    # Patch get_face_and_landmarks to return dummy values
    monkeypatch.setattr("cropper.get_face_and_landmarks", lambda *a, **k: ([10, 10, 90, 90], {
        "left_eye": (30, 30), "right_eye": (70, 30), "nose": (50, 50), "mouth_left": (35, 80), "mouth_right": (65, 80)
    }, None, Image.new("RGB", (100, 100)), {}))
    out = head_bust_crop("dummy_path.jpg")
    assert isinstance(out, Image.Image)

def test_auto_crop(dummy_pil_img, dummy_box, dummy_landmarks, dummy_metadata):
    out = auto_crop(dummy_pil_img, 10, 10, dummy_box, dummy_landmarks, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_crop_chin_image(dummy_pil_img, dummy_box, dummy_metadata):
    out = crop_chin_image(dummy_pil_img, 10, dummy_box, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_crop_nose_image(dummy_pil_img, dummy_box, dummy_landmarks):
    out = crop_nose_image(dummy_pil_img, dummy_box, dummy_landmarks)
    assert isinstance(out, Image.Image)

def test_crop_below_lips_image(dummy_pil_img, dummy_landmarks, dummy_metadata):
    out = crop_below_lips_image(dummy_pil_img, 10, dummy_landmarks, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_crop_frontal_image_preview(dummy_pil_img, dummy_landmarks, dummy_metadata):
    out = crop_frontal_image_preview(dummy_pil_img, dummy_landmarks, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_crop_profile_image_preview(dummy_pil_img, dummy_box, dummy_metadata):
    out = crop_profile_image_preview(dummy_pil_img, dummy_box, dummy_metadata)
    assert isinstance(out, Image.Image)

def test_map_slider_to_multiplier():
    assert map_slider_to_multiplier(0) == 0.5
    assert map_slider_to_multiplier(100) == 1.5
    assert map_slider_to_multiplier(50) == 1.0

def test_map_slider_to_blur_radius():
    assert map_slider_to_blur_radius(0) == 0
    assert map_slider_to_blur_radius(100) == 5
    assert map_slider_to_blur_radius(50) == 2.5

def test_apply_circle_mask(dummy_pil_img):
    out = apply_circle_mask(dummy_pil_img.copy())
    assert out.mode == "RGBA"

def test_apply_filter(dummy_pil_img):
    out = apply_filter(dummy_pil_img, "Brightness", 80)
    assert isinstance(out, Image.Image)

def test_apply_sepia(dummy_pil_img):
    out = apply_sepia(dummy_pil_img, 0.8)
    assert isinstance(out, Image.Image)

def test_remove_background_transparent(dummy_pil_img):
    arr = np.array(dummy_pil_img)
    out = remove_background_transparent(arr)
    assert isinstance(out, np.ndarray)

def test_apply_aspect_ratio_filter(dummy_pil_img):
    out = apply_aspect_ratio_filter(dummy_pil_img, 1.5)
    assert isinstance(out, Image.Image)