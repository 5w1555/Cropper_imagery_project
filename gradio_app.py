# gradio_app.py

import os
import gradio as gr

# Core image‐handling and crop routines
from Cropper_project.cropper import (
    get_face_and_landmarks,
    auto_crop,
    crop_frontal_image_preview,
    crop_profile_image_preview,
    crop_chin_image,
    crop_nose_image,
    crop_below_lips_image,
    apply_aspect_ratio_filter,
    apply_filter,
)

# Batch‐processing orchestration
from processing import process_images_threaded


def generate_preview(
    input_folder, frontal_margin, profile_margin,
    sharpen, use_frontal, use_profile, correct_rotation,
    crop_style, filter_name, intensity, aspect_ratio
):
    valid_exts = (".jpg", ".jpeg", ".png", ".heic")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    if not files:
        return None

    file_path = os.path.join(input_folder, files[0])
    result = get_face_and_landmarks(
        file_path, sharpen=sharpen, apply_rotation=correct_rotation
    )
    if not result or not result[0] or not result[1]:
        return None

    box, landmarks, cv_img, pil_img, metadata = result

    funcs = {
        "auto": lambda: auto_crop(pil_img, frontal_margin, profile_margin, box, landmarks, metadata),
        "frontal": lambda: crop_frontal_image_preview(pil_img, box, landmarks, metadata, margin=frontal_margin, lip_offset=50) if use_frontal else None,
        "profile": lambda: crop_profile_image_preview(pil_img, box, metadata, margin=profile_margin, neck_offset=50) if use_profile else None,
        "chin": lambda: crop_chin_image(pil_img, margin=frontal_margin, box=box, metadata=metadata, chin_offset=20),
        "nose": lambda: crop_nose_image(pil_img, box, landmarks, metadata, margin=0),
        "below_lips": lambda: crop_below_lips_image(pil_img, margin=frontal_margin, landmarks=landmarks, metadata=metadata, offset=10),
    }

    cropped = funcs.get(crop_style, lambda: None)()
    if cropped is None:
        return None

    # Enforce aspect ratio and filter
    if aspect_ratio:
        cropped = apply_aspect_ratio_filter(cropped, aspect_ratio)
    cropped = apply_filter(cropped, filter_name, intensity)
    return cropped


def process_images(
    input_folder, output_folder,
    frontal_margin, profile_margin,
    sharpen, use_frontal, use_profile, correct_rotation,
    crop_style, filter_name, intensity, aspect_ratio,
    progress=gr.Progress()
):
    def update_progress(current, total, message):
        progress((current / total), f"Processed {current}/{total}: {message}")

    processed, total = process_images_threaded(
        input_folder=input_folder,
        output_folder=output_folder,
        frontal_margin=frontal_margin,
        profile_margin=profile_margin,
        sharpen=sharpen,
        use_frontal=use_frontal,
        use_profile=use_profile,
        progress_callback=update_progress,
        cancel_func=lambda: False,
        apply_rotation=correct_rotation,
        crop_style=crop_style,
        filter_name=filter_name,
        filter_intensity=intensity,
        aspect_ratio=aspect_ratio
    )
    return f"Completed: {processed}/{total} images"


with gr.Blocks(title="Image Cropper") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            input_folder   = gr.Textbox(label="Input Folder", placeholder="Path to input folder")
            output_folder  = gr.Textbox(label="Output Folder", placeholder="Path to output folder")
            load_preview   = gr.Button("Load Preview")
            start_process  = gr.Button("Start Processing")
            cancel_process = gr.Button("Cancel")

            frontal_margin = gr.Number(20, label="Frontal Margin (px)")
            profile_margin = gr.Number(20, label="Profile Margin (px)")
            sharpen_cb     = gr.Checkbox(True, label="Sharpen Image")
            frontal_cb     = gr.Checkbox(True, label="Use Frontal Cropping")
            profile_cb     = gr.Checkbox(True, label="Use Profile Cropping")
            rotation_cb    = gr.Checkbox(True, label="Correct Face Rotation")

            crop_style     = gr.Dropdown(
                                ["auto","frontal","profile","chin","nose","below_lips"],
                                value="auto", label="Crop Style"
                              )
            filter_name    = gr.Dropdown(
                                ["None","Brightness","Contrast","Blur","Edge Detection"],
                                value="None", label="Filter"
                              )
            intensity      = gr.Slider(0, 100, value=50, label="Intensity")
            aspect_ratio   = gr.Dropdown(
                                {"3:2":3/2, "4:3":4/3, "16:9":16/9},
                                value=None, label="Aspect Ratio"
                             )

        with gr.Column(scale=1):
            preview_img = gr.Image(label="Preview", interactive=False)
            status      = gr.Textbox(label="Status", interactive=False)
            progress_ui = gr.Progress()

    # Wire up events
    load_preview.click(
        fn=generate_preview,
        inputs=[
            input_folder, frontal_margin, profile_margin,
            sharpen_cb, frontal_cb, profile_cb, rotation_cb,
            crop_style, filter_name, intensity, aspect_ratio
        ],
        outputs=preview_img
    )

    start_process.click(
        fn=process_images,
        inputs=[
            input_folder, output_folder,
            frontal_margin, profile_margin,
            sharpen_cb, frontal_cb, profile_cb, rotation_cb,
            crop_style, filter_name, intensity, aspect_ratio
        ],
        outputs=status
    )

    cancel_process.click(lambda: "Processing cancelled.", outputs=status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
# Note: The above code assumes that the necessary functions and modules are defined in the same directory.
