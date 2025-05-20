import os
import tempfile
import shutil

import gradio as gr
from config import get_preset_labels, key_for_label, get_preset_by_key

# Core image‐handling and crop routines
from cropper import (
    get_face_and_landmarks,
    auto_crop,
    apply_aspect_ratio_filter,
    apply_filter,
)

def generate_preview(
    input_files,  # <-- this will be a list
    margin, filter_name, intensity, aspect_ratio
):
    # 1️⃣ Guard against “no files uploaded”
    if not input_files:
        return None, None, "❌ No files uploaded."

    # 2️⃣ Pick the first file from the list
    file_obj = input_files[0]
    img_path = file_obj.name

    try:
        result = get_face_and_landmarks(img_path, sharpen=False, apply_rotation=True)
        if not result or not result[0] or not result[1]:
            return None, None, "❌ No face detected."

        box, landmarks, cv_img, pil_img, metadata = result

        cropped = auto_crop(pil_img, margin, margin, box, landmarks, metadata)
        if aspect_ratio:
            cropped = apply_aspect_ratio_filter(cropped, aspect_ratio)
        cropped = apply_filter(cropped, filter_name, intensity)

        return pil_img, cropped, "✅ Preview generated."
    except Exception as e:
        return None, None, f"❌ Error during preview: {e}"

def process_images(
    input_files, margin, filter_name, intensity, aspect_ratio
):
    """
    Takes multiple uploaded files, auto‐crops each, writes them to a temp
    folder, zips that folder, and returns a status + download link.
    """
    try:
        if not input_files:
            return "❌ No files uploaded.", None

        # Create a temporary directory
        tmp_dir = tempfile.mkdtemp(prefix="cropped_")
        for file in input_files:
            # run the same pipeline per image
            result = get_face_and_landmarks(
                file.name,
                sharpen=False,
                apply_rotation=True
            )
            if not result or not result[0] or not result[1]:
                continue  # skip non‐faces

            box, landmarks, cv_img, pil_img, metadata = result
            cropped = auto_crop(pil_img, margin, margin, box, landmarks, metadata)
            if aspect_ratio:
                cropped = apply_aspect_ratio_filter(cropped, aspect_ratio)
            cropped = apply_filter(cropped, filter_name, intensity)

            # Save out
            base = os.path.basename(file.name)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(tmp_dir, f"{name}_cropped{ext}")
            cropped.save(out_path)

        # Zip the folder
        zip_path = shutil.make_archive(tmp_dir, 'zip', tmp_dir)
        return f"✅ Processed {len(os.listdir(tmp_dir))} images.", zip_path
    except Exception as e:
        return f"❌ Error: {str(e)}", None

with gr.Blocks(title="🔪 One-Click Image Cropper") as demo:
    # ——— Header —————————————
    gr.Markdown(
        """
        ## 📸 One-Click Portrait Cropper
        Upload one or more images, and get perfectly centered, social-media ready portraits instantly.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # ——— Preset selector ——————
            preset_dd = gr.Dropdown(
                choices=get_preset_labels(),
                value=(get_preset_labels()[0] if get_preset_labels() else None),
                label="Preset"
            )

            # ——— File uploader ——————
            input_files = gr.File(
                label="Upload Images",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png"]
            )

            # ——— Parameters ——————
            margin = gr.Number(
                value=30, label="Crop Margin (px)",
                info="Padding around the detected face"
            )
            filter_name = gr.Dropdown(
                choices=["None", "Brightness", "Contrast", "Blur", "Edge Detection"],
                value="None", label="Filter"
            )
            intensity = gr.Slider(
                minimum=0, maximum=100, value=50, label="Filter Intensity"
            )
            aspect_ratio = gr.Dropdown(
                choices={"None": None, "4:3": 4/3, "3:2": 3/2, "16:9": 16/9},
                value=None, label="Aspect Ratio"
            )

            # ——— Action buttons ——————
            preview_btn  = gr.Button("🔍 Preview First Image")
            process_btn  = gr.Button("🚀 Process & Download All")

        with gr.Column(scale=1):
            # ——— Outputs ——————
            original_img = gr.Image(label="Original", interactive=False)
            preview_img  = gr.Image(label="Cropped Preview", interactive=False)
            status       = gr.Textbox(label="Status", interactive=False)
            download_zip = gr.File(label="Download Results")

    # ——— Bind Preset → margin & aspect_ratio ——————
    def apply_preset(label):
        key = key_for_label(label)
        cfg = get_preset_by_key(key)
        m = cfg.get("margin", 30)
        return m, cfg.get("target_ratio", None)

    preset_dd.change(
        fn=apply_preset,
        inputs=[preset_dd],
        outputs=[margin, aspect_ratio]
    )

    # ——— Preview event ——————
    preview_btn.click(
        fn=generate_preview,
        inputs=[input_files, margin, filter_name, intensity, aspect_ratio],
        outputs=[original_img, preview_img, status]
    )

    # ——— Batch process event ——————
    process_btn.click(
        fn=process_images,
        inputs=[input_files, margin, filter_name, intensity, aspect_ratio],
        outputs=[status, download_zip]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
