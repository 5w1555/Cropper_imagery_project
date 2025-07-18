import os
import tempfile
import shutil
import time

import gradio as gr
from config import get_preset_labels, key_for_label, get_preset_by_key
from cropper import (
    get_face_and_landmarks,
    auto_crop,
    head_bust_crop,
    apply_aspect_ratio_filter,
    apply_filter,
)

# Custom CSS for branding and styling
CUSTOM_CSS = """
/* Brand Colors */
:root {
    --brand-primary: #2563eb;
    --brand-secondary: #1e40af;
    --brand-accent: #3b82f6;
    --brand-success: #10b981;
    --brand-warning: #f59e0b;
    --brand-error: #ef4444;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
    padding: 2rem 1rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.2);
}

.header-container h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: bold;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.header-container p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
}

/* Button Styling */
.btn-primary {
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-accent)) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, var(--brand-success), #059669) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
}

.btn-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4) !important;
}

/* Progress Bar Styling */
.progress-container {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 4px solid var(--brand-primary);
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--brand-primary), var(--brand-accent));
    transition: width 0.3s ease;
}

/* Image Comparison Slider */
.comparison-container {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

/* Status Messages */
.status-success {
    color: var(--brand-success);
    font-weight: 600;
}

.status-error {
    color: var(--brand-error);
    font-weight: 600;
}

.status-warning {
    color: var(--brand-warning);
    font-weight: 600;
}

/* Card Styling */
.settings-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
}

/* Logo Area */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.logo-text {
    font-size: 3rem;
    background: linear-gradient(135deg, var(--brand-primary), var(--brand-accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
"""


def parse_ratio(r):
    """
    Convert a preset or UI value (str like "4:5", "None", or float/None)
    into a float aspect ratio or None.
    """
    if isinstance(r, str):
        if r.lower() == "none":
            return None
        if ":" in r:
            w, h = r.split(":", 1)
            try:
                return float(w) / float(h)
            except ValueError:
                return None
        try:
            return float(r)
        except ValueError:
            return None
    elif isinstance(r, (int, float)):
        return float(r)
    return None


from PIL import Image, ImageOps

def generate_preview(
    preset_label,
    input_files,
    margin,
    filter_name,
    intensity,
    aspect_ratio,
    rotate
):
    if not input_files:
        return None, None, "❌ No files uploaded."

    img_path = input_files[0].name

    # Load the raw “before” image (with correct EXIF orientation)
    try:
        before = ImageOps.exif_transpose(Image.open(img_path))
    except Exception as e:
        return None, None, f"❌ Failed to load original: {e}"

    # Now run your existing crop + aspect + filter pipeline
    key    = key_for_label(preset_label)
    method = "headbust" if key == "headbust" else "auto"
    ratio  = parse_ratio(aspect_ratio)

    if method == "headbust":
        cropped = head_bust_crop(
            input_path=img_path,
            margin=int(margin),
            target_ratio=ratio,
            conf_threshold=0.3
        )
    else:
        box, landmarks, cv_img, pil_img, metadata = get_face_and_landmarks(
            img_path, conf_threshold=0.3, apply_rotation=rotate
        )
        if box is None:
            return before, None, "❌ No face detected."
        cropped = auto_crop(
            pil_img,
            frontal_margin=int(margin),
            profile_margin=int(margin),
            box=box,
            landmarks=landmarks,
            metadata=metadata
        )
        if ratio and cropped:
            cropped = apply_aspect_ratio_filter(cropped, ratio)

    if cropped is None:
        return before, None, "❌ Crop failed."

    after = apply_filter(cropped, filter_name, intensity)
    return before, after, "✅ Preview generated."


def create_before_after_gallery(original_img, filtered_img):
    """Create a before/after gallery for comparison"""
    if original_img is None or filtered_img is None:
        return []
    
    # Save images temporarily to create gallery
    temp_dir = tempfile.mkdtemp(prefix="gallery_")
    
    original_path = os.path.join(temp_dir, "original.jpg")
    filtered_path = os.path.join(temp_dir, "filtered.jpg")
    
    original_img.save(original_path)
    filtered_img.save(filtered_path)
    
    return [original_path, filtered_path]


def process_images_with_progress(
    preset_label,
    input_files,
    margin,
    filter_name,
    intensity,
    aspect_ratio,
    rotate,
    progress=gr.Progress()
):
    if not input_files:
        return "❌ No files uploaded.", None, "0/0 images processed"

    key    = key_for_label(preset_label)
    method = "headbust" if key == "headbust" else "auto"
    ratio  = parse_ratio(aspect_ratio)

    tmp_dir = tempfile.mkdtemp(prefix="cropped_")
    count   = 0
    total_files = len(input_files)

    # Initialize progress
    progress(0, desc="Starting batch processing...")
    
    for i, f in enumerate(input_files):
        img = f.name
        
        # Update progress
        progress_pct = (i / total_files)
        progress(progress_pct, desc=f"Processing image {i+1}/{total_files}")

        if method == "headbust":
            bust = head_bust_crop(
                input_path=img,
                margin=int(margin),
                target_ratio=ratio,
                conf_threshold=0.3
            )
        else:
            box, landmarks, cv_img, pil_img, metadata = get_face_and_landmarks(
                img, conf_threshold=0.3, apply_rotation=rotate
            )
            if box is None:
                continue
            bust = auto_crop(
                pil_img,
                frontal_margin=int(margin),
                profile_margin=int(margin),
                box=box,
                landmarks=landmarks,
                metadata=metadata
            )
            if ratio and bust:
                bust = apply_aspect_ratio_filter(bust, ratio)

        if bust is None:
            continue

        out = apply_filter(bust, filter_name, intensity)
        name, ext = os.path.splitext(os.path.basename(img))
        out_path = os.path.join(tmp_dir, f"{name}_cropped{ext}")
        out.save(out_path)
        count += 1

    # Final progress update
    progress(1.0, desc="Creating download archive...")
    
    if count == 0:
        shutil.rmtree(tmp_dir)
        return "❌ No faces detected in any images.", None, f"0/{total_files} images processed"

    zip_path = shutil.make_archive(tmp_dir, 'zip', tmp_dir)
    return f"✅ Successfully processed {count} out of {total_files} images!", zip_path, f"{count}/{total_files} images processed"


# Create the Gradio interface
with gr.Blocks(title="🔪 One-Click Image Cropper", css=CUSTOM_CSS) as demo:
    
    # Header with branding
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-container">
                    <div class="logo-container">
                        <span class="logo-text">🔪</span>
                    </div>
                    <h1>One-Click Portrait Cropper</h1>
                    <p>Professional image cropping with AI-powered face detection and custom filters</p>
                </div>
            """)

    with gr.Row():
        # Left Column - Controls
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ **Crop Settings**")
                
                preset_dd = gr.Dropdown(
                    choices=get_preset_labels(),
                    value=(get_preset_labels()[0] if get_preset_labels() else None),
                    label="📋 Preset Configuration",
                    info="Choose a predefined cropping preset"
                )
                
                input_files = gr.File(
                    label="📁 Upload Images",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"]
                )
                
                margin = gr.Number(
                    value=30,
                    label="📏 Crop Margin (pixels)",
                    info="Distance from face edges to image border"
                )
                
                aspect_ratio = gr.Dropdown(
                    choices=[None, 1.0, 4/5, 1.91, 9/16, 3/2, 5/4, 16/9],
                    value=None,
                    label="📐 Aspect Ratio",
                    info="Force specific width:height ratio",
                    type="value"
                )
                
                rotate = gr.Checkbox(
                    label="🔄 Auto-Rotate Based on Face Detection", 
                    value=True,
                    info="Automatically rotate images for optimal face alignment"
                )

            with gr.Group():
                gr.Markdown("### 🎨 **Visual Effects**")
                
                filter_name = gr.Dropdown(
                    choices=["None", "Brightness", "Contrast", "Blur",
                             "Edge Detection", "Sepia"],
                    value="None",
                    label="🎭 Filter Type",
                    info="Apply visual effects to cropped images"
                )
                
                intensity = gr.Slider(
                    0, 100, 
                    value=50, 
                    label="⚡ Filter Intensity",
                    info="Adjust the strength of the applied filter"
                )

            # Action Buttons
            with gr.Row():
                preview_btn = gr.Button(
                    "🔍 Preview First Image", 
                    variant="primary",
                    elem_classes=["btn-primary"]
                )
                process_btn = gr.Button(
                    "🚀 Process All Images", 
                    variant="secondary",
                    elem_classes=["btn-secondary"]
                )

        # Right Column - Results
        with gr.Column(scale=1):
            gr.Markdown("### 📸 **Before & After Comparison**")
            
            # Before/After Gallery
            comparison_gallery = gr.Gallery(
                label="Drag to compare Before ↔ After",
                show_label=True,
                elem_id="comparison-gallery",
                columns=2,
                rows=1,
                object_fit="contain",
                height="400px",
                interactive=False
            )
            
            # Status and Progress
            with gr.Group():
                status = gr.Textbox(
                    label="📊 Processing Status", 
                    interactive=False,
                    info="Current operation status and results"
                )
                
                progress_info = gr.Textbox(
                    label="📈 Progress Info",
                    interactive=False,
                    value="Ready to process images...",
                    info="Detailed progress information"
                )
            
            # Download Section
            gr.Markdown("### 📥 **Download Results**")
            download_zip = gr.File(
                label="💾 Download Processed Images",
            )

    # Event Handlers
    def apply_preset(label):
        cfg = get_preset_by_key(key_for_label(label))
        return cfg.get("margin", 30), parse_ratio(cfg.get("target_ratio", None))

    def enhanced_preview(preset_label, input_files, margin,
                     filter_name, intensity, aspect_ratio, rotate):
        before_img, after_img, status_msg = generate_preview(
            preset_label, input_files, margin,
            filter_name, intensity, aspect_ratio, rotate
        )
        # Gallery can accept a list of PIL.Images
        gallery_items = [before_img, after_img] if (before_img and after_img) else []
        return gallery_items, status_msg

    preset_dd.change(
        fn=apply_preset,
        inputs=preset_dd,
        outputs=[margin, aspect_ratio]
    )

    preview_btn.click(
        fn=enhanced_preview,
        inputs=[
            preset_dd, input_files, margin,
            filter_name, intensity, aspect_ratio, rotate
        ],
        outputs=[comparison_gallery, status]
    )
    
    process_btn.click(
        fn=process_images_with_progress,
        inputs=[
            preset_dd, input_files, margin,
            filter_name, intensity, aspect_ratio, rotate
        ],
        outputs=[status, download_zip, progress_info]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)