import os
import concurrent.futures
import multiprocessing


from Cropper_project.cropper import (
    get_face_and_landmarks,
    is_frontal_face,
    crop_frontal_image,
    crop_profile_image,
    crop_chin_image,
    crop_nose_image,
    crop_below_lips_image,
    auto_crop,
    apply_aspect_ratio_filter,
    apply_filter,
    save_image,
)


def process_batch(
    batch_filenames,
    input_folder,
    output_folder,
    frontal_margin,
    profile_margin,
    sharpen=True,
    use_frontal=True,
    use_profile=True,
    apply_rotation=True,
    crop_style="frontal",
    filter_name="None",
    filter_intensity=50,
    aspect_ratio=None  # New parameter for aspect ratio (float, e.g., 16/9)
):
    count = 0
    for filename in batch_filenames:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        try:
            result = get_face_and_landmarks(
                input_path, sharpen=sharpen, apply_rotation=apply_rotation
            )
            if result is None or result[0] is None or result[1] is None:
                print(f"{filename}: No face detected. Skipping...")
            else:
                box, landmarks, _, pil_img, metadata = result
                crop_functions = {
                    "frontal": lambda: (
                        crop_frontal_image(
                            pil_img, frontal_margin, landmarks, metadata, lip_offset=50
                        )
                        if use_frontal and is_frontal_face(landmarks)
                        else auto_crop(
                            pil_img,
                            frontal_margin,
                            profile_margin,
                            box,
                            landmarks,
                            metadata,
                            lip_offset=50,
                            neck_offset=50,
                        )
                    ),
                    "profile": lambda: (
                        crop_profile_image(pil_img, profile_margin, 50, box, metadata)
                        if use_profile
                        else None
                    ),
                    "chin": lambda: crop_chin_image(
                        pil_img, frontal_margin, box, metadata, chin_offset=20
                    ),
                    "nose": lambda: crop_nose_image(
                        pil_img, box, landmarks, metadata, margin=0
                    ),
                    "below_lips": lambda: crop_below_lips_image(
                        pil_img, frontal_margin, landmarks, metadata, offset=10
                    ),
                    "auto": lambda: auto_crop(
                        pil_img,
                        frontal_margin,
                        profile_margin,
                        box,
                        landmarks,
                        metadata,
                        lip_offset=50,
                        neck_offset=50
                    ),
                }
                cropped_img = crop_functions.get(crop_style, lambda: None)()
                if cropped_img and aspect_ratio:
                    cropped_img = apply_aspect_ratio_filter(cropped_img, aspect_ratio)
                if cropped_img:
                    cropped_img = apply_filter(cropped_img, filter_name, filter_intensity)
                    save_image(cropped_img, output_path, metadata)
                else:
                    print(f"{filename}: Cropping failed. Skipping...")
                if os.path.dirname(input_path) != os.path.dirname(output_path):
                    os.remove(input_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            count += 1
    return count


def process_images_threaded(
    input_folder,
    output_folder,
    frontal_margin,
    profile_margin,
    sharpen=True,
    use_frontal=True,
    use_profile=True,
    progress_callback=None,
    cancel_func=None,
    apply_rotation=True,
    crop_style="auto",  # Changed default to "auto"
    filter_name="None",
    filter_intensity=50,
    aspect_ratio=None  # New parameter for aspect ratio
):
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".heic")
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    total = len(filenames)
    batch_size = max(1, total // (multiprocessing.cpu_count() * 2))
    max_workers = min(4, len(filenames) // batch_size) if batch_size > 0 else 1
    batches = [filenames[i : i + batch_size] for i in range(0, total, batch_size)]
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_batch,
                batch,
                input_folder,
                output_folder,
                frontal_margin,
                profile_margin,
                sharpen,
                use_frontal,
                use_profile,
                apply_rotation,
                crop_style,
                filter_name,
                filter_intensity,
                aspect_ratio  # Passing aspect ratio along
            ): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            if cancel_func and cancel_func():
                break
            try:
                batch_count = future.result()
                processed += batch_count
                if progress_callback:
                    progress_callback(processed, total, "Batch processed")
            except Exception as e:
                print(f"Error in batch: {e}")
    return processed, total