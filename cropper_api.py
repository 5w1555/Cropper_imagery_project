import os
from config import PRESETS
from cropper import head_bust_crop


def crop_with_preset(input_path, model, preset_key, **overrides):
    """
    Crop an image using a named preset from presets.json.

    Args:
        input_path (str or PIL.Image): Path to input image or PIL.Image instance.
        model: Pre-initialized face-detection model (e.g., RetinaFace).
        preset_key (str): Key in PRESETS dict identifying margin and ratio.
        **overrides: Optional overrides for preset values (e.g., margin, target_ratio).

    Returns:
        PIL.Image or None: Cropped image, or None on failure.

    Raises:
        ValueError: If preset_key is not found in PRESETS.
    """
    # Lookup preset
    preset = PRESETS.get(preset_key)
    if preset is None:
        raise ValueError(f"Unknown preset '{preset_key}'")

    # Merge preset parameters with any overrides
    margin       = overrides.get('margin', preset.get('margin'))
    target_ratio = overrides.get('target_ratio', preset.get('target_ratio'))
    conf_thresh  = overrides.get('conf_threshold', preset.get('conf_threshold', 0.3))

    # Call core cropping function
    return head_bust_crop(
        input_path,
        model,
        margin=margin,
        target_ratio=target_ratio,
        conf_threshold=conf_thresh
    )


if __name__ == "__main__":
    # Example CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="Crop with a named preset.")
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('preset', choices=list(PRESETS.keys()), help='Preset key')
    parser.add_argument('--out', default='output.png', help='Output path')
    args = parser.parse_args()

    # Load model here if needed
    from retinaface.pre_trained_models import get_model
    import torch
    device = torch.device('cpu')
    model = get_model('resnet50_2020-07-20', max_size=2048, device=device)
    model.eval()

    cropped = crop_with_preset(args.image, model, args.preset)
    if cropped:
        # Ensure output directory exists
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        cropped.save(args.out)
        print(f"Saved cropped image to {args.out}")
    else:
        print("Cropping failed.")
