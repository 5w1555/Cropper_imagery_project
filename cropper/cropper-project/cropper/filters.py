def apply_filter(image, filter_type, intensity):
    if filter_type == "Brightness":
        return apply_brightness(image, intensity)
    elif filter_type == "Sepia":
        return apply_sepia(image, intensity)
    elif filter_type == "CircleMask":
        return apply_circle_mask(image)
    elif filter_type == "AspectRatio":
        return apply_aspect_ratio_filter(image, intensity)
    else:
        raise ValueError("Unknown filter type")

def apply_brightness(image, intensity):
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(intensity / 100.0)

def apply_sepia(image, intensity):
    sepia_filter = [
        [0.393 + 0.607 * (1 - intensity / 100.0), 0.769 - 0.769 * (1 - intensity / 100.0), 0.189 - 0.189 * (1 - intensity / 100.0)],
        [0.349 - 0.349 * (1 - intensity / 100.0), 0.686 + 0.314 * (1 - intensity / 100.0), 0.168 - 0.168 * (1 - intensity / 100.0)],
        [0.272 - 0.272 * (1 - intensity / 100.0), 0.534 - 0.534 * (1 - intensity / 100.0), 0.131 + 0.869 * (1 - intensity / 100.0)]
    ]
    return image.convert("RGB").transform(image.size, Image.MATRIX, sepia_filter)

def apply_circle_mask(image):
    from PIL import ImageDraw
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image.size
    draw.ellipse((0, 0, width, height), fill=255)
    return image.convert("RGBA").putalpha(mask)

def apply_aspect_ratio_filter(image, aspect_ratio):
    width, height = image.size
    new_height = int(width / aspect_ratio)
    return image.resize((width, new_height), Image.ANTIALIAS)