def map_slider_to_multiplier(slider_value):
    return 0.5 + (slider_value / 100)  # Maps 0 to 0.5 and 100 to 1.5

def map_slider_to_blur_radius(slider_value):
    return (slider_value / 100) * 5  # Maps 0 to 0 and 100 to 5

# Additional utility functions can be added here as needed.