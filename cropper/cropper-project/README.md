# Cropper Application

## Overview
The Cropper application is a Python package designed for image processing, specifically focused on cropping images based on facial landmarks and applying various enhancements and filters. It supports multiple image formats and provides utilities for managing color profiles.

## Directory Structure
```
cropper-project
├── cropper
│   ├── __init__.py
│   ├── io.py
│   ├── color_profile.py
│   ├── filters.py
│   ├── enhancement.py
│   ├── detection.py
│   ├── utils.py
│   └── cropping
│       ├── __init__.py
│       ├── strategies.py
│       ├── frontal.py
│       ├── profile.py
│       └── common.py
├── tests
│   └── test_cropper.py
└── README.md
```

## Installation
To install the required packages, you can use pip. Make sure you have Python 3.x installed, then run:

```
pip install -r requirements.txt
```

## Usage
1. **Import the Cropper package**:
   You can import the necessary modules from the cropper package in your Python scripts.

   ```python
   from cropper import io, color_profile, filters, enhancement, detection, utils, cropping
   ```

2. **Reading and Saving Images**:
   Use the `io` module to read and save images in various formats.

   ```python
   image = io.read_image('path/to/image.jpg')
   io.save_image(image, 'path/to/save/image.png')
   ```

3. **Applying Filters**:
   You can apply different filters to your images using the `filters` module.

   ```python
   filtered_image = filters.apply_sepia(image, intensity=0.8)
   ```

4. **Enhancing Images**:
   Use the `enhancement` module to enhance images, such as adjusting lighting for faces.

   ```python
   enhanced_image = enhancement.enhance_lighting_for_faces(image)
   ```

5. **Cropping Images**:
   The `cropping` module provides various strategies for cropping images based on facial landmarks.

   ```python
   cropped_image = cropping.strategies.head_bust_crop(image)
   ```

## Testing
To run the tests for the Cropper package, navigate to the `tests` directory and run:

```
pytest test_cropper.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.