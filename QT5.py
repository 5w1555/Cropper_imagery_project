# ----------------------------
# PyQt5 GUI Implementation
# ----------------------------

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



class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Cropper")
        self.resize(600, 700)  # increased height to accommodate preview and controls
        self.init_ui()
        create_required_folders()
        # Variables to store preview data
        self.current_pil_image = None
        self.current_landmarks = None
        self.current_box = None
        self.current_metadata = None
        self.worker = None

        # Set up a QTimer to throttle preview updates
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(300)  # delay in milliseconds
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview_now)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Folder selection grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        grid.addWidget(self.input_folder_edit, 0, 1)
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self.select_input_folder)
        grid.addWidget(btn_input, 0, 2)

        grid.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        grid.addWidget(self.output_folder_edit, 1, 1)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.select_output_folder)
        grid.addWidget(btn_output, 1, 2)
        layout.addLayout(grid)

        # Parameters (Margins)
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Frontal Margin (px):"))
        self.margin_edit = QLineEdit("20")
        params_layout.addWidget(self.margin_edit)
        params_layout.addWidget(QLabel("Profile Margin (px):"))
        self.side_trim_edit = QLineEdit("20")
        params_layout.addWidget(self.side_trim_edit)
        layout.addLayout(params_layout)

        # Connect margin edits to trigger the preview timer (instead of immediate update)
        self.margin_edit.textChanged.connect(self.restart_preview_timer)
        self.side_trim_edit.textChanged.connect(self.restart_preview_timer)

        # Checkboxes for additional options
        options_layout = QHBoxLayout()
        self.sharpen_checkbox = QCheckBox("Sharpen Image")
        self.sharpen_checkbox.setChecked(True)
        options_layout.addWidget(self.sharpen_checkbox)
        self.frontal_checkbox = QCheckBox("Use Frontal Cropping")
        self.frontal_checkbox.setChecked(True)
        options_layout.addWidget(self.frontal_checkbox)
        self.profile_checkbox = QCheckBox("Use Profile Cropping")
        self.profile_checkbox.setChecked(True)
        options_layout.addWidget(self.profile_checkbox)
        # Connect checkbox changes to trigger preview updates
        self.frontal_checkbox.stateChanged.connect(self.restart_preview_timer)
        self.profile_checkbox.stateChanged.connect(self.restart_preview_timer)
        layout.addLayout(options_layout)
        self.rotation_checkbox = QCheckBox("Correct Face Rotation")
        self.rotation_checkbox.setChecked(True)
        options_layout.addWidget(self.rotation_checkbox)

        # New Crop Style Controls
        crop_style_layout = QHBoxLayout()
        crop_style_layout.addWidget(QLabel("Crop Style:"))
        self.crop_style_combo = QComboBox()
        self.crop_style_combo.addItems(
            ["auto", "frontal", "profile", "chin", "nose", "below_lips"]
        )
        self.crop_style_combo.currentTextChanged.connect(self.restart_preview_timer)
        crop_style_layout.addWidget(self.crop_style_combo)
        layout.addLayout(crop_style_layout)

        # Filter controls
        filter_layout = QFormLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["None", "Brightness", "Contrast", "Blur", "Edge Detection"]
        )
        self.filter_combo.currentTextChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Filter:", self.filter_combo)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Intensity:", self.intensity_slider)
        
        # Aspect Ratio Dropdown for cropping
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["3:2", "4:3", "16:9"])
        self.aspect_ratio_combo.currentTextChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Aspect Ratio:", self.aspect_ratio_combo)
        
        layout.addLayout(filter_layout)

        # Progress display
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.start_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_processing)
        btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout)

        # Preview display area
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(600, 400)
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        # Button to load a preview image
        self.preview_button = QPushButton("Load Preview")
        self.preview_button.clicked.connect(self.load_preview)
        layout.addWidget(self.preview_button)

        central_widget.setLayout(layout)

    def restart_preview_timer(self):
        # Restart the timer every time a parameter changes.
        self.preview_timer.start()

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def update_progress(self, current, total, message):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processed {current}/{total} images. {message}")

    def pil_to_pixmap(self, pil_img):
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)

    def load_preview(self):
        input_folder = self.input_folder_edit.text().strip()
        if not input_folder:
            QMessageBox.critical(self, "Error", "Please select an input folder first.")
            return
        valid_exts = (".jpg", ".jpeg", ".png", ".heic")
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
        if not files:
            QMessageBox.critical(
                self, "Error", "No valid image files found in the input folder."
            )
            return
        file_path = os.path.join(input_folder, files[0])
        result = get_face_and_landmarks(
            file_path,
            sharpen=self.sharpen_checkbox.isChecked(),
            apply_rotation=self.rotation_checkbox.isChecked(),
        )
        if result is None or result[0] is None or result[1] is None:
            QMessageBox.critical(self, "Error", "Failed to process image for preview.")
            return
        box, landmarks, cv_img, pil_img, metadata = result
        self.current_pil_image = pil_img
        self.current_landmarks = landmarks
        self.current_box = box
        self.current_metadata = metadata
        self.update_preview_now()

    def update_preview_now(self):
        if not self.current_pil_image:
            return
        try:
            frontal_margin = int(self.margin_edit.text())
            profile_margin = int(self.side_trim_edit.text())
        except ValueError:
            return

        crop_style = self.crop_style_combo.currentText()
        # Use a dictionary to map crop style to its lambda:
        crop_funcs = {
            "frontal": lambda: (
                crop_frontal_image_preview(
                    self.current_pil_image,
                    frontal_margin,
                    self.current_landmarks,
                    self.current_metadata,
                    lip_offset=50,
                )
                if self.frontal_checkbox.isChecked() and self.current_landmarks
                else None
            ),
            "profile": lambda: (
                crop_profile_image_preview(
                    self.current_pil_image,
                    profile_margin,
                    50,
                    self.current_box,
                    self.current_metadata,
                )
                if self.profile_checkbox.isChecked() and self.current_box
                else None
            ),
            "chin": lambda: crop_chin_image(
                self.current_pil_image,
                frontal_margin,
                self.current_box,
                self.current_metadata,
                chin_offset=20,
            ),
            "nose": lambda: crop_nose_image(
                self.current_pil_image,
                self.current_box,
                self.current_landmarks,
                self.current_metadata,
                margin=0,
            ),
            "below_lips": lambda: crop_below_lips_image(
                self.current_pil_image,
                frontal_margin,
                self.current_landmarks,
                self.current_metadata,
                offset=10,
            ),
            "auto": lambda: auto_crop(
                self.current_pil_image,
                frontal_margin,
                profile_margin,
                self.current_box,
                self.current_landmarks,
                self.current_metadata,
            ),
        }
        cropped_img = crop_funcs.get(crop_style, lambda: None)()
        
        # Retrieve selected aspect ratio and enforce it if a crop exists
        selected_ratio = self.aspect_ratio_combo.currentText()
        if selected_ratio == "3:2":
            target_ratio = 3 / 2
        elif selected_ratio == "4:3":
            target_ratio = 4 / 3
        elif selected_ratio == "16:9":
            target_ratio = 16 / 9
        else:
            target_ratio = None

        if cropped_img and target_ratio:
            cropped_img = apply_aspect_ratio_filter(cropped_img, target_ratio)

        if cropped_img:
            filter_name = self.filter_combo.currentText()
            intensity = self.intensity_slider.value()
            cropped_img = apply_filter(cropped_img, filter_name, intensity)
            pixmap = self.pil_to_pixmap(cropped_img)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            self.preview_label.setText("No preview available.")


    def start_processing(self):
        input_folder = self.input_folder_edit.text().strip()
        output_folder = self.output_folder_edit.text().strip()
        try:
            frontal_margin = int(self.margin_edit.text())
            profile_margin = int(self.side_trim_edit.text())
        except ValueError:
            QMessageBox.critical(
                self, "Error", "Margin and Profile Margin must be integers."
            )
            return
        if not input_folder or not output_folder:
            QMessageBox.critical(
                self, "Error", "Please select both input and output folders."
            )
            return

        # Get selected aspect ratio from the dropdown
        selected_ratio = self.aspect_ratio_combo.currentText()
        if selected_ratio == "3:2":
            aspect_ratio = 3 / 2
        elif selected_ratio == "4:3":
            aspect_ratio = 4 / 3
        elif selected_ratio == "16:9":
            aspect_ratio = 16 / 9
        else:
            aspect_ratio = None

        sharpen = self.sharpen_checkbox.isChecked()
        use_frontal = self.frontal_checkbox.isChecked()
        use_profile = self.profile_checkbox.isChecked()
        crop_style = self.crop_style_combo.currentText()

        self.start_button.setEnabled(False)
        self.thread = QThread()
        self.worker = Worker(
            input_folder,
            output_folder,
            frontal_margin,
            profile_margin,
            sharpen,
            use_frontal,
            use_profile,
            self.rotation_checkbox.isChecked(),
            crop_style,
            self.filter_combo.currentText(),
            self.intensity_slider.value(),
            aspect_ratio  # Pass the aspect ratio here
        )
        self.worker.moveToThread(self.thread)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling processing...")
            self.start_button.setEnabled(False)

    def on_finished(self, processed, total):
        self.status_label.setText(f"Successfully processed {processed}/{total} images")
        self.progress_bar.setValue(100)
        QMessageBox.information(
            self, "Complete", f"Processed {processed} of {total} images"
        )
        self.start_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        self.worker = None

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        self.worker = None


class Worker(QObject):
    finished = pyqtSignal(int, int)
    progress_update = pyqtSignal(int, int, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_folder,
        output_folder,
        frontal_margin,
        profile_margin,
        sharpen,
        use_frontal,
        use_profile,
        correct_rotation,
        crop_style,
        filter_name,
        filter_intensity,
        aspect_ratio  # New parameter
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.frontal_margin = frontal_margin
        self.profile_margin = profile_margin
        self.sharpen = sharpen
        self.use_frontal = use_frontal
        self.use_profile = use_profile
        self.correct_rotation = correct_rotation
        self.crop_style = crop_style
        self.filter_name = filter_name
        self.filter_intensity = filter_intensity
        self.aspect_ratio = aspect_ratio
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled

    def run(self):
        try:
            processed, total = process_images_threaded(
                self.input_folder,
                self.output_folder,
                self.frontal_margin,
                self.profile_margin,
                self.sharpen,
                self.use_frontal,
                self.use_profile,
                self.progress_update.emit,
                cancel_func=self.is_cancelled,
                apply_rotation=self.correct_rotation,
                crop_style=self.crop_style,
                filter_name=self.filter_name,
                filter_intensity=self.filter_intensity,
                aspect_ratio=self.aspect_ratio
            )
            if self.is_cancelled():
                self.error.emit("Processing was cancelled.")
            else:
                self.finished.emit(processed, total)
        except Exception as e:
            self.error.emit(str(e))