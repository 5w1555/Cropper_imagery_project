## 🖼️ Cropper Imagery — Python Image Cropping Tool

**Cropper Imagery** automatically detects faces and crops images to platform-specific formats (Instagram, LinkedIn, TikTok, etc.).

Built to automate repetitive tasks like:

* Creating profile pictures
* Preparing images for social media
* Preprocessing e-commerce visuals

---

### ⚙️ Key Features

* Face detection with landmark recognition
* Auto-rotation correction
* Format presets for multiple platforms
* Batch processing for folders
* Single file preview with Gradio
* Image filters: sharpness, margins, lighting

---

### 📁 Project Structure

```
cropper_imagery/
├── cropper/            # Face-part-based cropping modules
├── processing.py       # Batch processing script
├── gradio_app.py       # Quick preview interface
├── presets.json        # Configurable format presets
├── README.md
```

---

### 📂 Installation & Usage

```bash
git clone https://github.com/TechBooper/Cropper_imagery_project
cd Cropper_imagery_project
pip install -r requirements.txt
```

**To process a folder (requires refactoring the file):**
Edit `processing.py` as you wish, then run:

```bash
python processing.py
```

**To preview a classic use (requires Gradio):**

```bash
python gradio_app.py
```
Go to: http://localhost:7860. Gradio is extremely versatile and can even be used through an API but do as you wish!

---

### 🔧 Included Formats

* `instagram_square` → 1:1
* `linkedin_cover` → 1.91:1
* `tiktok_story` → 9:16
* `headbust` → crops top of the face
* ✅ Fully customizable via `presets.json`

---

### 👨‍💻 Author

I'm a Python developer based in Île-de-France.

This is a personal production-oriented tool built entirely solo to help a friend's business.

---

### 📅 Roadmap (Next Steps)

* Full GUI (Tkinter or Web UI I don't know yet)
* Improved filter engine
* CLI integration
* Full test coverage
* Refactored functions (It's still very rough)

---

### 📄 License

MIT
