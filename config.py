import json
import os

# Path to the presets configuration file
PRESETS_PATH = os.path.join(os.path.dirname(__file__), 'presets.json')

# Load presets from JSON into a dictionary
try:
    with open(PRESETS_PATH, 'r', encoding='utf-8') as f:
        PRESETS = json.load(f)
except FileNotFoundError:
    PRESETS = {}
    print(f"Warning: presets.json not found at {PRESETS_PATH}. No presets loaded.")
except json.JSONDecodeError as e:
    PRESETS = {}
    print(f"Warning: Failed to parse presets.json: {e}. No presets loaded.")