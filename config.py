# config.py
import json
import os
from functools import lru_cache

# Path to the presets configuration file
PRESETS_PATH = os.path.join(os.path.dirname(__file__), 'presets.json')

@lru_cache()
def load_presets():
    try:
        with open(PRESETS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Direct access if you really need it
PRESETS = load_presets()

def get_preset_keys():
    """Returns the JSON keys, e.g. ['square_avatar', 'portrait', ...]"""
    return list(PRESETS.keys())

def get_preset_labels():
    """Returns the human‐readable names in same order as keys."""
    return [PRESETS[k]["name"] for k in get_preset_keys()]

def key_for_label(label):
    """Reverse lookup: label → key."""
    for key, cfg in PRESETS.items():
        if cfg.get("name") == label:
            return key
    raise KeyError(f"No preset with label {label!r}")

def get_preset_by_key(key):
    """Returns the full dict for a given key."""
    return PRESETS[key]
