"""Loads config.json and fills in any missing keys from the defaults, so
partial user edits (e.g. only overriding `dwell_seconds`) don't require
repeating the whole file.
"""
import copy
import json
import os

DEFAULTS = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "mirror": True,
    "model_path": "models/hand_landmarker.task",
    "actions_file": "data/actions.json",
    "log_file": "data/action_log.csv",
    "max_num_hands": 2,
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.6,
    "min_hand_presence_confidence": 0.6,
    "custom_action_recognition": {
        "enabled": True,
        "pose_match_threshold": 0.18,
        "motion_match_threshold": 0.22,
        "motion_sequence_length": 20,
        "motion_buffer_frames": 45,
        "recognition_cooldown_seconds": 1.5,
        "pose_recording_frames": 30,
        "motion_recording_max_frames": 60,
    },
    "cursor_control": {
        "enabled_by_default": False,
        "toggle_key": "m",
        "controlling_hand": "Right",
        "control_region": {"x_min": 0.12, "y_min": 0.12, "x_max": 0.88, "y_max": 0.88},
        "smoothing_alpha": 0.55,
        "dwell_seconds": 3.0,
        "dwell_stability_radius_px": 45,
        "scroll_sensitivity": 18.0,
        "scroll_dead_zone": 0.004,
        "scroll_invert": False,
        "zoom_sensitivity": 40.0,
        "zoom_dead_zone": 0.003,
        "zoom_invert": False,
    },
}


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    defaults = copy.deepcopy(DEFAULTS)
    if not os.path.exists(path):
        return defaults
    with open(path, "r") as f:
        user_config = json.load(f)
    return _deep_merge(defaults, user_config)
