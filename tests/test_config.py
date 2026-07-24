import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, DEFAULTS


def test_missing_file_returns_defaults():
    config = load_config("/nonexistent/path/config.json")
    assert config == DEFAULTS


def test_missing_file_returns_a_copy_not_the_shared_defaults():
    config = load_config("/nonexistent/path/config.json")
    config["camera_index"] = 99
    assert DEFAULTS["camera_index"] == 0  # unaffected


def test_partial_top_level_override():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"camera_index": 2}, f)
        config = load_config(path)
        assert config["camera_index"] == 2
        assert config["frame_width"] == DEFAULTS["frame_width"]  # untouched default


def test_partial_nested_override_keeps_sibling_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"cursor_control": {"dwell_seconds": 5.0}}, f)
        config = load_config(path)
        assert config["cursor_control"]["dwell_seconds"] == 5.0
        # sibling keys in the same nested section stay at their defaults
        assert config["cursor_control"]["smoothing_alpha"] == DEFAULTS["cursor_control"]["smoothing_alpha"]
        assert config["cursor_control"]["controlling_hand"] == DEFAULTS["cursor_control"]["controlling_hand"]


def test_deeply_nested_control_region_partial_override():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w") as f:
            json.dump({"cursor_control": {"control_region": {"x_min": 0.5}}}, f)
        config = load_config(path)
        assert config["cursor_control"]["control_region"]["x_min"] == 0.5
        assert config["cursor_control"]["control_region"]["y_min"] == \
            DEFAULTS["cursor_control"]["control_region"]["y_min"]
