"""Downloads MediaPipe's official hand_landmarker.task model file on first
run, since it's a ~7.8MB binary asset that doesn't belong committed to
the repo.
"""
import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MIN_EXPECTED_BYTES = 1_000_000  # sanity check against a truncated/failed download


def download_if_missing(model_path):
    if os.path.exists(model_path) and os.path.getsize(model_path) >= MIN_EXPECTED_BYTES:
        return model_path

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"Downloading hand landmark model to {model_path} ...")
    tmp_path = model_path + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp_path)

    if os.path.getsize(tmp_path) < MIN_EXPECTED_BYTES:
        os.remove(tmp_path)
        raise RuntimeError("downloaded model file looks truncated; try again")

    os.replace(tmp_path, model_path)
    print("Model ready.")
    return model_path
