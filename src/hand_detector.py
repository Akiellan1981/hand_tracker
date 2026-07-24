"""Wraps MediaPipe's HandLandmarker (Tasks API) for real-time video.

Important convention: the caller must feed this detector frames that have
already been horizontally flipped (mirrored), i.e. the natural "selfie
view" you'd see in any webcam app. MediaPipe's handedness classifier is
trained assuming a mirrored/selfie input and, given one, reports the
handedness of the person in the image directly (its "Left"/"Right" label
already matches what you'd call your own left/right hand). Feed it an
unflipped frame and every label comes out swapped.
"""
import time

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
)


class HandObservation:
    def __init__(self, handedness, score, landmarks):
        self.handedness = handedness  # "Left" or "Right"
        self.score = score
        self.landmarks = landmarks  # list of 21 (x, y, z) normalized tuples


class HandDetector:
    def __init__(self, model_path, max_num_hands=2, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6, min_hand_presence_confidence=0.6):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

    def detect(self, frame_rgb):
        """frame_rgb: an HxWx3 RGB numpy array (already mirrored). Returns a
        list of HandObservation, one per detected hand."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        observations = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            top = handedness[0]
            points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            observations.append(HandObservation(top.category_name, top.score, points))
        return observations

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
