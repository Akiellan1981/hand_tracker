import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import landmarks


def make_hand(finger_up):
    """Build a synthetic 21-point hand. finger_up: dict of
    {'thumb','index','middle','ring','pinky'} -> bool. Extended fingers
    get a tip far from the wrist; curled fingers get a tip close to the
    wrist (closer than its pip joint)."""
    points = [(0.0, 0.0, 0.0)] * 21
    points[0] = (0.0, 0.0, 0.0)  # wrist
    points[9] = (0.0, -0.3, 0.0)  # middle mcp, used for scale

    joints = {
        "thumb": (landmarks.THUMB_IP, landmarks.THUMB_TIP, -0.15),
        "index": (landmarks.INDEX_PIP, landmarks.INDEX_TIP, -0.4),
        "middle": (landmarks.MIDDLE_PIP, landmarks.MIDDLE_TIP, -0.45),
        "ring": (landmarks.RING_PIP, landmarks.RING_TIP, -0.4),
        "pinky": (landmarks.PINKY_PIP, landmarks.PINKY_TIP, -0.35),
    }
    points = list(points)
    for name, (pip_idx, tip_idx, extended_y) in joints.items():
        points[pip_idx] = (0.05, -0.2, 0.0)
        if finger_up.get(name):
            points[tip_idx] = (0.05, extended_y, 0.0)  # farther from wrist
        else:
            points[tip_idx] = (0.02, -0.05, 0.0)  # closer to wrist than pip
    return points


ALL_DOWN = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}


def test_finger_states_all_down_is_fist():
    hand = make_hand(ALL_DOWN)
    states = landmarks.finger_states(hand)
    assert landmarks.extended_count(states) == 0


def test_finger_states_index_only():
    up = dict(ALL_DOWN, index=True)
    hand = make_hand(up)
    states = landmarks.finger_states(hand)
    assert states["index"] is True
    assert states["middle"] is False
    assert landmarks.extended_count(states) == 1
    assert landmarks.only_these_extended(states, ["index"])


def test_finger_states_index_and_middle():
    up = dict(ALL_DOWN, index=True, middle=True)
    hand = make_hand(up)
    states = landmarks.finger_states(hand)
    assert landmarks.extended_count(states) == 2
    assert landmarks.only_these_extended(states, ["index", "middle"])


def test_normalize_landmarks_is_translation_and_scale_invariant():
    hand_a = make_hand(dict(ALL_DOWN, index=True))
    shifted = [(x + 5.0, y - 3.0, z) for (x, y, z) in hand_a]
    norm_a = landmarks.normalize_landmarks(hand_a)
    norm_shifted = landmarks.normalize_landmarks(shifted)
    for pa, pb in zip(norm_a, norm_shifted):
        assert abs(pa[0] - pb[0]) < 1e-9
        assert abs(pa[1] - pb[1]) < 1e-9


def test_euclidean_distance_zero_for_identical_vectors():
    a = [1.0, 2.0, 3.0, 4.0]
    assert landmarks.euclidean_distance(a, list(a)) == 0.0


def test_euclidean_distance_mismatched_length_raises():
    try:
        landmarks.euclidean_distance([1.0], [1.0, 2.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_resample_sequence_preserves_endpoints():
    seq = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    out = landmarks.resample_sequence(seq, 10)
    assert len(out) == 10
    assert out[0] == [0.0, 0.0]
    assert out[-1] == [3.0, 3.0]


def test_resample_single_frame_repeats():
    out = landmarks.resample_sequence([[1.0, 2.0]], 5)
    assert out == [[1.0, 2.0]] * 5


def test_dtw_distance_zero_for_identical_sequences():
    seq = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert landmarks.dtw_distance(seq, seq) == 0.0


def test_pinch_distance_shrinks_as_fingers_come_together():
    far = [(0.0, 0.0, 0.0)] * 21
    far[landmarks.THUMB_TIP] = (0.0, 0.0, 0.0)
    far[landmarks.INDEX_TIP] = (1.0, 0.0, 0.0)
    near = list(far)
    near[landmarks.INDEX_TIP] = (0.1, 0.0, 0.0)
    assert landmarks.pinch_distance(near) < landmarks.pinch_distance(far)


def test_dtw_distance_robust_to_speed_difference():
    seq_a = [[float(i), 0.0] for i in range(5)]
    seq_b_slow = landmarks.resample_sequence(seq_a, 20)  # same path, more frames
    seq_c_different = [[0.0, float(i)] for i in range(5)]  # different path

    d_same_path = landmarks.dtw_distance(seq_a, seq_b_slow)
    d_diff_path = landmarks.dtw_distance(seq_a, seq_c_different)
    assert d_same_path < d_diff_path
