"""Pure math for hand landmarks: no camera/mediapipe dependency, so it is
unit-testable with synthetic points.

A "hand" is represented as a list of 21 (x, y, z) tuples in MediaPipe's
index order (0 = wrist, 4 = thumb tip, 8 = index tip, 12 = middle tip,
16 = ring tip, 20 = pinky tip).
"""
import math

WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
MIDDLE_MCP = 9

FINGER_JOINTS = {
    "thumb": (THUMB_IP, THUMB_TIP),
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

_EXTENSION_MARGIN = 1.05


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def finger_states(points):
    """Return {finger_name: is_extended} using a rotation-tolerant rule:
    a finger counts as extended when its tip sits farther from the wrist
    than its pip/ip joint does (by a small margin to avoid flicker at the
    boundary)."""
    wrist = points[WRIST]
    states = {}
    for name, (ref_idx, tip_idx) in FINGER_JOINTS.items():
        ref_dist = _dist(wrist, points[ref_idx])
        tip_dist = _dist(wrist, points[tip_idx])
        states[name] = tip_dist > ref_dist * _EXTENSION_MARGIN
    return states


def extended_count(states):
    return sum(1 for v in states.values() if v)


def only_these_extended(states, names):
    """True if exactly the fingers in `names` are extended, all others are not."""
    names = set(names)
    return all(states[f] == (f in names) for f in FINGER_ORDER)


def normalize_landmarks(points):
    """Translate so the wrist is the origin and scale by the wrist->middle-MCP
    distance, so the result is roughly invariant to hand position/distance
    from the camera. Returns a list of (x, y, z) tuples."""
    wrist = points[WRIST]
    scale = _dist(wrist, points[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    return [
        ((p[0] - wrist[0]) / scale, (p[1] - wrist[1]) / scale, (p[2] - wrist[2]) / scale)
        for p in points
    ]


def pinch_distance(points):
    """Distance between thumb tip and index tip. Feed normalized (scale
    invariant) points in so this stays consistent regardless of the
    hand's distance from the camera."""
    return _dist(points[THUMB_TIP], points[INDEX_TIP])


def flatten(points):
    flat = []
    for p in points:
        flat.extend(p)
    return flat


def euclidean_distance(a, b):
    """RMSE between two equal-length flat vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    total = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    return math.sqrt(total / len(a))


def resample_sequence(sequence, target_len):
    """Linearly resample a list of equal-length flat vectors (frames) to
    exactly `target_len` frames."""
    n = len(sequence)
    if n == 0:
        raise ValueError("cannot resample an empty sequence")
    if n == target_len:
        return [list(f) for f in sequence]
    if n == 1:
        return [list(sequence[0]) for _ in range(target_len)]

    dim = len(sequence[0])
    out = []
    for i in range(target_len):
        pos = i * (n - 1) / (target_len - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append([
            sequence[lo][d] * (1 - frac) + sequence[hi][d] * frac
            for d in range(dim)
        ])
    return out


def dtw_distance(seq_a, seq_b):
    """Classic O(n*m) dynamic time warping distance between two sequences of
    equal-length flat vectors, normalized by warp-path length so it's
    comparable across sequences of different lengths."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        raise ValueError("cannot DTW-compare an empty sequence")

    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float("inf")
    for j in range(1, m + 1):
        cost[0][j] = float("inf")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = euclidean_distance(seq_a[i - 1], seq_b[j - 1])
            cost[i][j] = d + min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1])

    return cost[n][m] / (n + m)
