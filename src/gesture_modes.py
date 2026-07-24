"""Classifies the fixed cursor-control gesture vocabulary from finger
extension states. Pure logic, no camera/OS dependency.

Vocabulary, one mode per exact finger shape of the controlling hand:
    FIST      - every finger curled, thumb included -> Select All
    MOVE      - index only (thumb curled) -> move the cursor
    SELECT    - index + middle (thumb curled) -> hold steady
                `dwell_seconds` to select-and-open (double-click)
    DRAG      - index + middle + ring (thumb curled) -> click-and-drag
                while held
    SCROLL    - index + middle + ring + pinky, thumb curled -> hand
                moving up/down scrolls up/down
    ZOOM      - thumb + index only (middle/ring/pinky curled) -> the
                thumb-index pinch distance drives zoom: spreading them
                apart zooms in, bringing them together zooms out
    CALIBRATE - all 5 fingers extended (open palm) -> recenter cursor
                smoothing and release any active select/drag
    NONE      - anything else -> no cursor action

The thumb is required down for MOVE/SELECT/DRAG/SCROLL specifically so
those poses can't be confused with ZOOM's thumb-out pinch shape.
"""

MOVE = "move"
SELECT = "select"
DRAG = "drag"
FIST = "fist"
SCROLL = "scroll"
ZOOM = "zoom"
CALIBRATE = "calibrate"
NONE = "none"


def _matches(states, up=(), down=()):
    return all(states[f] for f in up) and all(not states[f] for f in down)


def classify_mode(states):
    if _matches(states, down=("thumb", "index", "middle", "ring", "pinky")):
        return FIST
    if _matches(states, up=("index",), down=("thumb", "middle", "ring", "pinky")):
        return MOVE
    if _matches(states, up=("index", "middle"), down=("thumb", "ring", "pinky")):
        return SELECT
    if _matches(states, up=("index", "middle", "ring"), down=("thumb", "pinky")):
        return DRAG
    if _matches(states, up=("index", "middle", "ring", "pinky"), down=("thumb",)):
        return SCROLL
    if _matches(states, up=("thumb", "index"), down=("middle", "ring", "pinky")):
        return ZOOM
    if _matches(states, up=("thumb", "index", "middle", "ring", "pinky")):
        return CALIBRATE
    return NONE


class ModeTracker:
    """Tracks the previous frame's mode and reports the mode entered/exited
    this frame (None if the mode is unchanged)."""

    def __init__(self):
        self.previous = NONE

    def update(self, mode):
        if mode == self.previous:
            return None, None
        entered, exited = mode, self.previous
        self.previous = mode
        return entered, exited
