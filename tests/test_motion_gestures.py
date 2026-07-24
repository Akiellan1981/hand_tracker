import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.motion_gestures import RelativeDeltaTracker


def test_first_update_returns_zero_ticks():
    tracker = RelativeDeltaTracker(sensitivity=10, dead_zone=0.001)
    assert tracker.update(0.5) == 0


def test_small_movement_below_dead_zone_is_ignored():
    tracker = RelativeDeltaTracker(sensitivity=10, dead_zone=0.05)
    tracker.update(0.5)
    assert tracker.update(0.51) == 0


def test_sustained_movement_accumulates_into_ticks():
    tracker = RelativeDeltaTracker(sensitivity=10, dead_zone=0.001)
    tracker.update(0.0)
    total_ticks = 0
    value = 0.0
    for _ in range(20):
        value += 0.05
        total_ticks += tracker.update(value)
    assert total_ticks > 0


def test_invert_flips_tick_sign():
    up = RelativeDeltaTracker(sensitivity=10, dead_zone=0.001, invert=False)
    down = RelativeDeltaTracker(sensitivity=10, dead_zone=0.001, invert=True)
    up.update(0.0)
    down.update(0.0)
    ticks_up = up.update(1.0)
    ticks_down = down.update(1.0)
    assert ticks_up > 0 and ticks_down < 0


def test_reset_forgets_last_value():
    tracker = RelativeDeltaTracker(sensitivity=10, dead_zone=0.001)
    tracker.update(0.0)
    tracker.reset()
    assert tracker.update(5.0) == 0  # treated as first sample again
