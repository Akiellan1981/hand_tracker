import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cursor_map import map_to_screen, EMASmoother, DwellSelector


REGION = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}


def test_map_to_screen_center():
    x, y = map_to_screen(0.5, 0.5, REGION, 1920, 1080)
    assert abs(x - 960) < 1e-6
    assert abs(y - 540) < 1e-6


def test_map_to_screen_clamps_outside_region():
    x, y = map_to_screen(-1.0, 2.0, REGION, 1000, 1000)
    assert x == 0.0
    assert y == 1000.0


def test_ema_smoother_converges_toward_target():
    smoother = EMASmoother(alpha=0.5)
    smoother.update(0.0, 0.0)
    for _ in range(30):
        result = smoother.update(100.0, 100.0)
    assert abs(result[0] - 100.0) < 0.01


def test_ema_smoother_rejects_bad_alpha():
    try:
        EMASmoother(alpha=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_dwell_selector_triggers_after_duration_when_steady():
    selector = DwellSelector(stability_radius=10, dwell_seconds=3.0)
    t = 0.0
    progress, triggered = selector.update(100, 100, armed=True, now=t)
    assert triggered is False
    assert progress == 0.0

    t = 2.9
    progress, triggered = selector.update(100, 100, armed=True, now=t)
    assert triggered is False
    assert 0.9 < progress < 1.0

    t = 3.1
    progress, triggered = selector.update(100, 100, armed=True, now=t)
    assert triggered is True
    assert progress == 1.0


def test_dwell_selector_resets_on_movement():
    selector = DwellSelector(stability_radius=10, dwell_seconds=3.0)
    selector.update(100, 100, armed=True, now=0.0)
    selector.update(100, 100, armed=True, now=2.0)
    # jump far away -> timer restarts
    progress, triggered = selector.update(500, 500, armed=True, now=2.1)
    assert triggered is False
    assert progress == 0.0
    progress, triggered = selector.update(500, 500, armed=True, now=5.2)
    assert triggered is True


def test_dwell_selector_resets_when_disarmed():
    selector = DwellSelector(stability_radius=10, dwell_seconds=3.0)
    selector.update(100, 100, armed=True, now=0.0)
    selector.update(100, 100, armed=False, now=0.5)
    progress, triggered = selector.update(100, 100, armed=True, now=0.6)
    assert triggered is False
    assert progress == 0.0


def test_dwell_selector_does_not_refire_until_released():
    selector = DwellSelector(stability_radius=10, dwell_seconds=1.0)
    selector.update(100, 100, armed=True, now=0.0)
    _, triggered = selector.update(100, 100, armed=True, now=1.1)
    assert triggered is True
    # still held past the dwell time: must not trigger again
    _, triggered_again = selector.update(100, 100, armed=True, now=2.0)
    assert triggered_again is False
    # release and re-arm: can trigger again
    selector.update(100, 100, armed=False, now=2.1)
    selector.update(100, 100, armed=True, now=2.2)
    _, triggered_third = selector.update(100, 100, armed=True, now=3.3)
    assert triggered_third is True
