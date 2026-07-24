import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gesture_modes as gm


def states(thumb=False, index=False, middle=False, ring=False, pinky=False):
    return {"thumb": thumb, "index": index, "middle": middle, "ring": ring, "pinky": pinky}


def test_index_only_thumb_curled_is_move():
    assert gm.classify_mode(states(index=True)) == gm.MOVE


def test_index_only_thumb_extended_is_not_move():
    # thumb out turns this into the ZOOM pinch shape's neighbor, not MOVE
    assert gm.classify_mode(states(thumb=True, index=True)) == gm.ZOOM


def test_index_and_middle_thumb_curled_is_select():
    assert gm.classify_mode(states(index=True, middle=True)) == gm.SELECT


def test_index_middle_ring_thumb_curled_is_drag():
    assert gm.classify_mode(states(index=True, middle=True, ring=True)) == gm.DRAG


def test_all_curled_including_thumb_is_fist():
    assert gm.classify_mode(states()) == gm.FIST


def test_four_fingers_thumb_curled_is_scroll():
    assert gm.classify_mode(states(index=True, middle=True, ring=True, pinky=True)) == gm.SCROLL


def test_thumb_and_index_only_is_zoom():
    assert gm.classify_mode(states(thumb=True, index=True)) == gm.ZOOM


def test_open_hand_all_five_is_calibrate():
    assert gm.classify_mode(states(thumb=True, index=True, middle=True, ring=True, pinky=True)) == gm.CALIBRATE


def test_unrecognized_combination_is_none():
    assert gm.classify_mode(states(index=True, ring=True)) == gm.NONE


def test_mode_tracker_reports_transitions_once():
    tracker = gm.ModeTracker()
    entered, exited = tracker.update(gm.MOVE)
    assert entered == gm.MOVE and exited == gm.NONE

    entered, exited = tracker.update(gm.MOVE)
    assert entered is None and exited is None

    entered, exited = tracker.update(gm.DRAG)
    assert entered == gm.DRAG and exited == gm.MOVE
