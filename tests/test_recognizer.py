import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.action_store import Action, ActionStore
from src.recognizer import Recognizer, HandMotionBuffer


def test_pose_match_within_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Thumbs Up", "pose", "Any", [0.0, 0.0, 0.0], threshold=0.1))
        recognizer = Recognizer(store, motion_sequence_length=10, cooldown_seconds=1.0)

        match = recognizer.match("Right", [0.01, 0.0, 0.0], HandMotionBuffer(30), now=0.0)
        assert match is not None
        assert match.name == "Thumbs Up"


def test_pose_no_match_outside_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Thumbs Up", "pose", "Any", [0.0, 0.0, 0.0], threshold=0.05))
        recognizer = Recognizer(store, motion_sequence_length=10, cooldown_seconds=1.0)

        match = recognizer.match("Right", [1.0, 1.0, 1.0], HandMotionBuffer(30), now=0.0)
        assert match is None


def test_hand_filter_excludes_wrong_hand():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Left Only", "pose", "Left", [0.0, 0.0], threshold=0.1))
        recognizer = Recognizer(store, motion_sequence_length=10, cooldown_seconds=1.0)

        assert recognizer.match("Right", [0.0, 0.0], HandMotionBuffer(30), now=0.0) is None
        assert recognizer.match("Left", [0.0, 0.0], HandMotionBuffer(30), now=0.0) is not None


def test_cooldown_suppresses_repeat_trigger():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Wave", "pose", "Any", [0.0], threshold=0.1))
        recognizer = Recognizer(store, motion_sequence_length=10, cooldown_seconds=2.0)

        first = recognizer.match("Any", [0.0], HandMotionBuffer(30), now=0.0)
        assert first is not None

        during_cooldown = recognizer.match("Any", [0.0], HandMotionBuffer(30), now=1.0)
        assert during_cooldown is None

        after_cooldown = recognizer.match("Any", [0.0], HandMotionBuffer(30), now=2.5)
        assert after_cooldown is not None


def test_motion_action_matches_similar_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        template = [[float(i), 0.0] for i in range(10)]
        store.add(Action("Swipe Right", "motion", "Any", template, threshold=0.5))
        recognizer = Recognizer(store, motion_sequence_length=10, cooldown_seconds=1.0)

        buffer = HandMotionBuffer(30)
        for i in range(12):
            buffer.push([float(i) * (10 / 12), 0.0])

        match = recognizer.match("Any", [0.0, 0.0], buffer, now=0.0)
        assert match is not None
        assert match.name == "Swipe Right"
