import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.action_store import Action, ActionStore


def test_add_get_list_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "actions.json")
        store = ActionStore(path)
        action = Action(
            name="Peace Sign",
            kind="pose",
            hand="Right",
            template=[0.0, 1.0, 2.0],
            threshold=0.2,
        )
        store.add(action)

        reloaded = ActionStore(path)
        got = reloaded.get("Peace Sign")
        assert got is not None
        assert got.kind == "pose"
        assert got.hand == "Right"
        assert got.template == [0.0, 1.0, 2.0]
        assert reloaded.list()[0].name == "Peace Sign"


def test_duplicate_name_rejected_without_overwrite():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Wave", "motion", "Any", [[0.0]], 0.2))
        try:
            store.add(Action("Wave", "motion", "Any", [[1.0]], 0.3))
            assert False, "expected ValueError"
        except ValueError:
            pass
        # overwrite=True should succeed
        store.add(Action("Wave", "motion", "Any", [[1.0]], 0.3), overwrite=True)
        assert store.get("Wave").threshold == 0.3


def test_remove_action():
    with tempfile.TemporaryDirectory() as tmp:
        store = ActionStore(os.path.join(tmp, "actions.json"))
        store.add(Action("Fist Bump", "pose", "Any", [0.0], 0.1))
        store.remove("Fist Bump")
        assert store.get("Fist Bump") is None
        try:
            store.remove("Fist Bump")
            assert False, "expected KeyError"
        except KeyError:
            pass


def test_invalid_kind_rejected():
    try:
        Action("Bad", "not-a-kind", "Any", [0.0], 0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass
