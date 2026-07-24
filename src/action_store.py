"""Persistence for user-defined custom actions: named poses or short
motions the user records themselves, each with its own hand filter,
match threshold, and optional trigger (log / print / shell command).
Pure Python + JSON, no camera/mediapipe dependency.
"""
import json
import os
import time


class Action:
    def __init__(self, name, kind, hand, template, threshold, trigger=None, created_at=None):
        if kind not in ("pose", "motion"):
            raise ValueError("kind must be 'pose' or 'motion'")
        if hand not in ("Left", "Right", "Any"):
            raise ValueError("hand must be 'Left', 'Right', or 'Any'")
        self.name = name
        self.kind = kind
        self.hand = hand
        self.template = template
        self.threshold = threshold
        self.trigger = trigger or {"log": True, "print": True, "command": None}
        self.created_at = created_at or time.time()

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "hand": self.hand,
            "template": self.template,
            "threshold": self.threshold,
            "trigger": self.trigger,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d):
        return Action(
            name=d["name"],
            kind=d["kind"],
            hand=d["hand"],
            template=d["template"],
            threshold=d["threshold"],
            trigger=d.get("trigger"),
            created_at=d.get("created_at"),
        )


class ActionStore:
    def __init__(self, path):
        self.path = path
        self._actions = {}
        self.load()

    def load(self):
        self._actions = {}
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            raw = json.load(f)
        for item in raw.get("actions", []):
            action = Action.from_dict(item)
            self._actions[action.name] = action

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {"actions": [a.to_dict() for a in self._actions.values()]}
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.path)

    def add(self, action, overwrite=False):
        if action.name in self._actions and not overwrite:
            raise ValueError(f"an action named '{action.name}' already exists")
        self._actions[action.name] = action
        self.save()

    def remove(self, name):
        if name not in self._actions:
            raise KeyError(f"no action named '{name}'")
        del self._actions[name]
        self.save()

    def get(self, name):
        return self._actions.get(name)

    def list(self):
        return sorted(self._actions.values(), key=lambda a: a.name.lower())

    def __contains__(self, name):
        return name in self._actions
