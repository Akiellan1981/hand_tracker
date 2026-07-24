import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cursor_controller import select_all_keys


def test_select_all_keys_uses_command_on_macos(monkeypatch):
    monkeypatch.setattr("src.cursor_controller.sys.platform", "darwin")
    assert select_all_keys() == ["command", "a"]


def test_select_all_keys_uses_ctrl_elsewhere(monkeypatch):
    monkeypatch.setattr("src.cursor_controller.sys.platform", "linux")
    assert select_all_keys() == ["ctrl", "a"]

    monkeypatch.setattr("src.cursor_controller.sys.platform", "win32")
    assert select_all_keys() == ["ctrl", "a"]


def test_cursor_controller_unavailable_is_graceful(monkeypatch, capsys):
    from src.cursor_controller import CursorController

    controller = CursorController()
    # In this sandbox pyautogui has no display to attach to, so this
    # exercises the real no-pyautogui/no-display fallback path.
    if controller.available:
        return  # environment has a working pyautogui; nothing to assert here
    controller.move_to(100, 100)
    controller.mouse_down()
    controller.mouse_up()
    controller.scroll(1)
    controller.zoom(1)
    controller.hotkey("ctrl", "a")
    assert controller.screen_size() == (1920, 1080)
