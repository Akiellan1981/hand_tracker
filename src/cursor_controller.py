"""Thin wrapper around pyautogui for OS mouse/keyboard control. Import is
lazy and guarded: on a machine without a display (or without pyautogui
installed) the rest of the app keeps working, just with cursor control
disabled and a one-time warning instead of a crash.
"""
import sys


class CursorController:
    def __init__(self):
        self._pyautogui = None
        self._available = False
        self._warned = False
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # snap mouse to a screen corner to force-abort
            pyautogui.PAUSE = 0
            self._pyautogui = pyautogui
            self._available = True
        except Exception as exc:
            self._import_error = exc

    @property
    def available(self):
        return self._available

    @property
    def failsafe_exception(self):
        """Exception type raised when the user slams the real mouse into a
        screen corner to force-abort cursor control. Returns an empty tuple
        (matches nothing) when pyautogui isn't available, so callers can
        always safely `except self.cursor.failsafe_exception:`."""
        if self._available:
            return self._pyautogui.FailSafeException
        return ()

    def screen_size(self):
        if not self._available:
            return (1920, 1080)
        return self._pyautogui.size()

    def _warn_once(self):
        if not self._warned:
            print(
                f"[cursor_controller] pyautogui unavailable ({getattr(self, '_import_error', 'unknown')}); "
                "cursor control is disabled. Install it with 'pip install pyautogui' "
                "on a machine with a display.",
                file=sys.stderr,
            )
            self._warned = True

    def move_to(self, x, y):
        if not self._available:
            return self._warn_once()
        # Let FailSafeException propagate to the caller uncaught: that's
        # pyautogui's abort signal when the mouse hits a screen corner, and
        # the app loop is responsible for turning that into a graceful
        # "cursor control disabled" rather than a crash.
        self._pyautogui.moveTo(x, y, _pause=False)

    def mouse_down(self, button="left"):
        if not self._available:
            return self._warn_once()
        self._pyautogui.mouseDown(button=button)

    def mouse_up(self, button="left"):
        if not self._available:
            return self._warn_once()
        self._pyautogui.mouseUp(button=button)

    def double_click(self, x, y):
        if not self._available:
            return self._warn_once()
        self._pyautogui.doubleClick(x=x, y=y)

    def hotkey(self, *keys):
        if not self._available:
            return self._warn_once()
        self._pyautogui.hotkey(*keys)

    def scroll(self, ticks):
        if not self._available:
            return self._warn_once()
        self._pyautogui.scroll(ticks)

    def zoom(self, ticks):
        """Simulates a held-Ctrl scroll, the conventional zoom shortcut in
        browsers, image viewers, maps, and most IDEs."""
        if not self._available:
            return self._warn_once()
        self._pyautogui.keyDown("ctrl")
        try:
            self._pyautogui.scroll(ticks)
        finally:
            self._pyautogui.keyUp("ctrl")


def select_all_keys():
    return ["command", "a"] if sys.platform == "darwin" else ["ctrl", "a"]
