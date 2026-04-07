import re
import sys
from typing import TypedDict

from textual.widgets import Static

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class StaticKwargs(TypedDict, total=False):
    expand: bool
    shrink: bool
    markup: bool
    name: str | None
    id: str | None
    classes: str | None
    disabled: bool


class PixelLlamaLoader(Static):
    """Pixelated llama loading animation using block characters"""

    def __init__(self, **kwargs: Unpack[StaticKwargs]):
        self.frame = 0
        # Pixelated llama frames using Unicode block characters
        self.frames = [
            # ── Frame 1 – all legs down (starting position) ─
            """
  ,
 ~)
 (_---;
   |~|
   | |""",
            # ── Frame 2 – lift right front leg ─
            """
  ,
 ~)
 (_---;
  /|~|
 / | |""",
            """
  ,
 ~)
 (_---;
  /|~|
  || |\\""",
            # ── Frame 3 – right front forward, lift left back ─
            """
  ,
 ~)
 (_---;
   |~|\\
   |\\| \\""",
        ]
        self.frames = [re.sub(r"^\n", "", x) for x in self.frames]

        super().__init__(self._get_display_text(), **kwargs)

    def _get_display_text(self) -> str:
        return f"{self.frames[self.frame]}"

    def on_mount(self) -> None:
        self.timer = self.set_interval(0.6, self._advance_frame)

    def _advance_frame(self) -> None:
        self.frame = (self.frame + 1) % len(self.frames)
        self.update(self._get_display_text())
