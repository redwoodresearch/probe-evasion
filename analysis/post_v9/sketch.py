"""Hand-drawn ("Excalidraw-like") look for the post's figures: wobbly lines from matplotlib's xkcd mode, a handwriting
font that ships with macOS, no white text outlines. Use as a context manager around any figure code:

    with sketch():
        draw_the_figure()
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# Excalidraw's own handwriting font, Virgil (SIL Open Font License; fonts/Virgil_LICENSE.md), converted from the
# published Virgil.woff2 to TTF with fontTools. Chalkboard SE (macOS) fills any glyph Virgil lacks.
VIRGIL = Path(__file__).resolve().parent / "fonts" / "Virgil.ttf"
font_manager.fontManager.addfont(str(VIRGIL))
FONT = [font_manager.FontProperties(fname=str(VIRGIL)).get_name(), "Chalkboard SE"]
# Excalidraw's pastel fills and near-black stroke
FILLS = {"yellow": "#fff3bf", "green": "#b2f2bb", "red": "#ffc9c9", "blue": "#d0ebff", "grey": "#e9ecef", "orange": "#ffd8a8"}
STROKE = "#1e1e1e"


@contextmanager
def sketch(scale: float = 0.8, length: float = 150, randomness: float = 1.5):
    with plt.xkcd(scale=scale, length=length, randomness=randomness):
        plt.rcParams.update({"font.family": FONT, "path.effects": [], "axes.linewidth": 1.4, "font.size": 13})
        yield
