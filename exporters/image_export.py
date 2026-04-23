"""Image export using the shared ASCII image renderer."""
from core.ascii_image_renderer import render_ascii_image
from PIL import ImageFont
import math


def export_image(ascii_text, file_path, font_path, font_size=12,
                 color_image=None, bg_color=(0, 0, 0), fg_color=(255, 255, 255),
                 invert=False):
    """
    Export ASCII art as a PNG image with optional per-character coloring.

    Args:
        ascii_text: The ASCII art string.
        file_path: Output file path.
        font_path: Path to the monospace font.
        font_size: Font size for rendering.
        color_image: Optional BGR numpy array at grid resolution for coloring.
        bg_color: Background color (R, G, B).
        fg_color: Foreground color (R, G, B).
        invert: If True, invert colors.
    """
    font = ImageFont.truetype(font_path, font_size)
    char_width = font.getlength("M")
    ascent, descent = font.getmetrics()
    char_height = ascent + descent

    if char_width == 0:
        char_width = 10
    if char_height == 0:
        char_height = 20

    img = render_ascii_image(
        ascii_text, font, char_width, char_height,
        color_image=color_image,
        bg_color=bg_color, fg_color=fg_color,
        invert=invert, transparent_bg=True
    )
    img.save(file_path)
