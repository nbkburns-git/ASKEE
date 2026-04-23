"""
Shared ASCII-to-image renderer.
Renders ASCII text with optional per-character coloring to a PIL Image.
Used by both the preview widget and image/video exporters.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


def render_ascii_image(ascii_text, font, char_width, char_height,
                       color_image=None, bg_color=(0, 0, 0),
                       fg_color=(255, 255, 255), invert=False,
                       transparent_bg=False):
    """
    Render ASCII text to a PIL RGB or RGBA Image.

    Two-pass approach for performance:
      1. Render all text as white-on-black (one draw.text per line).
      2. Build a color field from color_image using np.repeat (vectorized).
      3. Multiply the text alpha mask by the color field.

    Args:
        ascii_text: The ASCII art string.
        font: PIL ImageFont instance.
        char_width: Width of one character cell in pixels.
        char_height: Height of one character cell in pixels.
        color_image: Optional BGR numpy array at grid resolution (rows x cols x 3).
        bg_color: Background color tuple (R, G, B).
        fg_color: Foreground color tuple (R, G, B) used when no color_image.
        invert: If True, swap bg/fg and invert colors.
        transparent_bg: If True, returns RGBA image with transparent background.

    Returns:
        PIL Image (RGB or RGBA).
    """
    lines = ascii_text.split('\n')
    if not lines or not lines[0]:
        if transparent_bg:
            return Image.new('RGBA', (100, 100), color=(0,0,0,0))
        return Image.new('RGB', (100, 100), color=bg_color)

    cw = int(math.ceil(char_width))
    ch = int(math.ceil(char_height))
    cols = max(len(line) for line in lines)
    rows = len(lines)
    width = cols * cw
    height = rows * ch

    if width == 0 or height == 0:
        if transparent_bg:
            return Image.new('RGBA', (100, 100), color=(0,0,0,0))
        return Image.new('RGB', (100, 100), color=bg_color)

    if invert:
        bg_color, fg_color = fg_color, bg_color

    # Pass 1: Render text as white on black to get alpha mask
    text_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(text_img)
    
    w_i = font.getlength("i")
    w_m = font.getlength("M")
    is_monospace = math.isclose(w_i, w_m, abs_tol=1e-5)

    if is_monospace:
        for i, line in enumerate(lines):
            if line:
                draw.text((0, i * ch), line, font=font, fill=255)
    else:
        # Force variable-width fonts into a strict mathematical grid
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char != ' ':
                    char_w = font.getlength(char)
                    offset_x = (cw - char_w) / 2.0
                    draw.text((j * cw + offset_x, i * ch), char, font=font, fill=255)

    alpha = np.array(text_img, dtype=np.float32) / 255.0

    # Build color field
    if color_image is not None and len(color_image.shape) == 3:
        # color_image is BGR at (rows, cols, 3) — convert to RGB
        color_rgb = color_image[:, :, ::-1].astype(np.uint8)

        if invert:
            color_rgb = 255 - color_rgb

        # Expand each grid cell to pixel dimensions using np.repeat (vectorized)
        color_field = np.repeat(np.repeat(color_rgb, ch, axis=0), cw, axis=1)

        # Trim to exact image dimensions (in case of rounding)
        color_field = color_field[:height, :width]

        # Pad if needed
        if color_field.shape[0] < height or color_field.shape[1] < width:
            padded = np.zeros((height, width, 3), dtype=np.uint8)
            h_min = min(color_field.shape[0], height)
            w_min = min(color_field.shape[1], width)
            padded[:h_min, :w_min] = color_field[:h_min, :w_min]
            color_field = padded
    else:
        # Solid foreground color
        color_field = np.full((height, width, 3), fg_color, dtype=np.uint8)

    if transparent_bg:
        result_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        # Pre-multiply RGB with alpha so transparent pixels are strictly [0,0,0,0]
        result_rgba[:, :, :3] = (color_field * alpha[..., np.newaxis]).astype(np.uint8)
        result_rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
        return Image.fromarray(result_rgba, 'RGBA')
    else:
        # Composite: bg + alpha * (color - bg)
        bg_arr = np.full((height, width, 3), bg_color, dtype=np.uint8)
        result = bg_arr.astype(np.float32) + alpha[:, :, np.newaxis] * (
            color_field.astype(np.float32) - bg_arr.astype(np.float32)
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result, 'RGB')
