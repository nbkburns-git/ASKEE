"""ANSI color text export."""


def export_ansi(ascii_text, color_image=None, invert=False):
    """
    Export ASCII text with optional ANSI 24-bit Truecolor codes.

    Args:
        ascii_text: The ASCII art string.
        color_image: Optional BGR numpy array at grid resolution (rows x cols x 3).
        invert: If True, invert colors.

    Returns:
        str: ANSI-colored text string.
    """
    if color_image is None:
        return ascii_text

    lines = ascii_text.split('\n')
    ansi_lines = []

    # Handle both color and grayscale images
    if len(color_image.shape) == 2:
        # Grayscale — no color to apply
        return ascii_text

    rows, cols = color_image.shape[:2]

    for i, line in enumerate(lines):
        if i >= rows:
            ansi_lines.append(line)
            continue
        parts = []
        for j, char in enumerate(line):
            if j >= cols:
                parts.append(char)
                continue
            b, g, r = int(color_image[i, j, 0]), int(color_image[i, j, 1]), int(color_image[i, j, 2])
            if invert:
                r, g, b = 255 - r, 255 - g, 255 - b
            parts.append(f"\033[38;2;{r};{g};{b}m{char}")
        parts.append("\033[0m")
        ansi_lines.append("".join(parts))

    return "\n".join(ansi_lines)
