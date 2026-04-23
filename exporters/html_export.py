"""HTML export with optional per-character coloring."""
import os
import html as html_lib


def export_html(ascii_text, file_path, color_image=None, invert=False):
    """
    Export ASCII art as an HTML file.

    Args:
        ascii_text: The ASCII art string.
        file_path: Output file path.
        color_image: Optional BGR numpy array at grid resolution for per-char color.
        invert: If True, white background with dark text.
    """
    bg = "#ffffff" if invert else "#000000"
    fg = "#000000" if invert else "#ffffff"

    lines = ascii_text.split('\n')

    if color_image is not None and len(color_image.shape) == 3:
        # Per-character colored HTML
        body_lines = []
        rows, cols = color_image.shape[:2]
        for i, line in enumerate(lines):
            spans = []
            for j, char in enumerate(line):
                escaped = html_lib.escape(char)
                if i < rows and j < cols:
                    b, g, r = int(color_image[i, j, 0]), int(color_image[i, j, 1]), int(color_image[i, j, 2])
                    if invert:
                        r, g, b = 255 - r, 255 - g, 255 - b
                    spans.append(f'<span style="color:rgb({r},{g},{b})">{escaped}</span>')
                else:
                    spans.append(escaped)
            body_lines.append("".join(spans))
        pre_content = "\n".join(body_lines)
    else:
        # Monochrome — just escape HTML entities
        pre_content = html_lib.escape(ascii_text)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ASCII Art</title>
<style>
    body {{ background-color: {bg}; color: {fg}; margin: 0; display: flex;
           justify-content: center; align-items: center; min-height: 100vh; }}
    pre {{ font-family: 'Consolas', 'DejaVu Sans Mono', 'Courier New', monospace;
           font-size: 8px; line-height: 1.0; letter-spacing: 0; white-space: pre; }}
</style>
</head>
<body>
<pre>{pre_content}</pre>
</body>
</html>
"""
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
