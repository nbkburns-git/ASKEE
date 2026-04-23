"""Plain text ASCII export."""
import os


def export_txt(ascii_text, file_path):
    """Save ASCII art as a plain text file."""
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(ascii_text)
