import numpy as np
import platform
import os
import math
from PIL import Image, ImageFont, ImageDraw


def find_monospace_font():
    """Find a suitable monospace font for the current platform."""
    system = platform.system()
    candidates = []
    if system == "Windows":
        font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        candidates = [
            os.path.join(font_dir, "consola.ttf"),
            os.path.join(font_dir, "cour.ttf"),
            os.path.join(font_dir, "lucon.ttf"),
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.dfont",
            "/Library/Fonts/Courier New.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

class FontManager:
    @staticmethod
    def get_system_fonts():
        """Scan OS font directories and return a dict mapping nice names to file paths."""
        system = platform.system()
        directories = []
        if system == "Windows":
            windir = os.environ.get("WINDIR", "C:\\Windows")
            directories.append(os.path.join(windir, "Fonts"))
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            if local_appdata:
                directories.append(os.path.join(local_appdata, "Microsoft\\Windows\\Fonts"))
        elif system == "Darwin":
            directories.extend([
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")
            ])
        else:
            directories.extend([
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts")
            ])

        fonts = {}
        for d in directories:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                for file in files:
                    ext = file.lower()
                    if ext.endswith('.ttf') or ext.endswith('.otf'):
                        path = os.path.join(root, file)
                        try:
                            font = ImageFont.truetype(path, 10)
                            family, style = font.getname()
                            name = f"{family} {style}" if style and style.lower() not in ["regular", "normal"] else family
                            if name not in fonts:
                                fonts[name] = path
                        except Exception:
                            name = os.path.splitext(file)[0].replace('-', ' ').title()
                            if name not in fonts:
                                fonts[name] = path

        return {k: fonts[k] for k in sorted(fonts.keys())}

    def __init__(self, font_path=None, font_size=20):
        if font_path is None:
            font_path = find_monospace_font()

        self.font_path = font_path
        self.font_size = font_size

        if font_path and os.path.exists(font_path):
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            self.font = ImageFont.load_default()
            self.font_path = None

        self._calculate_metrics()
        self._ramp_cache = {}
    def set_font(self, font_path, font_size=None):
        """Update the active font, recalculate metrics, and clear the ramp cache."""
        self.font_path = font_path
        if font_size is not None:
            self.font_size = font_size
            
        if font_path and os.path.exists(font_path):
            self.font = ImageFont.truetype(font_path, self.font_size)
        else:
            self.font = ImageFont.load_default()
            self.font_path = None
            
        self._calculate_metrics()
        self._ramp_cache = {}

    def _calculate_metrics(self):
        """Calculate character cell dimensions using precise font metrics."""
        w_i = self.font.getlength("i")
        w_m = self.font.getlength("M")
        self.is_monospace = math.isclose(w_i, w_m, abs_tol=1e-5)

        if self.is_monospace:
            width = w_m
        else:
            # Average width for better grid sizing
            sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:-=+*#%@"
            width = sum(self.font.getlength(c) for c in sample) / len(sample)

        ascent, descent = self.font.getmetrics()
        height = ascent + descent

        if width == 0 or height == 0:
            width, height = 10, 20

        self.char_width = width
        self.char_height = height
        self.aspect_ratio = width / height

    def generate_ramp(self, charset):
        """
        Calculate ink density of each character and return sorted string
        from least dense (darkest) to most dense (brightest).
        Results are cached per charset string.
        """
        if charset in self._ramp_cache:
            return self._ramp_cache[charset]

        densities = []
        cell_w = int(np.ceil(self.char_width))
        cell_h = int(np.ceil(self.char_height))

        for char in charset:
            img = Image.new('L', (cell_w * 2, cell_h * 2), color=0)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, font=self.font, fill=255)
            img = img.crop((0, 0, cell_w, cell_h))
            density = np.mean(np.array(img))
            densities.append((char, density))

        densities.sort(key=lambda x: x[1])
        ramp = "".join([c for c, d in densities])

        self._ramp_cache[charset] = ramp
        return ramp


# --- Existing ---
CHARSET_STANDARD = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
CHARSET_SIMPLE = " .:-=+*#%@"
CHARSET_BLOCK = " ░▒▓█"
CHARSET_MINIMAL = ",;:+&X"
CHARSET_DETAILED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|)(1}{][?-_+~<>i!lI;:,\"^`'. "

# --- Balanced / Practical ---
CHARSET_BALANCED = " .,:;irsXA253hMHGS#9B&@"
CHARSET_MEDIUM = " .,:;i+=ox%#@"
CHARSET_SOFT = " .,'`:"
CHARSET_HARD = " .:-=+*#MW&"

# --- High Detail Variants ---
CHARSET_EXTREME_DETAIL = " .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
CHARSET_DENSE = "`.-':_,^=;><+!rc*/z?sLTv)J7(|F{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VOGbUAKXHm8RD#$Bg0MNWQ%&@"

# --- Minimal / Performance ---
CHARSET_TINY = " .#@"
CHARSET_BINARY = " @"
CHARSET_LOWFI = " .-#"

# --- Unicode / High Fidelity ---
CHARSET_BLOCK_FULL = " ▁▂▃▄▅▆▇█"
CHARSET_BLOCK_VERTICAL = " ▏▎▍▌▋▊▉█"
CHARSET_BRAILLE = " ⠀⠁⠃⠇⠧⠷⠿⣿"

# --- Edge / Line Rendering ---
CHARSET_EDGE = " .-~=*#"
CHARSET_LINES = " /|\\-_"
CHARSET_BOX = " ─│┌┐└┘┼"

# --- Stylized ---
CHARSET_RETRO = " .:-=+*#%@"
CHARSET_HEAVY = " .:-=+*#WMB8&"
CHARSET_WEIRD = "@#W$9876543210?!abc;:+=-,._"

# --- Numeric Bias ---
CHARSET_NUMERIC = " .1234567890@"
CHARSET_DIGITS = " 0123456789"

# --- Restricted / Themed ---
CHARSET_X_SET = " ,;:+&X"
CHARSET_SYMBOLS = " .,:;+=*#@"
CHARSET_ALPHA = " abcdefghijklmnopqrstuvwxyz"
CHARSET_UPPER = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- Contrast Variants ---
CHARSET_LOW_CONTRAST = " .,:;i"
CHARSET_MID_CONTRAST = " .:-=+*"
CHARSET_HIGH_CONTRAST = " .#@"

# --- Experimental ---
CHARSET_SMOOTH = " .,:-=+*#%@"
CHARSET_SHARP = " `^\";!i><~+_-?][}{1)(|\\/"
CHARSET_RANDOM_STYLE = " qwertyuiopasdfghjklzxcvbnm#@"


# Registry mapping display name -> charset string (used by UI)
CHARSET_REGISTRY = {
    "Standard": CHARSET_STANDARD,
    "Simple": CHARSET_SIMPLE,
    "Block": CHARSET_BLOCK,
    "Minimal": CHARSET_MINIMAL,
    "Detailed": CHARSET_DETAILED,
    "Balanced": CHARSET_BALANCED,
    "Medium": CHARSET_MEDIUM,
    "Soft": CHARSET_SOFT,
    "Hard": CHARSET_HARD,
    "Extreme Detail": CHARSET_EXTREME_DETAIL,
    "Dense": CHARSET_DENSE,
    "Tiny": CHARSET_TINY,
    "Binary": CHARSET_BINARY,
    "Lo-Fi": CHARSET_LOWFI,
    "Block Full": CHARSET_BLOCK_FULL,
    "Block Vertical": CHARSET_BLOCK_VERTICAL,
    "Braille": CHARSET_BRAILLE,
    "Edge": CHARSET_EDGE,
    "Lines": CHARSET_LINES,
    "Box": CHARSET_BOX,
    "Retro": CHARSET_RETRO,
    "Heavy": CHARSET_HEAVY,
    "Weird": CHARSET_WEIRD,
    "Numeric": CHARSET_NUMERIC,
    "Digits": CHARSET_DIGITS,
    "X Set": CHARSET_X_SET,
    "Symbols": CHARSET_SYMBOLS,
    "Alpha": CHARSET_ALPHA,
    "Upper": CHARSET_UPPER,
    "Low Contrast": CHARSET_LOW_CONTRAST,
    "Mid Contrast": CHARSET_MID_CONTRAST,
    "High Contrast": CHARSET_HIGH_CONTRAST,
    "Smooth": CHARSET_SMOOTH,
    "Sharp": CHARSET_SHARP,
    "Random Style": CHARSET_RANDOM_STYLE,
}
