<div align="center">

```
    ___   _____ __ __  ___________
   /   | / ___// //_/ / ____/ ____/
  / /| | \__ \/ ,<   / __/ / __/
 / ___ |___/ / /| | / /___/ /___
/_/  |_/____/_/ |_|/_____/_____/
```

# ASKEE

**ASCII System & Encoding Engine**

A high-performance, GPU-accelerated image and video-to-ASCII converter with a real-time GUI, professional-grade rendering pipeline, and multi-format export.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41cd52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)]()

</div>

---

<img width="2298" height="1652" alt="Screenshot 2026-04-23 221644" src="https://github.com/user-attachments/assets/f22962a8-05e2-476b-b575-8c055191ddda" />
<img width="3200" height="4440" alt="6" src="https://github.com/user-attachments/assets/078420fd-0a31-4cdd-933b-fd6946a0ea22" />
<img width="1200" height="1200" alt="2" src="https://github.com/user-attachments/assets/22231506-0100-4465-851e-09e84fa80aea" />

---

## ✨ Features

### 🎨 Rendering Engine
- **Perceptual character mapping** — Characters sorted by actual ink density, not arbitrary ordering
- **30+ built-in character sets** — Standard, Block, Braille, Dense, Retro, and more
- **Custom character sets** — Define your own ramps or fill the grid with custom text
- **Dithering** — Ordered (Bayer 4×4) and Floyd-Steinberg error-diffusion
- **Edge detection** — Sobel-based edge overlay with hybrid and edge-only modes
- **JIT-compiled rendering** via Numba (`@njit` with `parallel=True`) for near-native speed

### 🖼️ Image Processing Pipeline
- **Gamma correction** with LUT-based acceleration
- **Brightness & contrast** (CLAHE adaptive histogram equalization)
- **Sharpening** (unsharp mask) and **denoising** (bilateral filter)
- **Chroma key** background removal with eyedropper tool and adjustable tolerance
- **Invert mode** for light-on-dark or dark-on-light output
- **Alpha channel preservation** throughout the entire pipeline

### 🎬 Video Support
- **Real-time playback** with live ASCII preview
- **Temporal coherence** — Exponential Moving Average (EMA) smoothing to reduce flicker
- **Multi-threaded export** — Concurrent frame rendering using `ThreadPoolExecutor`
- **Audio preservation** via FFmpeg muxing
- **GIF export** with transparency support (FFmpeg palette generation or Pillow fallback)

### 🔤 Typography
- **System font scanning** — Automatically detects all installed `.ttf` / `.otf` fonts
- **Monospace & proportional font support** — Variable-width fonts are force-fitted to a strict grid
- **Adjustable preview font size** (4–20px)
- **Per-character coloring** — Each ASCII character inherits the color of its source pixel

### 📦 Export Formats
| Format | Description |
|--------|-------------|
| **TXT** | Plain-text ASCII art |
| **HTML** | Styled `<pre>` block with optional per-character `<span>` coloring |
| **ANSI** | 24-bit Truecolor escape sequences for terminal display |
| **PNG** | High-resolution rendered image with transparent background support |
| **MP4** | Full video export with configurable resolution and audio |
| **GIF** | Animated GIF with optional transparency |

### ⚙️ Presets
Quickly switch between tuned configurations:

| Preset | Columns | Description |
|--------|---------|-------------|
| Default | 150 | Balanced starting point |
| High Quality | 250 | CLAHE + Floyd-Steinberg dithering + sharpening |
| Fast | 80 | Low-resolution for quick previews |
| Edge Heavy | 150 | Sobel edge overlay with sharpening |
| High Contrast | 150 | CLAHE + ordered dithering + gamma boost |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **FFmpeg** (optional, for video audio preservation and GIF export)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ASKEE.git
cd ASKEE

# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python src/main.py
```

Or on Windows, simply double-click **`run.bat`**.

---

## 🏗️ Architecture

```
ASKEE/
├── src/
│   ├── main.py                          # Entry point
│   ├── core/
│   │   ├── renderer.py                  # Numba-accelerated ASCII mapping & edge overlay
│   │   ├── ascii_image_renderer.py      # Two-pass text-to-image rasterizer
│   │   ├── font_manager.py              # Font discovery, metrics, density ramp generation
│   │   ├── image_proc.py                # Gamma, CLAHE, sharpen, denoise, dithering
│   │   ├── video_proc.py                # OpenCV video capture wrapper
│   │   └── temporal.py                  # EMA temporal smoothing filter
│   ├── exporters/
│   │   ├── txt_export.py                # Plain text
│   │   ├── html_export.py               # Colored HTML
│   │   ├── ansi_export.py               # ANSI Truecolor escape codes
│   │   ├── image_export.py              # High-res PNG rendering
│   │   └── video_export.py              # Concurrent MP4/GIF export with FFmpeg
│   └── ui/
│       └── main_window.py               # PySide6 GUI (controls, preview, export bar)
├── requirements.txt
└── run.bat
```

### Rendering Pipeline

```
Input Image/Frame
       │
       ▼
┌──────────────────┐
│  Image Processing │  ← Gamma, Brightness, CLAHE, Sharpen, Denoise
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Resize to Grid   │  ← (cols × rows) using aspect-corrected dimensions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Dithering        │  ← Ordered (Bayer) or Floyd-Steinberg
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Luminance → ASCII│  ← Density-sorted character ramp (Numba JIT)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Edge Overlay     │  ← Sobel gradient → directional characters ( | - / \ )
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Chroma Key Mask  │  ← Background removal via color distance thresholding
└────────┬─────────┘
         │
         ▼
      Output
  (Text + Color Map)
```

---

## 📋 Dependencies

| Package | Purpose |
|---------|---------|
| [PySide6](https://pypi.org/project/PySide6/) | Qt6 GUI framework |
| [OpenCV](https://pypi.org/project/opencv-python/) | Image/video I/O and processing |
| [NumPy](https://pypi.org/project/numpy/) | Array operations |
| [Numba](https://pypi.org/project/numba/) | JIT compilation for hot loops |
| [Pillow](https://pypi.org/project/Pillow/) | Font rendering and image export |
| [ffmpeg-python](https://pypi.org/project/ffmpeg-python/) | FFmpeg integration |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ and way too many ASCII characters.**

</div>
