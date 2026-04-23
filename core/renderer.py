import numpy as np
import cv2
from numba import njit, prange
from core.image_proc import apply_ordered_dithering, floyd_steinberg_dither


@njit(parallel=True)
def map_luminance_to_ascii(gray_image, ramp_array):
    """Map each pixel's luminance to a character codepoint from the ramp. Parallelized."""
    h, w = gray_image.shape
    out = np.zeros((h, w), dtype=np.uint32)
    n_chars = len(ramp_array)

    for i in prange(h):
        for j in range(w):
            val = gray_image[i, j]
            idx = int((val / 255.0) * (n_chars - 1))
            out[i, j] = ramp_array[idx]

    return out


@njit(parallel=True)
def apply_edge_overlay(codepoints, magnitude, angle, threshold, edge_only):
    """Vectorized edge character overlay using Numba."""
    h, w = codepoints.shape
    pipe = np.uint32(ord('|'))
    dash = np.uint32(ord('-'))
    fslash = np.uint32(ord('/'))
    bslash = np.uint32(ord('\\'))
    space = np.uint32(ord(' '))

    for i in prange(h):
        for j in range(w):
            if magnitude[i, j] > threshold:
                a = angle[i, j]
                if a < 0:
                    a += 180.0
                if (a >= 0 and a < 22.5) or (a >= 157.5 and a <= 180):
                    codepoints[i, j] = pipe
                elif a >= 22.5 and a < 67.5:
                    codepoints[i, j] = bslash
                elif a >= 67.5 and a < 112.5:
                    codepoints[i, j] = dash
                else:
                    codepoints[i, j] = fslash
            elif edge_only:
                codepoints[i, j] = space
    return codepoints




class Renderer:
    def __init__(self, font_manager):
        self.font_manager = font_manager
        self.ramp = ""
        self.ramp_array = np.array([], dtype=np.uint32)

    def set_ramp(self, charset):
        """Generate and cache the density-sorted character ramp."""
        self.ramp = self.font_manager.generate_ramp(charset)
        self.ramp_array = np.array([ord(c) for c in self.ramp], dtype=np.uint32)

    def render_frame(self, image, cols, rows=None, dither_mode='none',
                     edge_mode='none', invert=False, drop_color=None, drop_tolerance=0, raw_image=None, custom_text=""):
        """
        Render an image to ASCII text.

        Returns:
            tuple: (ascii_text, color_image_at_grid_resolution)
                   color_image is BGR at (rows, cols) for per-character coloring.
        """
        if len(self.ramp_array) == 0:
            return "", image

        h, w = image.shape[:2]

        if rows is None:
            rows = int((cols * self.font_manager.char_width * h) /
                       (w * self.font_manager.char_height))
        rows = max(1, rows)
        cols = max(1, cols)

        resized = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)

        # Extract color and grayscale
        if len(resized.shape) == 3:
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3]
                color = resized[:, :, :3]
                gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
            elif resized.shape[2] == 2:
                alpha = resized[:, :, 1]
                color = cv2.cvtColor(resized[:, :, 0], cv2.COLOR_GRAY2BGR)
                gray = resized[:, :, 0]
            else:
                alpha = None
                color = resized.copy()
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            alpha = None
            color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = resized.copy()

        # Invert luminance if requested
        if invert:
            gray = 255 - gray

        # Apply dithering
        if dither_mode == 'ordered':
            gray = apply_ordered_dithering(gray)
        elif dither_mode == 'floyd-steinberg':
            gray = floyd_steinberg_dither(gray, levels=len(self.ramp_array)).astype(np.uint8)

        if custom_text:
            text_len = len(custom_text)
            total_chars = rows * cols
            # Repeat the custom text to fill the grid
            repeated = (custom_text * ((total_chars // text_len) + 1))[:total_chars]
            codepoints = np.array([ord(c) for c in repeated], dtype=np.uint32).reshape((rows, cols))
        else:
            # Map luminance to ASCII characters
            ramp = self.ramp_array
            if invert:
                ramp = ramp[::-1].copy()
            codepoints = map_luminance_to_ascii(gray, ramp)

        # Edge detection overlay (vectorized)
        if edge_mode != 'none':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            angle = np.arctan2(sobely, sobelx) * 180 / np.pi
            edge_only = edge_mode == 'edge_only'
            codepoints = apply_edge_overlay(
                codepoints, magnitude, angle, 50.0, edge_only
            )

        # Chroma Key masking (vectorized)
        if drop_color is not None and drop_tolerance > 0:
            if alpha is None:
                alpha = np.full(gray.shape, 255, dtype=np.uint8)
                
            threshold_dist = (drop_tolerance / 100.0) * 442.0
                
            if raw_image is not None:
                raw_color = raw_image[:, :, :3] if len(raw_image.shape) == 3 else cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
                dist = np.linalg.norm(raw_color.astype(np.float32) - np.array(drop_color, dtype=np.float32), axis=2)
                raw_mask = np.where(dist <= threshold_dist, 0, 255).astype(np.uint8)
                mask_resized = cv2.resize(raw_mask, (cols, rows), interpolation=cv2.INTER_AREA)
                alpha = np.where(mask_resized < 128, 0, alpha)
            else:
                dist = np.linalg.norm(color.astype(np.float32) - np.array(drop_color, dtype=np.float32), axis=2)
                alpha = np.where(dist <= threshold_dist, 0, alpha)

        if alpha is not None:
            space_ord = np.uint32(ord(' '))
            codepoints = np.where(alpha < 128, space_ord, codepoints)

        # Build text output (vectorized)
        lines = []
        for i in range(rows):
            line = "".join(chr(c) for c in codepoints[i])
            lines.append(line)

        return "\n".join(lines), color
