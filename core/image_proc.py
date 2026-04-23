import cv2
import numpy as np
from numba import njit


def apply_gamma(image, gamma=1.0):
    """Apply gamma correction using a lookup table."""
    if gamma == 1.0:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_brightness(image, amount=0):
    """Adjust brightness. amount range: -100 to 100."""
    if amount == 0:
        return image
    return cv2.convertScaleAbs(image, alpha=1.0, beta=amount)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE adaptive histogram equalization."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def apply_sharpen(image, amount=1.0):
    """Unsharp mask sharpening."""
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def detect_edges(image, method='sobel'):
    """Detect edges using Sobel or Canny."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if method == 'canny':
        return cv2.Canny(gray, 100, 200)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        if np.max(magnitude) > 0:
            magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        else:
            magnitude = np.uint8(magnitude)
        return magnitude


def apply_denoise(image, amount=0):
    """Bilateral filter denoising. Handles grayscale and color."""
    if amount <= 0:
        return image
    d = max(1, int(amount * 5))
    if d % 2 == 0:
        d += 1
    sigma = 75 * amount
    if len(image.shape) == 2:
        img3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        filtered = cv2.bilateralFilter(img3, d, sigma, sigma)
        return cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(image, d, sigma, sigma)


BAYER_4X4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
]) / 16.0

_bayer_cache = {}

def apply_ordered_dithering(gray_image):
    """Apply Bayer matrix ordered dithering."""
    h, w = gray_image.shape
    
    if (h, w) not in _bayer_cache:
        bayer_tiled = np.tile(BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        _bayer_cache[(h, w)] = (bayer_tiled - 0.5) * 0.2
        
    img_norm = gray_image / 255.0
    dithered = img_norm + _bayer_cache[(h, w)]
    return np.clip(dithered * 255, 0, 255).astype(np.uint8)


@njit
def floyd_steinberg_dither(gray_image, levels=16):
    """Floyd-Steinberg error diffusion dithering with configurable quantization levels."""
    h, w = gray_image.shape
    img = gray_image.astype(np.float32)
    step = 255.0 / max(1, levels - 1)
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = np.round(old_pixel / step) * step
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if x + 1 < w: img[y, x + 1] += quant_error * 7.0 / 16.0
            if y + 1 < h:
                if x - 1 >= 0: img[y + 1, x - 1] += quant_error * 3.0 / 16.0
                img[y + 1, x] += quant_error * 5.0 / 16.0
                if x + 1 < w: img[y + 1, x + 1] += quant_error * 1.0 / 16.0
    return img


def process_image(image, gamma=1.0, brightness=0, clahe_clip=0.0, sharpen=0.0, denoise=0.0):
    """Full preprocessing pipeline. Separates and preserves alpha channel."""
    alpha = None
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            img_main = image[:, :, :3].copy()
        elif image.shape[2] == 2:
            alpha = image[:, :, 1]
            img_main = image[:, :, 0].copy()
        else:
            img_main = image.copy()
    else:
        img_main = image.copy()

    if denoise > 0:
        img_main = apply_denoise(img_main, denoise)
    if brightness != 0:
        img_main = apply_brightness(img_main, brightness)
    if gamma != 1.0:
        img_main = apply_gamma(img_main, gamma)
    if clahe_clip > 0:
        img_main = apply_clahe(img_main, clip_limit=clahe_clip)
    if sharpen > 0:
        img_main = apply_sharpen(img_main, sharpen)

    if alpha is not None:
        return np.dstack((img_main, alpha))
    return img_main
