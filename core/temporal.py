import numpy as np

class TemporalFilter:
    def __init__(self):
        self.previous_frame = None

    def apply(self, current_frame, smoothing_factor=0.0):
        """
        Apply Exponential Moving Average (EMA) to reduce temporal flicker.
        smoothing_factor: 0.0 (no smoothing) to 0.99 (heavy smoothing).
        """
        if smoothing_factor <= 0.0 or self.previous_frame is None:
            self.previous_frame = current_frame.copy().astype(np.float32)
            return current_frame

        # Ensure shapes match (user might have changed resolution)
        if self.previous_frame.shape != current_frame.shape:
            self.previous_frame = current_frame.copy().astype(np.float32)
            return current_frame

        # Blend
        alpha = 1.0 - smoothing_factor
        blended = (current_frame.astype(np.float32) * alpha) + (self.previous_frame * smoothing_factor)
        self.previous_frame = blended
        
        return blended.astype(np.uint8)

    def reset(self):
        self.previous_frame = None
