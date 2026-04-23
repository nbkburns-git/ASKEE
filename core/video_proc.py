import cv2


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.fps == 0 or self.fps is None:
            self.fps = 30.0

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_resolution(self):
        return self.width, self.height

    def get_duration(self):
        """Duration in seconds."""
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0

    def set_position(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
