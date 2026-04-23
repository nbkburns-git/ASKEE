"""Video export with full parameter support."""
import os
import subprocess
import cv2
import numpy as np
from PIL import ImageFont
import concurrent.futures
import multiprocessing
from core.ascii_image_renderer import render_ascii_image
from core.temporal import TemporalFilter


def export_video(video_processor, output_path, font_path, font_size, cols,
                 renderer, gamma, clahe_clip, process_image_func,
                 brightness=0, sharpen=0.0, denoise=0.0,
                 dither_mode='none', edge_mode='none',
                 invert=False, use_color=True, drop_color=None,
                 drop_tolerance=0, temporal_smooth=0.0,
                 preserve_audio=False, progress_callback=None, custom_text=""):
    """
    Export video as ASCII art video file concurrently.

    Args:
        video_processor: VideoProcessor instance.
        output_path: Output MP4 file path.
        font_path: Path to monospace font.
        font_size: Font size for rendering.
        cols: Number of ASCII columns.
        renderer: Renderer instance.
        gamma, clahe_clip, brightness, sharpen, denoise: Processing params.
        dither_mode, edge_mode: Rendering modes.
        invert: Invert colors.
        use_color: Per-character coloring.
        drop_color, drop_tolerance: Chroma key parameters.
        temporal_smooth: EMA smoothing factor.
        progress_callback: Optional callable(frame_idx, total_frames).
    """
    video_processor.set_position(0)

    font = ImageFont.truetype(font_path, font_size)
    char_width = font.getlength("M")
    ascent, descent = font.getmetrics()
    char_height = max(1, ascent + descent)
    if char_width == 0:
        char_width = 1

    import math
    cw = int(math.ceil(char_width))
    ch = int(math.ceil(char_height))

    temporal_filter = TemporalFilter()

    # Render first frame to determine output dimensions
    frame = video_processor.get_frame()
    if frame is None:
        return

    frame = temporal_filter.apply(frame, temporal_smooth)
    processed = process_image_func(frame, gamma=gamma, clahe_clip=clahe_clip,
                                   brightness=brightness, sharpen=sharpen, denoise=denoise)
    ascii_text, color_data = renderer.render_frame(processed, cols,
                                                   dither_mode=dither_mode,
                                                   edge_mode=edge_mode, invert=invert,
                                                   drop_color=drop_color, drop_tolerance=drop_tolerance,
                                                   custom_text=custom_text)

    lines = ascii_text.split('\n')
    out_cols = max(len(line) for line in lines) if lines else 1
    out_rows = len(lines)
    width = out_cols * cw
    height = out_rows * ch

    is_gif = output_path.lower().endswith(".gif")
    temp_path = output_path + ".tmp.mp4"
    
    out = None
    ffmpeg_proc = None
    pillow_frames = []
    use_pillow_fallback = False
    
    if not is_gif:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, video_processor.fps, (width, height))
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgba",
            "-r", str(video_processor.fps),
            "-i", "-",
            "-vf", "split[s0][s1];[s0]palettegen=reserve_transparent=1[p];[s1][p]paletteuse=alpha_threshold=128",
            "-gifflags", "-offsetting-transdiff",
            "-loop", "0",
            output_path
        ]
        try:
            import subprocess
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            use_pillow_fallback = True
            print("FFmpeg not found. Falling back to Pillow (high memory usage for long GIFs).")

    video_processor.set_position(0)
    temporal_filter.reset()
    total_frames = video_processor.frame_count
    
    # Thread worker function
    def render_task(smoothed_frame):
        p = process_image_func(smoothed_frame, gamma=gamma, clahe_clip=clahe_clip,
                               brightness=brightness, sharpen=sharpen, denoise=denoise)
        a_text, c_data = renderer.render_frame(p, cols,
                                               dither_mode=dither_mode,
                                               edge_mode=edge_mode, invert=invert,
                                               drop_color=drop_color, drop_tolerance=drop_tolerance,
                                               raw_image=smoothed_frame, custom_text=custom_text)
        c_arg = c_data if use_color else None
        p_img = render_ascii_image(
            a_text, font, char_width, char_height,
            color_image=c_arg, invert=invert, transparent_bg=is_gif
        )
        f_out = np.array(p_img)
        if not is_gif:
            f_out = cv2.cvtColor(f_out, cv2.COLOR_RGB2BGR)
            
        if f_out.shape[1] != width or f_out.shape[0] != height:
            f_out = cv2.resize(f_out, (width, height), interpolation=cv2.INTER_NEAREST)
        return f_out

    frame_idx = 0
    futures = {}
    
    def write_frame(frame_out):
        if not is_gif and out:
            out.write(frame_out)
        elif ffmpeg_proc:
            ffmpeg_proc.stdin.write(frame_out.tobytes())
        elif use_pillow_fallback:
            from PIL import Image
            # Ensure the frame is passed correctly
            pillow_frames.append(Image.fromarray(frame_out, 'RGBA' if is_gif else 'RGB'))

    # Use ThreadPoolExecutor for concurrent rendering
    optimal_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        while True:
            # Sequentially read and apply temporal smoothing
            frame = video_processor.get_frame()
            if frame is None:
                break
            
            smoothed = temporal_filter.apply(frame, temporal_smooth)
            
            # Submit to pool
            futures[frame_idx] = executor.submit(render_task, smoothed)
            frame_idx += 1
            
            # Keep queue bounded to avoid memory explosion
            if len(futures) > optimal_workers * 2:
                # Wait and write oldest frame
                oldest_idx = min(futures.keys())
                frame_out = futures.pop(oldest_idx).result()
                write_frame(frame_out)
                if progress_callback:
                    progress_callback(oldest_idx + 1, total_frames)

        # Drain remaining futures
        for idx in sorted(futures.keys()):
            frame_out = futures[idx].result()
            write_frame(frame_out)
            if progress_callback:
                progress_callback(idx + 1, total_frames)

    if out:
        out.release()
        
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    elif use_pillow_fallback and pillow_frames:
        dur = int(1000 / video_processor.fps) if video_processor.fps > 0 else 33
        if is_gif:
            # For GIF, we rely on Pillow's save method. Let Pillow auto-detect transparency from RGBA.
            pillow_frames[0].save(output_path, save_all=True, append_images=pillow_frames[1:], duration=dur, loop=0, disposal=2)
        else:
            pillow_frames[0].save(output_path, save_all=True, append_images=pillow_frames[1:], duration=dur, loop=0)

    # Multiplex audio for MP4
    success = False
    
    if preserve_audio and not is_gif:
        try:
            # Use ffmpeg to copy video from temp_path and audio from original video
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-i", video_processor.video_path,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:a", "aac",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                success = True
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                print(f"FFmpeg failed: {result.stderr}")
        except FileNotFoundError:
            print("FFmpeg not found in PATH. Audio will not be preserved.")

    if not success and not is_gif:
        # Fallback to just renaming the silent video
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
