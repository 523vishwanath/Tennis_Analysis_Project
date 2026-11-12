import cv2
import os
import shutil
import subprocess

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 30
    height, width, _ = output_video_frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps



def convert_to_h264(video_path):
    """Re-encode a video to H.264 (plays in every browser and in QuickTime). Returns True if it worked."""
    try:
        import imageio_ffmpeg                      # pip install imageio-ffmpeg (brings its own ffmpeg)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print("  (no ffmpeg found, video kept as mp4v: install imageio-ffmpeg to play it in a browser)")
        return False

    temp_path = video_path + '.h264.mp4'
    command = [ffmpeg, '-y', '-loglevel', 'error', '-i', video_path,
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               temp_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(temp_path):
        print("  (h264 conversion failed:", result.stderr.strip()[:200], ")")
        return False
    os.replace(temp_path, video_path)
    return True



def get_device():
    # Apple GPU (mps) on a Mac, NVIDIA GPU (cuda) on a PC, otherwise the CPU
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


def read_frames_at(video_path, frame_numbers):
    # only the frames we need (instead of the whole video in memory)
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames
