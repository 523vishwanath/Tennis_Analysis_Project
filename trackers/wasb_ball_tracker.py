import importlib.util
import os
from collections import deque

import cv2
import numpy as np
import torch
import yaml

from trackers.ball_tracker import BallTracker
from utils import get_device

# WASB code lives in TennisProject/wasb (loaded by file path, so its own 'utils'/'models'
# folders never clash with the project's 'utils')
WASB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wasb')
INPUT_W, INPUT_H = 512, 288
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Config(dict):
    # dict that also allows cfg.MODEL.EXTRA style access (hrnet.py uses both)
    def __init__(self, data):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in data.items()})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def load_hrnet_class():
    spec = importlib.util.spec_from_file_location('wasb_hrnet', os.path.join(WASB_DIR, 'src', 'models', 'hrnet.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HRNet


def get_affine(frame_w, frame_h, inv=False):
    # same as WASB's get_transform: keep the aspect ratio, fit max(w, h) into 512, centered
    scale = max(frame_w, frame_h)
    center = np.array([frame_w / 2., frame_h / 2.], dtype=np.float32)
    src = np.array([center,
                    center + [0, -scale / 2.],
                    center + [-scale / 2., -scale / 2.]], dtype=np.float32)
    dst_center = np.array([INPUT_W / 2., INPUT_H / 2.], dtype=np.float32)
    dst = np.array([dst_center,
                    dst_center + [0, -INPUT_W / 2.],
                    dst_center + [-INPUT_W / 2., -INPUT_W / 2.]], dtype=np.float32)
    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


class WASBBallTracker(BallTracker):
    """
    Ball tracker using the WASB (HRNet) tennis model, which looks at 3 frames at once.
    Works as a drop-in for BallTracker: same detect_frame_candidates / interpolate /
    get_ball_shot_frames, so main.py doesn't change apart from creating this class.
    """
    def __init__(self, model_path = 'wasb/pretrained_weights/wasb_tennis_best.pth.tar', threshold = 0.5):
        self.device = get_device()
        self.threshold = threshold

        with open(os.path.join(WASB_DIR, 'src', 'configs', 'model', 'wasb.yaml')) as f:
            model_cfg = yaml.safe_load(f)
        HRNet = load_hrnet_class()
        self.model = HRNet(Config(model_cfg))
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

        self.frames_in = model_cfg['frames_in']
        self.window = deque(maxlen=self.frames_in)     # last 3 prepared frames
        self.frame_size = None

    def prepare_frame(self, frame):
        warped = cv2.warpAffine(frame, self.trans, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        return torch.from_numpy(rgb.transpose(2, 0, 1))

    def detect_frame_candidates(self, frame):
        # EVERY ball the model sees in this frame: [x1, y1, x2, y2, confidence], best first.
        # Uses this frame + the 2 frames before it (sliding window, one result per frame).
        frame_h, frame_w = frame.shape[:2]
        if self.frame_size != (frame_w, frame_h):        # new video size -> new transform
            self.frame_size = (frame_w, frame_h)
            self.trans = get_affine(frame_w, frame_h)
            self.trans_inv = get_affine(frame_w, frame_h, inv=True)
            self.box_half = max(4.0, 0.008 * frame_h)
            self.window.clear()

        self.window.append(self.prepare_frame(frame))
        frames = list(self.window)
        while len(frames) < self.frames_in:               # first frames of the video
            frames.insert(0, frames[0])

        x = torch.cat(frames, dim=0).unsqueeze(0).to(self.device)     # (1, 9, 288, 512)
        with torch.no_grad():
            heatmap = torch.sigmoid(self.model(x)[0])[0, -1].float().cpu().numpy()   # last frame = this frame
        return self.find_balls(heatmap)

    def find_balls(self, heatmap):
        candidates = []
        if heatmap.max() <= self.threshold:
            return candidates
        _, binary = cv2.threshold(heatmap, self.threshold, 1, cv2.THRESH_BINARY)
        n_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
        for label in range(1, n_labels):
            ys, xs = np.where(labels == label)
            weights = heatmap[ys, xs]
            score = float(weights.sum())
            x = float(np.sum(xs * weights) / score)
            y = float(np.sum(ys * weights) / score)
            x_img, y_img = self.trans_inv @ np.array([x, y, 1.0])
            x_img, y_img = float(x_img), float(y_img)
            candidates.append([x_img - self.box_half, y_img - self.box_half,
                               x_img + self.box_half, y_img + self.box_half,
                               float(weights.max()), score])
        candidates.sort(key=lambda c: c[5], reverse=True)
        return [c[:5] for c in candidates]

    def reset(self):
        # call before a new video when reusing the same tracker
        self.window.clear()
        self.frame_size = None
