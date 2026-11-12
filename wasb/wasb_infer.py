"""
WASB ball detection for the TennisProject.

Runs the WASB (HRNet) tennis model on a video and saves the ball
candidates in the same format as tracker_stubs/ball_candidates.pkl:
    a list with one entry per frame, each entry a list of [x1, y1, x2, y2, conf]

Usage (inside the WASB-SBDT repo folder):
    python wasb_infer.py --video swingVideo.mp4 \
                         --weights pretrained_weights/wasb_tennis_best.pth.tar \
                         --out ball_candidates.pkl \
                         --preview preview.mp4
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import yaml

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, REPO_SRC)
from models.hrnet import HRNet                    # noqa: E402
from utils.image import get_affine_transform      # noqa: E402

INPUT_W, INPUT_H = 512, 288
FRAMES_IN = 3
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_transform(frame_w, frame_h, inv=0):
    center = np.array([frame_w / 2., frame_h / 2.], dtype=np.float32)
    scale = max(frame_h, frame_w) * 1.0
    return get_affine_transform(center, scale, 0, [INPUT_W, INPUT_H], inv=inv)


class Config(dict):
    """dict that also allows cfg.MODEL.EXTRA style access (hrnet.py uses both)."""
    def __init__(self, data):
        super().__init__({k: Config(v) if isinstance(v, dict) else v for k, v in data.items()})

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def load_model(weights_path, device):
    cfg_path = os.path.join(REPO_SRC, 'configs', 'model', 'wasb.yaml')
    with open(cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    model = HRNet(Config(model_cfg))
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        # weights saved from nn.DataParallel have 'module.' in front of every key
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        print("WARNING: no weights given, running with random weights (test only)")
    return model.to(device).eval()


def prepare_frame(frame, trans):
    warped = cv2.warpAffine(frame, trans, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))          # (3, H, W)


def find_balls(heatmap, trans_inv, threshold, box_half):
    """Every blob above the threshold becomes one candidate box."""
    candidates = []
    if heatmap.max() <= threshold:
        return candidates
    _, binary = cv2.threshold(heatmap, threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        weights = heatmap[ys, xs]
        score = float(weights.sum())
        x = float(np.sum(xs * weights) / score)
        y = float(np.sum(ys * weights) / score)
        # back from the 512x288 heatmap to the original video pixels
        x_img, y_img = trans_inv @ np.array([x, y, 1.0])
        confidence = float(weights.max())
        x_img, y_img = float(x_img), float(y_img)
        candidates.append([x_img - box_half, y_img - box_half,
                           x_img + box_half, y_img + box_half,
                           confidence, score])
    # best blob first
    candidates.sort(key=lambda c: c[5], reverse=True)
    return [c[:5] for c in candidates]


def run(args):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Device:", device)
    model = load_model(args.weights, device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f} fps, {total} frames")

    trans = get_transform(frame_w, frame_h)
    trans_inv = get_transform(frame_w, frame_h, inv=1)
    box_half = max(4.0, 0.008 * frame_h)

    writer = None
    if args.preview:
        writer = cv2.VideoWriter(args.preview, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (frame_w, frame_h))

    ball_candidates = []
    window_frames, window_tensors = [], []
    frame_index = 0

    def process_window(n_real=FRAMES_IN):
        x = torch.cat(window_tensors, dim=0).unsqueeze(0).to(device)   # (1, 9, H, W)
        with torch.no_grad():
            heatmaps = torch.sigmoid(model(x)[0])[0].float().cpu().numpy()  # (3, H, W)
        for i, frame in enumerate(window_frames[:n_real]):
            candidates = find_balls(heatmaps[i], trans_inv, args.threshold, box_half)
            ball_candidates.append(candidates)
            if writer is not None:
                for j, (x1, y1, x2, y2, conf) in enumerate(candidates):
                    color = (0, 255, 255) if j == 0 else (0, 128, 255)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), int(box_half) + 4, color, 2)
                    cv2.putText(frame, f"{conf:.2f}", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                writer.write(frame)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window_frames.append(frame)
        window_tensors.append(prepare_frame(frame, trans))
        frame_index += 1
        if len(window_frames) == FRAMES_IN:
            process_window()
            window_frames, window_tensors = [], []
        if frame_index % 300 == 0:
            print(f"  {frame_index}/{total} frames")

    # last 1-2 frames: pad the window by repeating the final frame
    if window_frames:
        real = len(window_frames)
        while len(window_tensors) < FRAMES_IN:
            window_tensors.append(window_tensors[-1])
        process_window(real)

    cap.release()
    if writer is not None:
        writer.release()

    with open(args.out, 'wb') as f:
        pickle.dump(ball_candidates, f)

    found = sum(1 for c in ball_candidates if c)
    print(f"Saved {args.out}: ball found in {found}/{len(ball_candidates)} frames "
          f"({100 * found / max(1, len(ball_candidates)):.0f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--weights', default='pretrained_weights/wasb_tennis_best.pth.tar',
                        help="use 'none' to test with random weights")
    parser.add_argument('--out', default='ball_candidates.pkl')
    parser.add_argument('--preview', default=None, help='optional output video with the ball drawn')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    if args.weights.lower() == 'none':
        args.weights = None
    run(args)
