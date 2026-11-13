#!/bin/bash
# Run the whole TennisProject pipeline on a RunPod GPU.
# On the pod:  cd /workspace && git clone https://github.com/523vishwanath/Tennis_Analysis_Project.git
#              cd Tennis_Analysis_Project && bash runpod/setup_full_pipeline.sh
set -e
cd "$(dirname "$0")/.."

echo "== System packages (OpenCV needs libGL, ffmpeg for converting videos)"
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg > /dev/null

echo "== Python packages"
pip install -q -r requirements.txt gdown
pip install -q opencv-python-headless

echo "== Model weights"
mkdir -p models wasb/pretrained_weights input_videos output_videos_test tracker_stubs
[ -f models/keypoints_model.pth ] || gdown 1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta -O models/keypoints_model.pth
if [ ! -f wasb/pretrained_weights/wasb_tennis_best.pth.tar ]; then
    cp /workspace/WASB-SBDT/pretrained_weights/wasb_tennis_best.pth.tar wasb/pretrained_weights/ 2>/dev/null \
    || gdown 14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O wasb/pretrained_weights/wasb_tennis_best.pth.tar
fi

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo "Ready. Put your video in input_videos/ and run:"
echo "  python main.py --video input_videos/inputVideoTest.mp4 --output output_videos_test/runpod_test.mp4"
