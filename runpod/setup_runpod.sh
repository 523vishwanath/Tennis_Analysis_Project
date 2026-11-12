#!/bin/bash
# Run once on the RunPod pod:  bash setup_runpod.sh
set -e
cd /workspace
if [ ! -d WASB-SBDT ]; then
    git clone https://github.com/nttcom/WASB-SBDT.git
fi
cd WASB-SBDT
pip install -q opencv-python-headless pyyaml einops timm gdown pandas tqdm pillow scipy scikit-learn scikit-image matplotlib hydra-core omegaconf
mkdir -p pretrained_weights
if [ ! -f pretrained_weights/wasb_tennis_best.pth.tar ]; then
    gdown 14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z -O pretrained_weights/wasb_tennis_best.pth.tar
fi
cp /workspace/wasb_infer.py . 2>/dev/null || true
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo "Ready. Run: python wasb_infer.py --video /workspace/swingVideo.mp4 --out /workspace/ball_candidates.pkl --preview /workspace/wasb_preview.mp4"
