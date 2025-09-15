#!/usr/bin/env bash
python3 -m venv env/.venv
source env/.venv/bin/activate
pip install --upgrade pip wheel
pip install "torch>=2.2,<3.0" "torchvision>=0.17,<1.0" --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch==0.3.3 timm>=0.9,<1.0
pip install albumentations==1.4.7 opencv-python matplotlib pandas scikit-learn
pip install tqdm pyyaml einops
