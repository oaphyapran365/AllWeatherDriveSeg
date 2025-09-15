# src/infer.py
import os, sys, argparse, glob
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from augments import get_val_aug
from train_baseline import get_model  # same model factory as training

# -------- Cityscapes trainId color palette (19 classes) ----------
CITYSCAPES_TRAINID_PALETTE = np.array([
    [128,  64,128],  # 0  road
    [244,  35,232],  # 1  sidewalk
    [ 70,  70, 70],  # 2  building
    [102, 102,156],  # 3  wall
    [190, 153,153],  # 4  fence
    [153, 153,153],  # 5  pole
    [250, 170, 30],  # 6  traffic light
    [220, 220,  0],  # 7  traffic sign
    [107, 142, 35],  # 8  vegetation
    [152, 251,152],  # 9  terrain
    [ 70, 130,180],  # 10 sky
    [220,  20, 60],  # 11 person
    [255,   0,  0],  # 12 rider
    [  0,  0,142],  # 13 car
    [  0,  0, 70],  # 14 truck
    [  0, 60,100],  # 15 bus
    [  0, 80,100],  # 16 train
    [  0,  0,230],  # 17 motorcycle
    [119, 11, 32],  # 18 bicycle
], dtype=np.uint8)
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # for 255

def colorize_trainids(mask_hw: np.ndarray) -> np.ndarray:
    """mask_hw: HxW np.uint8 with values in {0..18} or 255."""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask_hw != 255
    out[valid] = CITYSCAPES_TRAINID_PALETTE[mask_hw[valid]]
    out[~valid] = IGNORE_COLOR
    return out

def overlay(img_bgr: np.ndarray, mask_rgb: np.ndarray, alpha=0.5) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(img_bgr, 1 - alpha, mask_bgr, alpha, 0)

def load_model(ckpt_path: str, n_classes=19, encoder_name="resnet101", device="cuda"):
    try:
        model = get_model(n_classes, encoder_name=encoder_name)
    except TypeError:
        model = get_model(n_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def forward_image(model, img_bgr: np.ndarray, pad_h=1088, pad_w=2048, device="cuda"):
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aug = get_val_aug((pad_h, pad_w))
    sample = aug(image=img_rgb, mask=np.zeros((h0, w0), dtype=np.uint8))  # dummy mask for shape
    tensor = sample["image"].unsqueeze(0).to(device)  # [1,3,H,W]
    logits = model(tensor)                            # [1,C,H,W]
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()  # HxW trainIds
    # crop back to original (we padded)
    pred = pred[:h0, :w0]
    return pred

def bn_adapt_on_images(model, imgs_bgr, pad_h, pad_w, device="cuda", iters=1, bs=4):
    """Update BN stats using unlabeled target images."""
    # put only BN layers in train mode (others eval)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.training = True; m.track_running_stats = True
        else:
            m.training = False
    aug = get_val_aug((pad_h, pad_w))
    with torch.no_grad():
        for _ in range(iters):
            # mini-batches
            for i in range(0, len(imgs_bgr), bs):
                batch = imgs_bgr[i:i+bs]
                tensors = []
                for img_bgr in batch:
                    h0, w0 = img_bgr.shape[:2]
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    sample = aug(image=img_rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
                    tensors.append(sample["image"])
                x = torch.stack(tensors, dim=0).to(device)
                _ = model(x)
    model.eval()

def is_image(path):
    ext = Path(path).suffix.lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pth")
    ap.add_argument("--input", required=True, help="image file | folder | video file")
    ap.add_argument("--out", required=True, help="output file or folder")
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--pad_h", type=int, default=1088)
    ap.add_argument("--pad_w", type=int, default=2048)
    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--bn_adapt_iters", type=int, default=0, help=">0 to adapt BN on inputs before predicting")
    ap.add_argument("--bn_bs", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True) if not os.path.splitext(args.out)[1] else os.makedirs(os.path.dirname(args.out), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, encoder_name=args.encoder, device=device)

    inp = args.input
    if os.path.isdir(inp):
        # ---------- Folder of images ----------
        imgs = sorted([p for p in glob.glob(os.path.join(inp, "*")) if is_image(p)])
        if len(imgs) == 0:
            raise FileNotFoundError(f"No images found in {inp}")
        # Optional BN-Adapt over all images
        if args.bn_adapt_iters > 0:
            imgs_bgr = [cv2.imread(p, cv2.IMREAD_COLOR) for p in imgs]
            bn_adapt_on_images(model, imgs_bgr, args.pad_h, args.pad_w, device=device, iters=args.bn_adapt_iters, bs=args.bn_bs)
        for p in tqdm(imgs, desc="Infer (images)"):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            pred = forward_image(model, bgr, args.pad_h, args.pad_w, device)
            mask_rgb = colorize_trainids(pred)
            over = overlay(bgr, mask_rgb, alpha=args.overlay_alpha)
            base = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(args.out, f"{base}_mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.out, f"{base}_overlay.png"), over)

    elif is_image(inp):
        # ---------- Single image ----------
        bgr = cv2.imread(inp, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {inp}")
        if args.bn_adapt_iters > 0:
            bn_adapt_on_images(model, [bgr], args.pad_h, args.pad_w, device=device, iters=args.bn_adapt_iters, bs=1)
        pred = forward_image(model, bgr, args.pad_h, args.pad_w, device)
        mask_rgb = colorize_trainids(pred)
        over = overlay(bgr, mask_rgb, alpha=args.overlay_alpha)
        root, ext = os.path.splitext(args.out)
        if ext == "":  # treat as folder
            os.makedirs(args.out, exist_ok=True)
            base = Path(inp).stem
            cv2.imwrite(os.path.join(args.out, f"{base}_mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.out, f"{base}_overlay.png"), over)
        else:
            # if a file is provided, write overlay there and a sibling mask
            cv2.imwrite(args.out, over)
            cv2.imwrite(root + "_mask.png", cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

    else:
        # ---------- Video ----------
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {inp}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = args.out if os.path.splitext(args.out)[1] else os.path.join(args.out, Path(inp).stem + "_overlay.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # Optionally BN-adapt on a first pass
        if args.bn_adapt_iters > 0:
            frames = []
            # Gather a subset of frames (up to ~512 frames for speed)
            max_adapt_frames = 512
            while len(frames) < max_adapt_frames:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            bn_adapt_on_images(model, frames, args.pad_h, args.pad_w, device=device, iters=args.bn_adapt_iters, bs=args.bn_bs)
            cap.release()
            cap = cv2.VideoCapture(inp)  # reopen for actual write

        # Process frames
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="Infer (video)")
        while True:
            ret, frame = cap.read()
            if not ret: break
            pred = forward_image(model, frame, args.pad_h, args.pad_w, device)
            mask_rgb = colorize_trainids(pred)
            over = overlay(frame, mask_rgb, alpha=args.overlay_alpha)
            writer.write(over)
            pbar.update(1)
        pbar.close()
        cap.release()
        writer.release()

if __name__ == "__main__":
    main()
