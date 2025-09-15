# src/make_demo_video.py
import os, argparse, time, cv2, numpy as np
from pathlib import Path

# Reuse your infer utilities
from infer import load_model, forward_image, colorize_trainids, overlay
from augments import get_val_aug  # just to keep pad sizes consistent

def hstack_triptych(img_bgr, mask_rgb, over_bgr):
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return np.concatenate([img_bgr, mask_bgr, over_bgr], axis=1)

def draw_hud(frame, fps_ema, title="", y0=36):
    txts = [title, f"FPS (EMA): {fps_ema:5.1f}"]
    for i, t in enumerate(txts):
        y = y0 + i*28
        cv2.putText(frame, t, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, t, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def load_image_paths(pairs_txt):
    paths = []
    with open(pairs_txt) as f:
        for line in f:
            img, *_ = line.strip().split()
            if Path(img).suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"]:
                paths.append(img)
    return paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="file with 'img label' per line (we use img)")
    ap.add_argument("--ckpt",  required=True, help="model .pth")
    ap.add_argument("--out",   required=True, help="output mp4")
    ap.add_argument("--seconds", type=int, default=40, help="target video length")
    ap.add_argument("--fps",     type=int, default=30, help="output video FPS")
    ap.add_argument("--pad_h", type=int, default=1088)
    ap.add_argument("--pad_w", type=int, default=2048)
    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--triptych", action="store_true", help="input | mask | overlay")
    ap.add_argument("--title", default="DeepLabV3+ R101 • Cityscapes→ACDC")
    ap.add_argument("--ema_beta", type=float, default=0.9, help="FPS EMA smoothing")
    args = ap.parse_args()

    imgs = load_image_paths(args.pairs)
    if not imgs: raise SystemExit(f"No images in {args.pairs}")
    # choose N frames evenly across val set for ~seconds
    N = min(len(imgs), args.fps * args.seconds)
    idxs = np.linspace(0, len(imgs)-1, N).astype(int)
    sel = [imgs[i] for i in idxs]

    # Read first frame to size the video
    first = cv2.imread(sel[0], cv2.IMREAD_COLOR)
    H, W = first.shape[:2]
    frame_w = W*3 if args.triptych else W
    frame_h = H
    os.makedirs(Path(args.out).parent, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame_w, frame_h))

    model = load_model(args.ckpt, encoder_name=args.encoder)
    fps_ema = 0.0
    for k, p in enumerate(sel, 1):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        pred = forward_image(model, bgr, pad_h=args.pad_h, pad_w=args.pad_w)
        dt = max(time.perf_counter() - t0, 1e-6)
        fps_inst = 1.0 / dt
        fps_ema = fps_inst if fps_ema == 0 else args.ema_beta*fps_ema + (1-args.ema_beta)*fps_inst

        mask_rgb = colorize_trainids(pred)
        over = overlay(bgr, mask_rgb, alpha=args.overlay_alpha)
        frame = hstack_triptych(bgr, mask_rgb, over) if args.triptych else over.copy()

        draw_hud(frame, fps_ema, title=args.title)
        # optional frame index
        cv2.putText(frame, f"{k}/{N}", (frame.shape[1]-140, frame.shape[0]-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{k}/{N}", (frame.shape[1]-140, frame.shape[0]-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"[OK] Demo saved to {args.out}")
