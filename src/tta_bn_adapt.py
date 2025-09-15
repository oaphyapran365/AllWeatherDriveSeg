# src/tta_bn_adapt.py
import os, sys, argparse
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairListSegDataset
from augments import get_val_aug
from train_baseline import get_model  # reuse model def

@torch.no_grad()
def evaluate(model, loader, device, n_cls=19, ignore_index=255, desc="Eval"):
    model.eval()
    hist_flat = torch.zeros(n_cls * n_cls, device=device, dtype=torch.int64)
    for imgs, labs in tqdm(loader, desc=desc, leave=False):
        imgs, labs = imgs.to(device), labs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        valid = (labs >= 0) & (labs < n_cls) & (labs != ignore_index)
        if valid.any():
            inds = n_cls * labs[valid].to(torch.int64) + preds[valid]
            hist_flat.index_add_(0, inds, torch.ones_like(inds, dtype=torch.int64))
    hist = hist_flat.view(n_cls, n_cls).float()
    tp = torch.diag(hist)
    denom = hist.sum(1) + hist.sum(0) - tp
    ious = tp / denom.clamp_min(1)
    return torch.nanmean(ious).item() * 100.0

def bn_adapt(model, loader, device, iters=1):
    """
    Running-statistics adaptation for BatchNorm: run forward passes on target data
    (no labels, no grads) with model in train() so BN updates its running means/vars.
    """
    # put all BN layers into train mode; keep others in eval
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.training = True
            m.track_running_stats = True
        else:
            m.training = False

    n = 0
    with torch.no_grad():
        for _ in range(iters):
            for imgs, _ in loader:
                imgs = imgs.to(device)
                _ = model(imgs)
                n += imgs.size(0)
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=1, help="epochs over target data for BN-Adapt")
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--pad_h", type=int, default=1088)
    ap.add_argument("--pad_w", type=int, default=2048)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loader reused for both eval and BN-adapt
    ds = PairListSegDataset(args.pairs, aug=get_val_aug((args.pad_h, args.pad_w)))
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    # build & load
    try:
        model = get_model(19, encoder_name=args.encoder)
    except TypeError:
        model = get_model(19)
    state = torch.load(args.ckpt, map_name="cpu") if False else torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # before
    miou_before = evaluate(model, loader, device, desc="ACDC zero-shot")
    print(f"[BN-Adapt] before: mIoU={miou_before:.2f}")

    # adapt BN
    seen = bn_adapt(model, loader, device, iters=args.iters)
    print(f"[BN-Adapt] updated BN running stats using {seen} samples (iters={args.iters})")

    # after
    miou_after = evaluate(model, loader, device, desc="ACDC BN-Adapt")
    print(f"[BN-Adapt] after:  mIoU={miou_after:.2f}")
