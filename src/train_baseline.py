# src/train_baseline.py
import os, sys, csv, time, argparse
from datetime import datetime

# let "from datasets import ..." work without PYTHONPATH
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from datasets import PairListSegDataset
from augments import get_train_aug, get_val_aug


def get_model(n_classes=19):
    return smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    )
#----------------------
@torch.no_grad()
def evaluate(model, loader, device, n_cls=19, ignore_index=255, desc="Eval"):
    model.eval()
    # 1-D histogram for pairs (gt,pred), then reshape to [n_cls, n_cls]
    hist_flat = torch.zeros(n_cls * n_cls, device=device, dtype=torch.int64)

    for imgs, labs in tqdm(loader, desc=desc, leave=False):
        imgs, labs = imgs.to(device), labs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        valid = (labs >= 0) & (labs < n_cls) & (labs != ignore_index)
        if valid.any():
            inds = n_cls * labs[valid].to(torch.int64) + preds[valid]
            # either index_add_ ...
            hist_flat.index_add_(0, inds, torch.ones_like(inds, dtype=torch.int64))
            # ... or (equivalently) use bincount:
            # hist_flat += torch.bincount(inds, minlength=n_cls*n_cls)

    hist = hist_flat.view(n_cls, n_cls).float()
    tp = torch.diag(hist)
    denom = hist.sum(1) + hist.sum(0) - tp
    ious = tp / denom.clamp_min(1)
    miou = torch.nanmean(ious).item() * 100.0
    return miou

#----------------------------


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def init_csv(log_csv: str):
    ensure_parent_dir(log_csv)
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "epoch", "lr", "train_loss", "val_miou"])

def append_csv(log_csv: str, epoch: int, lr: float, train_loss: float, val_miou: float):
    with open(log_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(timespec="seconds"), epoch, f"{lr:.6g}", f"{train_loss:.6f}", f"{val_miou:.4f}"])

def main(args):
    os.makedirs(args.out, exist_ok=True)
    log_csv = os.path.join(args.out, "metrics.csv")
    init_csv(log_csv)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # datasets & loaders
    train_ds = PairListSegDataset(args.train_pairs, aug=get_train_aug((768,768), robust=args.robust))
    val_ds   = PairListSegDataset(args.val_pairs,   aug=get_val_aug((1024,2048)))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=2,      shuffle=False, num_workers=args.workers, pin_memory=True)

    # model / opt / loss
    model = get_model(19).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_miou = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for imgs, labs in pbar:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(n_batches, 1)

        # eval on CS val
        val_miou = evaluate(model, val_loader, device, desc="Eval-CS")

        # save checkpoints
        torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))

        # CSV log
        lr_now = optimizer.param_groups[0]["lr"]
        append_csv(log_csv, epoch, lr_now, train_loss, val_miou)

        elapsed = (time.time() - start_time) / 60.0
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_mIoU={val_miou:.2f}  best={best_miou:.2f}  time={elapsed:.1f}m")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", default="dataset/cityscapes_train_pairs.txt")
    ap.add_argument("--val_pairs",   default="dataset/cityscapes_val_pairs.txt")
    ap.add_argument("--out",         default="runs/cs_deeplabv3p_r101")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--robust", action="store_true", help="enable weather-like augs for robustness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
