# src/eval_acdc.py
import os, sys, csv, argparse
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairListSegDataset
from augments import get_val_aug
from train_baseline import get_model  # reuse same model def




@torch.no_grad()
def evaluate(model, loader, device, n_cls=19, ignore_index=255, desc="Eval-ACDC"):
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
    miou = torch.nanmean(ious).item() * 100.0
    return miou

def maybe_append_csv(csv_path, label, miou):
    if not csv_path: return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # ensure header exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["time","epoch","lr","train_loss","val_miou", "note", "acdc_miou"])
    # append with note
    from datetime import datetime
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.now().isoformat(timespec="seconds"), "", "", "", "", label, f"{miou:.4f}"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="dataset/acdc_val_pairs.txt", help="image label pairs txt (ACDC)")
    ap.add_argument("--ckpt",  required=True, help="path to model checkpoint (.pth)")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=None, help="optional: append result to this CSV (e.g., runs/.../metrics.csv)")
    ap.add_argument("--note", default="ACDC-val", help="short label to record in CSV")
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--pad_h", type=int, default=1088)
    ap.add_argument("--pad_w", type=int, default=2048)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairListSegDataset(args.pairs, aug=get_val_aug((args.pad_h, args.pad_w)))

    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = get_model(19, encoder_name=args.encoder) if "encoder_name" in get_model.__code__.co_varnames else get_model(19)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    miou = evaluate(model, loader, device)
    print(f"[ACDC] mIoU: {miou:.2f}")
    maybe_append_csv(args.csv, args.note, miou)
