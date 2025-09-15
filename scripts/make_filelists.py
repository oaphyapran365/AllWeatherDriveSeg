import os, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print(f"[OK] {path}  ({len(rows)} pairs)")

# ---------- Cityscapes (you created Cityscapes_clean) ----------
def make_cityscapes():
    img_root = os.path.join(ROOT, "Cityscapes_clean", "leftImg8bit")
    lab_root = os.path.join(ROOT, "Cityscapes_clean", "gtFine")
    for split in ["train", "val"]:
        imgs = sorted(glob.glob(os.path.join(img_root, split, "*", "*_leftImg8bit.png")))
        rows = []
        for p in imgs:
            city = os.path.basename(os.path.dirname(p))
            base = os.path.basename(p).replace("_leftImg8bit.png","_gtFine")
            lab_trainids = os.path.join(lab_root, split, city, base + "_labelTrainIds.png")
            lab_labelids  = os.path.join(lab_root, split, city, base + "_labelIds.png")
            # prefer trainIds; fall back to labelIds if needed
            lab = lab_trainids if os.path.isfile(lab_trainids) else lab_labelids
            if os.path.isfile(lab):
                rows.append(f"{p} {lab}")
        write(os.path.join(ROOT, f"cityscapes_{split}_pairs.txt"), rows)


# ---------- ACDC (keep your current layout) ----------
# images: ACDC/rgb_anon/<cond>/<split>/**/_rgb_anon.png
# labels: ACDC/gt_trainval_ref/<cond>/<split_ref>/**/_gt_ref_labelIds.png
# where split_ref is train->train_ref, val->val_ref
def make_acdc():
    img_root = os.path.join(ROOT, "ACDC", "rgb_anon")
    lab_root = os.path.join(ROOT, "ACDC", "gt_trainval_ref")
    conds = ["fog", "night", "rain", "snow"]
    for split in ["train","val"]:
        split_ref = f"{split}_ref"
        rows = []
        for c in conds:
            imgs = sorted(glob.glob(os.path.join(img_root, c, split, "**", "*_rgb_anon.png"), recursive=True))
            for p in imgs:
                # path relative to images split
                rel = os.path.relpath(p, os.path.join(img_root, c, split))
                lab = os.path.join(lab_root, c, split_ref, rel).replace("_rgb_anon.png","_gt_ref_labelIds.png")
                if os.path.isfile(lab):
                    rows.append(f"{p} {lab}")
        write(os.path.join(ROOT, f"acdc_{split}_pairs.txt"), rows)

if __name__ == "__main__":
    make_cityscapes()
    make_acdc()
