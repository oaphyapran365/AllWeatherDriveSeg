# scripts/plot_metrics.py
import argparse, csv
import matplotlib.pyplot as plt

def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def read_csv(path):
    epochs, train_loss, val_miou = [], [], []
    eval_notes, eval_mious = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            e = safe_int(row.get("epoch", ""))
            tl = safe_float(row.get("train_loss", ""))
            vm = safe_float(row.get("val_miou", ""))

            # training/val rows have epoch
            if e is not None:
                if tl is not None: epochs.append(e); train_loss.append(tl)
                if vm is not None: # align by epoch for miou as well
                    # ensure we have same length; if not, still collect
                    pass
                # collect val_miou if present
                if vm is not None:
                    # store in parallel with epochs
                    pass
            # eval-only rows (no epoch) may have acdc_miou + note
            ac = row.get("acdc_miou", None)
            note = row.get("note", None)
            if ac is not None and ac != "":
                mf = safe_float(ac)
                if mf is not None:
                    eval_notes.append(note if note else "eval")
                    eval_mious.append(mf)

    # rebuild val_miou aligned with epochs from the CSV by re-reading, so we don't miss some rows
    # (simpler approach: recompute val_miou list via a second pass)
    val_by_epoch = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            e = safe_int(row.get("epoch", ""))
            vm = safe_float(row.get("val_miou", ""))
            if e is not None and vm is not None:
                val_by_epoch[e] = vm
    val_miou = [val_by_epoch[e] for e in epochs if e in val_by_epoch]

    return epochs, train_loss, val_miou, eval_notes, eval_mious

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to metrics.csv")
    ap.add_argument("--out", default=None, help="output base name (without extension)")
    args = ap.parse_args()

    epochs, loss, miou, eval_notes, eval_mious = read_csv(args.csv)

    # ---- Plot training loss curve ----
    if len(epochs) > 0 and len(loss) > 0:
        plt.figure()
        plt.plot(epochs, loss, label="train_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training Loss"); plt.legend()
        if args.out: plt.savefig(args.out + "_loss.png", bbox_inches="tight")
        else: plt.show()
    else:
        print("[plot] No epoch+train_loss rows found; skipping loss curve.")

    # ---- Plot validation mIoU curve ----
    if len(epochs) > 0 and len(miou) > 0:
        plt.figure()
        plt.plot(epochs[:len(miou)], miou, label="val_mIoU")
        plt.xlabel("epoch"); plt.ylabel("mIoU (%)"); plt.title("Validation mIoU"); plt.legend()
        if args.out: plt.savefig(args.out + "_miou.png", bbox_inches="tight")
        else: plt.show()
    else:
        print("[plot] No epoch+val_mIoU rows found; skipping mIoU curve.")

    # ---- Optional: ACDC eval bars (note vs mIoU) ----
    if len(eval_notes) > 0:
        plt.figure()
        xs = range(len(eval_notes))
        plt.bar(xs, eval_mious)
        plt.xticks(xs, eval_notes, rotation=30, ha="right")
        plt.ylabel("mIoU (%)"); plt.title("ACDC Eval (notes)")
        if args.out: plt.savefig(args.out + "_acdc_eval.png", bbox_inches="tight")
        else: plt.show()
    else:
        print("[plot] No ACDC eval rows (note+acdc_miou) found; skipping eval bar chart.")
