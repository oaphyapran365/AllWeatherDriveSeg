# src/datasets.py
import torch, cv2
from torch.utils.data import Dataset
import numpy as np

class PairListSegDataset(Dataset):
    def __init__(self, pairs_txt, aug=None, ignore_index=255):
        self.items = [l.strip().split() for l in open(pairs_txt).read().strip().splitlines()]
        self.aug = aug
        self.ignore_index = ignore_index

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        img_path, lab_path = self.items[i]
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        # ensure label is int64 BEFORE aug
        if isinstance(lab, np.ndarray):
            lab = lab.astype(np.int64)

        if self.aug:
            out = self.aug(image=img, mask=lab)
            img, lab = out["image"], out["mask"]  # likely already torch tensors
            # enforce dtypes
            if isinstance(img, torch.Tensor):
                # img should be float (ToTensorV2 + Normalize)
                if img.dtype != torch.float32:
                    img = img.float()
            else:
                # fallback (no ToTensorV2 in aug): HWC -> CHW
                img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0

            if isinstance(lab, torch.Tensor):
                lab = lab.long()
            else:
                lab = torch.from_numpy(lab).long()
        else:
            # No aug path (rare here): convert manually
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            lab = torch.from_numpy(lab.astype(np.int64))

        return img, lab
