# src/augments.py

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_aug(img_size=(768,768), robust=False):
    base = [
        A.LongestMaxSize(max(img_size)),
        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0,0,0),          # pad image with black
            mask_value=255          # pad label with ignore index
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2,0.2,0.2,0.1,p=0.5),
    ]
    if robust:
        base += [
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.35, alpha_coef=0.08, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, blur_value=3, p=0.3),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.25),
            A.RandomBrightnessContrast(0.25,0.25,p=0.5),
            A.RandomGamma(gamma_limit=(80,120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ]
    base += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(base)

def get_val_aug(img_size=(1088, 2048)):
    return A.Compose([
        A.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0,0,0),
            mask_value=255
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

