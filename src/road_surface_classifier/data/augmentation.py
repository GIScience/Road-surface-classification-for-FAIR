import albumentations as A
import cv2


def build_bag_augmentations() -> A.ReplayCompose:
    return A.ReplayCompose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=8.0, sat_shift_limit=25.0),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.8),
        A.OneOf([
            A.RandomRain(),
            A.RandomFog(),
            A.RandomSnow(),
            A.RandomSunFlare(),
        ], p=0.15),
        A.RandomShadow(p=0.2),
        A.Downscale(
            scale_range=(0.10, 0.35),
            interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
            p=0.4,
        ),
        A.ImageCompression(quality_range=(25, 90), p=0.4),
    ])


def build_patch_augmentations(patch_size_px: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(patch_size_px, patch_size_px), scale=(0.9, 1.0), p=0.2),
        A.Perspective(scale=(0, 0.05), keep_size=True, p=0.2),
        A.OneOf([
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(),
            A.MultiplicativeNoise(),
        ], p=0.1),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.05, 0.15), hole_width_range=(0.05, 0.15), p=0.1),
    ])
