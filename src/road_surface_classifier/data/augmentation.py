import io
import math

import albumentations as A
import cv2
import numpy as np
from PIL import Image

PATCH_GSD = 15.0 / 256  # metres per pixel of a stored patch


def jpeg(image, quality):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=quality)
    return np.asarray(Image.open(buffer))


def rotate(image, angle):
    n = image.shape[0]
    matrix = cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (n, n), borderMode=cv2.BORDER_REFLECT_101)


def nearest(image, step, phase):
    idx = ((np.arange(int(len(image) / step)) + phase) * step).astype(int)
    return image[idx][:, idx]


def oam_tile_chain(image, tile_gsd, native_gsd, angle, phase, overview_quality, tile_quality):
    """Imitates how OpenAerialMap serves a drone image as a web tile."""
    size = image.shape[0]
    pad = size // 4
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    n = image.shape[0]
    image = rotate(image, angle)  # road-aligned patch -> north-up tile grid
    level = 2 ** math.floor(math.log2(tile_gsd / native_gsd))  # finest COG overview not coarser than the tile
    overview_step = native_gsd * level / PATCH_GSD
    tile_step = tile_gsd / PATCH_GSD
    overview = jpeg(nearest(image, overview_step, phase[0]), overview_quality)
    tile = jpeg(nearest(overview, tile_step / overview_step, phase[1]), tile_quality)
    image = rotate(cv2.resize(tile, (n, n)), -angle)
    return image[pad:pad + size, pad:pad + size]


class OAMTileEmulation(A.ImageOnlyTransform):
    def get_params(self):
        zoom = self.py_random.choice([18, 19, 20])
        latitude = self.py_random.uniform(0, 55)
        return {
            "tile_gsd": 156543.03392 * math.cos(math.radians(latitude)) / 2**zoom,
            "native_gsd": self.py_random.uniform(0.02, 0.08),
            "angle": self.py_random.uniform(0, 90),
            "phase": (self.py_random.random(), self.py_random.random()),
            "overview_quality": self.py_random.randint(70, 80),
            "tile_quality": self.py_random.randint(80, 90),
        }

    def apply(self, img, tile_gsd, native_gsd, angle, phase, overview_quality, tile_quality, **params):
        return oam_tile_chain(img, tile_gsd, native_gsd, angle, phase, overview_quality, tile_quality)


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
        A.OneOf([  # half the bags get the OAM tile chain, the other half the downscale + JPEG block
            OAMTileEmulation(p=0.5),
            A.Sequential([
                A.Downscale(
                    scale_range=(0.10, 0.35),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.4,
                ),
                A.ImageCompression(quality_range=(25, 90), p=0.4),
            ], p=0.5),
        ], p=1.0),
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
