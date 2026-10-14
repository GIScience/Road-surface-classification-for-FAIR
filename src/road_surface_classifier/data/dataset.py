import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from road_surface_classifier.data.materialize import DATASET_CONFIG_FILENAME, PATCHES_FILENAME, ROADS_FILENAME


@dataclass(frozen=True)
class BagMode:
    fixed_bag: bool
    full_bag: bool
    augment: bool


MODES = {
    "train":  BagMode(fixed_bag=False, full_bag=False, augment=True),
    "stable": BagMode(fixed_bag=False, full_bag=True,  augment=False),
    "eval":   BagMode(fixed_bag=True,  full_bag=True,  augment=False),
}


def materialized_scenes(dataset_dir) -> list[str]:
    return sorted(path.parent.name for path in Path(dataset_dir).glob(f"*/{PATCHES_FILENAME}"))


def load_roads(dataset_dir: Path, scene_ids: list):
    roads, bags = [], []
    for scene_id in scene_ids:
        scene_dir = dataset_dir / scene_id
        stored = {}
        patches = pd.read_parquet(scene_dir / PATCHES_FILENAME, columns=["road_id", "bag", "path"])
        for (road_id, _), bag in patches.groupby(["road_id", "bag"]):
            stored.setdefault(road_id, []).append([scene_dir / path for path in bag["path"]])
        scene_roads = pd.read_parquet(scene_dir / ROADS_FILENAME, columns=["road_id", "label"])
        scene_roads = scene_roads[scene_roads["road_id"].isin(list(stored))]
        roads.append(scene_roads.assign(scene_id=scene_id))
        bags += [stored[road_id] for road_id in scene_roads["road_id"]]
    return pd.concat(roads, ignore_index=True), bags


def augment(patches, bag_transform, patch_transform):
    if bag_transform is not None:
        first = bag_transform(image=patches[0])
        patches = [first["image"], *(bag_transform.replay(first["replay"], image=patch)["image"] for patch in patches[1:])]
    if patch_transform is not None:
        patches = [patch_transform(image=patch)["image"] for patch in patches]
    return patches


def lower_resolution(images, factor):
    small = F.interpolate(images, scale_factor=1 / factor, mode="area")
    return F.interpolate(small, size=images.shape[-2:], mode="bilinear")


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        label_map: dict,
        scene_ids=None,
        seed=None,
        mode="train",
        bag_transform=None,
        patch_transform=None,
        bag_size=None,
        downsample=1,
    ) -> None:
        dataset_dir = Path(dataset_dir)
        self.label_map = label_map
        self.seed = seed
        self.mode = mode
        self.bag_transform = bag_transform
        self.patch_transform = patch_transform
        # Evaluation knobs: at most bag_size patches per bag, each downsampled by this factor and scaled back up.
        self.bag_size = bag_size
        self.downsample = downsample

        self.config = json.loads((dataset_dir / DATASET_CONFIG_FILENAME).read_text())
        self.scene_ids = list(materialized_scenes(dataset_dir) if scene_ids is None else scene_ids)
        self.roads, self.bags = load_roads(dataset_dir, self.scene_ids)
        self.labels = [-1 if pd.isna(label) else label_map[label] for label in self.roads["label"]]
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.roads)

    def __getitem__(self, idx: int):
        mode = MODES[self.mode]
        rng = np.random.default_rng([self.seed, idx]) if mode.fixed_bag else self._rng
        bag = self.bags[idx][rng.integers(len(self.bags[idx]))]
        patches = [np.asarray(Image.open(bag[slot])) for slot in self.choose_slots(len(bag), mode.full_bag, rng)]
        if mode.augment:
            patches = augment(patches, self.bag_transform, self.patch_transform)
        images = torch.stack([torch.from_numpy(patch.astype(np.float32) / 255.0).permute(2, 0, 1) for patch in patches])
        if self.downsample > 1:
            images = lower_resolution(images, self.downsample)
        return images, torch.tensor(self.labels[idx], dtype=torch.long)

    def choose_slots(self, n_slots, full_bag, rng):
        if self.bag_size is not None:
            return np.sort(rng.choice(n_slots, min(self.bag_size, n_slots), replace=False))
        if full_bag:
            return np.arange(n_slots)
        return rng.choice(n_slots, int(rng.integers(1, n_slots + 1)), replace=False)


def validation_scenes(scene_ids, n_folds: int, fold: int, seed=None) -> list[str]:
    scene_ids = sorted(scene_ids)
    if seed is not None:
        scene_ids = [str(scene) for scene in np.random.default_rng(seed).permutation(scene_ids)]
    return scene_ids[fold::n_folds]


def with_replay(dataset: PatchDataset, replay_dir, n_samples: int):
    replay = PatchDataset(replay_dir, dataset.label_map, mode="stable")
    weights = [0.5 / len(dataset)] * len(dataset) + [0.5 / len(replay)] * len(replay)
    return ConcatDataset([dataset, replay]), WeightedRandomSampler(weights, n_samples)


def collate_bags(bags):
    images, labels = zip(*bags)
    return torch.cat(images), [len(bag) for bag in images], torch.stack(labels)


def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    for part, dataset in enumerate(getattr(worker_info.dataset, "datasets", [worker_info.dataset])):
        dataset._rng = np.random.default_rng([worker_info.seed % 2**32, part])
