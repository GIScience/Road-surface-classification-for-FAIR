from pathlib import Path

import lightning as L
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.pytorch.callbacks import Callback, EarlyStopping, EMAWeightAveraging, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from road_surface_classifier.data.augmentation import build_bag_augmentations, build_patch_augmentations
from road_surface_classifier.data.dataset import PatchDataset, collate_bags, materialized_scenes, validation_scenes, worker_init_fn
from road_surface_classifier.evaluate import FinalEvaluation
from road_surface_classifier.explain import ExplainBags
from road_surface_classifier.model import BACKBONES, POOLS, MILModel

LABEL_MAP = {0: 0, 1: 1, 2: 1}
CLASS_NAMES = ["unpaved", "paved"]


class StableEpochs(Callback):
    def __init__(self, dataset: PatchDataset, n_epochs: int):
        self.dataset = dataset
        self.n_epochs = n_epochs

    def on_fit_start(self, trainer, pl_module):
        self.start = trainer.max_epochs - self.n_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.should_stop and trainer.current_epoch < self.start:
            self.start = trainer.current_epoch
        self.dataset.mode = "stable" if trainer.current_epoch >= self.start else "train"
        pl_module.log("stable_epoch", float(self.dataset.mode == "stable"))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start:
            trainer.fit_loop.min_epochs = trainer.current_epoch + 1 + self.n_epochs


if __name__ == "__main__":
    p = ArgumentParser(default_config_files=["config/train.yaml"])
    p.add_argument("--config", action=ActionConfigFile)
    p.add_argument("--backbone", default="dinov3", choices=list(BACKBONES))
    p.add_argument("--pool", default="attention", choices=list(POOLS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=5)
    p.add_argument("--layer-decay", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--stable-epochs", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets/v1"))
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=None)
    p.add_argument("--val-scenes", type=list[str], default=None)
    p.add_argument("--log-dir", type=Path, default=Path("lightning_logs"))
    p.add_argument("--plot-bags", type=int, default=4)
    p.add_argument("--plot-every", type=int, default=10)
    args = p.parse_args()

    scenes = materialized_scenes(args.dataset_dir)
    val_scenes = args.val_scenes or validation_scenes(scenes, args.n_folds, args.fold, args.split_seed)
    train_scenes = [scene for scene in scenes if scene not in val_scenes]
    val_ds = PatchDataset(args.dataset_dir, LABEL_MAP, scene_ids=val_scenes, seed=0, mode="eval")
    patch_size_px = val_ds.config["patch_size_px"]
    train_ds = PatchDataset(
        args.dataset_dir,
        LABEL_MAP,
        scene_ids=train_scenes,
        bag_transform=build_bag_augmentations(),
        patch_transform=build_patch_augmentations(patch_size_px),
    )

    loader = dict(collate_fn=collate_bags, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, **loader)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, **loader)
    eval_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, **loader)

    counts = torch.bincount(torch.tensor(train_ds.labels), minlength=len(CLASS_NAMES))
    class_weights = (counts.sum() / (len(CLASS_NAMES) * counts)).tolist()
    model = MILModel(
        class_weights,
        args.backbone,
        args.pool,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        label_smoothing=args.label_smoothing,
        layer_decay=args.layer_decay,
        weight_decay=args.weight_decay,
    )

    logger = TensorBoardLogger(args.log_dir, name="road-surface-classifier")
    logger.log_hyperparams({"dataset_dir": str(args.dataset_dir), "n_folds": args.n_folds, "fold": args.fold,
                            "split_seed": args.split_seed, "val_scenes": val_scenes, **val_ds.config})
    best_val_acc_checkpoint = ModelCheckpoint(monitor="val_acc", mode="max", filename="best-acc-{epoch}-{step}")
    fixed_bags = [val_ds[int(idx)] for idx in torch.linspace(0, len(val_ds) - 1, args.plot_bags).long()]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        default_root_dir=args.log_dir,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(filename="last-{epoch}-{step}"),
            best_val_acc_checkpoint,
            EarlyStopping(monitor="val_loss", patience=args.patience),
            LearningRateMonitor(logging_interval="step"),
            ExplainBags(fixed_bags, val_ds, CLASS_NAMES, args.plot_every),
            StableEpochs(train_ds, args.stable_epochs),
            FinalEvaluation(eval_dl, best_val_acc_checkpoint, "ema" if args.ema_decay else "last"),
            *([EMAWeightAveraging(device="cuda" if torch.cuda.is_available() else None, decay=args.ema_decay)]
              if args.ema_decay else []),
        ],
    )
    trainer.fit(model, train_dl, val_dl)
