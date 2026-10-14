import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.optim import param_groups_layer_decay
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC


class MeanPool(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        weights = torch.full((len(x),), 1 / len(x), device=x.device)
        return x.mean(0), weights


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x):
        gated = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        scores = self.w(gated).squeeze(-1)
        weights = torch.softmax(scores, 0)
        return weights @ x, weights


POOLS = {"mean": MeanPool, "attention": AttentionPool}

BACKBONES = {
    "dinov3": "vit_small_patch16_dinov3.lvd1689m",
    "convnext": "convnext_tiny",
    "efficientnet": "efficientnet_b0",
}


class MILModel(L.LightningModule):
    def __init__(self, class_weights, backbone="dinov3", pool="attention", lr=1e-4, freeze_epochs=0, label_smoothing=0.0,
                 layer_decay=1.0, weight_decay=0.05, trainable="all", pretrained=True):
        super().__init__()
        self.save_hyperparameters(ignore="pretrained")
        self.backbone = timm.create_model(BACKBONES[backbone], pretrained=pretrained, num_classes=0)
        dim = self.backbone.num_features
        self.pool = POOLS[pool](dim)
        self.classifier = nn.Linear(dim, len(class_weights))
        self.register_buffer("mean", torch.tensor(self.backbone.pretrained_cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.backbone.pretrained_cfg["std"]).view(1, 3, 1, 1))
        self.register_buffer("class_weights", torch.tensor(class_weights), persistent=False)
        self.train_acc = MulticlassAccuracy(num_classes=len(class_weights), average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=len(class_weights), average="macro")
        self.val_auroc = MulticlassAUROC(num_classes=len(class_weights), average="macro")

    def feature_maps(self, images):
        return self.backbone.forward_features((images - self.mean) / self.std)

    def classify(self, feature_maps, sizes):
        features = self.backbone.forward_head(feature_maps)
        bag_features, weights = zip(*(self.pool(bag) for bag in features.split(sizes)))
        logits = self.classifier(torch.stack(bag_features))
        return logits, torch.cat(weights)

    def forward(self, images, sizes):
        return self.classify(self.feature_maps(images), sizes)

    def bag_step(self, batch):
        images, sizes, labels = batch
        logits, _ = self(images, sizes)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=self.hparams.label_smoothing)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.bag_step(batch)
        self.train_acc(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(labels))
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, batch_size=len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.bag_step(batch)
        self.val_acc(logits, labels)
        self.val_auroc(logits.float().softmax(1), labels)
        self.log("val_loss", loss, batch_size=len(labels))
        self.log("val_acc", self.val_acc, batch_size=len(labels))
        self.log("val_auroc", self.val_auroc, batch_size=len(labels))

    def on_train_epoch_start(self):
        frozen = self.hparams.trainable != "all" or self.current_epoch < self.hparams.freeze_epochs
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(not frozen)
        self.pool.requires_grad_(self.hparams.trainable != "classifier")

    def configure_optimizers(self):
        groups = [] if self.hparams.trainable != "all" else param_groups_layer_decay(
            self.backbone, self.hparams.weight_decay, layer_decay=self.hparams.layer_decay)
        groups.append({"params": [self.classifier.weight], "lr_scale": 1.0})
        pool = [] if self.hparams.trainable == "classifier" else [*self.pool.parameters()]
        groups.append({"params": [*pool, self.classifier.bias], "lr_scale": 1.0, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.trainable != "all":
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[self.hparams.lr * group["lr_scale"] for group in groups],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.freeze_epochs / self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}



def split_weights(state_dict, hyper_parameters):
    encoder = {key.removeprefix("backbone."): value for key, value in state_dict.items() if key.startswith("backbone.")}
    head = {key: value for key, value in state_dict.items() if not key.startswith("backbone.")}
    return encoder, {"hyper_parameters": dict(hyper_parameters), "state_dict": head}


def load_model(head_path, encoder_path=None, **hyper_parameters):
    head = torch.load(head_path, map_location="cpu")
    model = MILModel(**{**head["hyper_parameters"], **hyper_parameters}, pretrained=encoder_path is None)
    if encoder_path is not None:
        model.backbone.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    model.load_state_dict(model.state_dict() | head["state_dict"])
    return model
