import os

import torch
from torchvision.models.detection import (
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights
)
from torchvision.models.detection.ssd import SSDHead

from src.models.model_interface import BaseModel
from src.dataset.torch_dataloader import get_torch_dataloader
from src.evaluation.metrics import compute_detection_metrics


class Custom_SSD(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        models = {
            "vgg16": {
                "fn": ssd300_vgg16,
                "weights": SSD300_VGG16_Weights.DEFAULT
            },
            "v3_large": {
                "fn": ssdlite320_mobilenet_v3_large,
                "weights": SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            }
        }

        ssd_cfg = cfg.get("ssd", {})
        self.name = ssd_cfg["name"]
        self.epochs = ssd_cfg["epochs"]
        self.model_type = ssd_cfg["model_type"]
        self.optimizer_name = ssd_cfg["optimizer_name"]
        self.lr = ssd_cfg["lr"]
        self.weight_decay = ssd_cfg["weight_decay"]
        self.device = ssd_cfg["device"]
        self.seed = ssd_cfg["seed"]
        self.num_classes = self.num_classes + 1

        if self.model_type not in models:
            raise ValueError(
                f"Can't find model '{self.model_type}'. "
                f"Available: {list(models.keys())}"
            )

        chosen = models[self.model_type]
        self.model = chosen["fn"](weights=chosen["weights"])

        in_channels = []
        for layer in self.model.head.classification_head.module_list:
            if hasattr(layer, "in_channels"):
                in_channels.append(layer.in_channels)
            else:
                first_sublayer = next(layer.children())
                in_channels.append(first_sublayer.in_channels)

        num_anchors = self.model.anchor_generator.num_anchors_per_location()

        self.model.head = SSDHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )
        self.model.to(self.device)

        self.optimizer = self.get_optimizer()
        self.data_loader_func = get_torch_dataloader

    def get_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_name == "SGD":
            return torch.optim.SGD(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "AdamW":
            return torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        raise ValueError(f"Can't find {self.optimizer_name} in optimizers!")

    def train(self):
        print(f"Start learning {self.name} on {self.device} "
              f"for {self.epochs} epochs...")

        for epoch in range(1, self.epochs+1):
            self.model.train()
            epoch_loss  = 0.0
            data_loader = self.data_loader_func(epoch_id=epoch)

            for images, targets in data_loader:
                images  = list(img.to(self.device) for img in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()

            avg_loss = epoch_loss / len(data_loader)

            metrics = self.evaluate()

            self.logger.log_epoch(
                epoch=epoch + 1,
                loss=avg_loss,
                metrics=metrics
            )

        auto_path = os.path.join(self.save_dir, f"{self.name}_last.pth")
        self.save_model(auto_path)
        self.logger.save_plots()

    def evaluate(self, data_loader=None) -> dict:
        if data_loader is None:
            data_loader = self.data_loader_func(
                is_valid=True
            )

        self.model.train()
        total_loss = 0.0
        all_preds, all_gts = [], []

        with torch.no_grad():
            for images, targets in data_loader:
                images      = list(img.to(self.device) for img in images)
                targets_dev = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                loss_dict   = self.model(images, targets_dev)
                total_loss += sum(v.item() for v in loss_dict.values())

                self.model.eval()
                preds = self.model(images)
                self.model.train()

                for pred, tgt in zip(preds, targets):
                    all_preds.append({
                        "boxes":  pred["boxes"].cpu(),
                        "scores": pred["scores"].cpu(),
                        "labels": pred["labels"].cpu(),
                    })
                    all_gts.append({
                        "boxes":  tgt["boxes"].cpu(),
                        "labels": tgt["labels"].cpu(),
                    })

        avg_loss = total_loss / len(data_loader)
        metrics  = compute_detection_metrics(
            predictions=all_preds,
            ground_truths=all_gts,
            num_classes=self.num_classes,
            box_format="xyxy",
        )
        metrics["loss"] = round(avg_loss, 6)
        return metrics

    def predict(self, images, *args, **kwargs):
        self.model.eval()
        print(f"Predicting targets for {self.name}")
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            predictions = self.model(images)
        return [{k: v.cpu() for k, v in res.items()} for res in predictions]

    def save_model(self, custom_path=None):
        if custom_path is None:
            custom_path = os.path.join(
                self.save_dir, f"{self.name}_weights.pth"
            )
        os.makedirs(os.path.dirname(custom_path), exist_ok=True)
        torch.save(self.model.state_dict(), custom_path)
        print(f"Model saved in: {custom_path}")

    def load_model(self, path):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            print(f"Model loaded from: {path}")
        else:
            raise FileNotFoundError(f"Can't find weights: {path}")