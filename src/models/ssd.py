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
from src.dataset.torch_dataset import get_data_loader


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

        if self.model_type not in models:
            raise ValueError(f"Can't find model '{self.model_type}'." \
                             f"Available: {list(models.keys())}")
        
        chosen = models[self.model_type]
        self.model = chosen["fn"](weights=chosen["weights"])

        data_channels = self.model.head.classification_head.data_channels
        num_anchors  = self.model.head.classification_head.num_anchors
        self.model.head = SSDHead(
            in_channels=data_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )
        self.model.to(self.device)

        self.optimizer = self.get_optimizer()
        
        self.data_loader_func = get_data_loader


    def get_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_name == "SGD":
            return torch.optim.SGD(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "AdamW":
            return torch.optim.AdamW(
                params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        raise ValueError(f"Can't find {self.optimizer_name} in optimizers!")


    def train(self):
        self.model.train()
        
        print(f"Star learning {self.name} on {self.device}" \
              f"for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            data_loader = self.data_loader_func(epoch_id=epoch,
                                                shift_classes=True)
            for images, targets in data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()}
                            for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()
            
            print(f"Epoch {epoch+1}/{self.epochs} | " \
                  f"Loss: {epoch_loss / len(data_loader):.4f}")
            
        auto_path = os.path.join(self.save_dir, f"{self.name}_last.pth")
        self.save_model(auto_path)

    def predict(self, images, *args, **kwargs):
        self.model.eval()
        print(f"Predicting targets for {self.name}")
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            predictions = self.model(images)
        
        return [{k: v.cpu() for k, v in res.items()} for res in predictions]


    def save_model(self, custom_path=None):
        if custom_path is None:
            custom_path = os.path.join(self.save_dir,
                                    f"{self.name}_weights.pth")
            
        os.makedirs(os.path.dirname(custom_path), exist_ok=True)
        torch.save(self.model.state_dict(), custom_path)
        print(f"Model saved in: {custom_path}")

    def load_model(self, path):
        if os.path.exists(path):
            
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            print(f"Model downloaded from: {path}")
        else:
            raise FileNotFoundError(f"Can't find path for weights: {path}")