import os
import torch
import torch.nn as nn
from src.models.model_interface import BaseModel
from src.dataset.torch_dataset import get_data_loader


class Custom_DETR(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        
        detr_cfg = cfg.get("detr", {})
        self.name = detr_cfg["name"]
        self.epochs = detr_cfg["epochs"]
        self.optimizer_name = detr_cfg["optimizer_name"]
        self.lr = detr_cfg["lr"]
        self.weight_decay = detr_cfg["weight_decay"]
        self.device = detr_cfg["device"]
        self.seed = detr_cfg["seed"]
        self.num_queries = detr_cfg["num_queries"]

        self.detr_num_classes = self.num_classes + 1 
        
        base_detr = torch.hub.load(
            'facebookresearch/detr',
            'detr_resnet50', 
            pretrained=True
        )
        in_features = base_detr.class_embed.in_features
        
        base_detr.class_embed = nn.Linear(
            in_features=in_features, 
            out_features=self.detr_num_classes
        )
        base_detr.num_queries = self.num_queries
        self.model = base_detr
        self.model.to(self.device)

        matcher = torch.hub.load(
            'facebookresearch/detr', 
            'hungarian_matcher', 
            cost_class=1, 
            cost_bbox=5, 
            cost_giou=2
        )
        
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        
        self.criterion = torch.hub.load(
            'facebookresearch/detr', 
            'set_criterion', 
            num_classes=self.num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            eos_coef=0.1,
            losses=['labels', 'boxes', 'cardinality']
        )
        self.criterion.to(self.device)

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
        print(f"Start learning {self.name} on {self.device} " \
              f"for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            self.model.train()
            self.criterion.train()
            epoch_loss = 0
            
            data_loader = self.data_loader_func(epoch_id=epoch,
                                                is_detr=True)
            for images, targets in data_loader:
                images = torch.stack(images).to(self.device)

                detr_targets = []
                for t in targets:
                    detr_targets.append({
                        'labels': t['labels'].to(self.device),
                        'boxes': t['boxes'].to(self.device)
                    })

                outputs = self.model(images)
                
                loss_dict = self.criterion(outputs, detr_targets)
                weight_dict = self.criterion.weight_dict
                
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

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
        
        images = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
        
        prob = outputs['pred_logits'].softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = outputs['pred_boxes']
        
        predictions = []
        for b_idx in range(images.shape[0]):
            predictions.append({
                'boxes': boxes[b_idx].cpu(),
                'scores': scores[b_idx].cpu(),
                'labels': labels[b_idx].cpu()
            })
            
        return predictions

    def save_model(self, custom_path=None):
        if custom_path is None:
            custom_path = os.path.join(self.save_dir, f"{self.name}_weights.pth")
            
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
