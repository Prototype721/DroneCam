import os

import torch
import torch.nn as nn

from src.models.model_interface import BaseModel
from src.dataset.detr_dataloader import get_detr_dataloader
from src.evaluation.metrics import compute_detection_metrics


import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor


class HungarianMatcher(nn.Module):
    """Этот класс вычисляет оптимальное по стоимости назначение между target и предсказаниями."""
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]

        # Вычисление L1 дистанции между боксами
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Вычисление GIoU (упрощенная заглушка или импорт из оригинального DETR util.box_ops)
        # Для простоты сборки проекта без лишних зависимостей часто используют внешнюю функцию box_cxcywh_to_xyxy
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in [linear_sum_assignment(c[k]) for k, c in enumerate(C.split(sizes, -1))]
        ]


# Вспомогательные функции для расчета GIoU (скопировано из DETR util/box_ops.py)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class SetCriterion(nn.Module):
    """Этот класс вычисляет лосс для DETR."""
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        indices = [(src.to(src_logits.device), tgt.to(src_logits.device)) for src, tgt in indices]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        indices = [(src.to(src_logits.device), tgt.to(src_logits.device)) for src, tgt in indices]
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()
        
        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses


class Custom_DETR(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        detr_cfg = cfg.get("detr", {})
        self.name           = detr_cfg["name"]
        self.epochs         = detr_cfg["epochs"]
        self.optimizer_name = detr_cfg["optimizer_name"]
        self.lr             = detr_cfg["lr"]
        self.weight_decay   = detr_cfg["weight_decay"]
        self.device         = detr_cfg["device"]
        self.seed           = detr_cfg["seed"]
        self.num_queries    = detr_cfg["num_queries"]

        self.detr_num_classes = self.num_classes + 1  # +1 for no-object class

        base_detr = torch.hub.load(
            "facebookresearch/detr",
            "detr_resnet50",
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

        matcher = HungarianMatcher(
            cost_class=1,
            cost_bbox=5,
            cost_giou=2
        )
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
        self.criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "boxes"]  # "cardinality" удален для упрощения кода лосса
        )
        self.criterion.to(self.device)

        self.optimizer        = self.get_optimizer()
        self.data_loader_func = get_detr_dataloader

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    def train(self):
        print(f"Start learning {self.name} on {self.device} "
              f"for {self.epochs} epochs...")

        for epoch in range(1, self.epochs+1):
            self.model.train()
            self.criterion.train()
            epoch_loss  = 0.0
            data_loader = self.data_loader_func(epoch_id=epoch)

            for images, targets in data_loader:
                images = torch.stack(images).to(self.device)

                detr_targets = [
                    {
                        "labels": t["labels"].to(self.device),
                        "boxes":  t["boxes"].to(self.device),
                    }
                    for t in targets
                ]

                outputs   = self.model(images)
                loss_dict = self.criterion(outputs, detr_targets)
                weight_dict = self.criterion.weight_dict
                losses    = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                epoch_loss += losses.item()

            avg_loss = epoch_loss / len(data_loader)  # fixed: was self.data_loader

            metrics = self.evaluate()

            self.logger.log_epoch(
                epoch=epoch + 1,
                loss=avg_loss,
                metrics=metrics
            )

        auto_path = os.path.join(self.save_dir, f"{self.name}_last.pth")
        self.save_model(auto_path)
        self.logger.save_plots()

    # ------------------------------------------------------------------

    def evaluate(self, data_loader=None) -> dict:
        """
        Evaluate on the validation split.

        Returns dict with: loss, mAP, precision, recall, f1,
                           mean_iou, per_class_ap.
        DETR boxes are in normalised [cx, cy, w, h] format, so we pass
        box_format="cxcywh_norm" to the metric helper.
        """
        if data_loader is None:
            data_loader = self.data_loader_func(is_valid=True)

        self.model.eval()
        self.criterion.eval()
        total_loss = 0.0
        all_preds, all_gts = [], []

        with torch.no_grad():
            for images, targets in data_loader:
                images = torch.stack(images).to(self.device)

                detr_targets = [
                    {
                        "labels": t["labels"].to(self.device),
                        "boxes":  t["boxes"].to(self.device),
                    }
                    for t in targets
                ]

                outputs   = self.model(images)
                loss_dict = self.criterion(outputs, detr_targets)
                weight_dict = self.criterion.weight_dict
                loss      = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )
                total_loss += loss.item()

                # Convert DETR output to per-image predictions
                prob   = outputs["pred_logits"].softmax(-1)
                scores, labels = prob[..., :-1].max(-1)
                boxes  = outputs["pred_boxes"]

                for b in range(images.shape[0]):
                    all_preds.append({
                        "boxes":  boxes[b].cpu(),
                        "scores": scores[b].cpu(),
                        "labels": labels[b].cpu(),
                    })
                    all_gts.append({
                        "boxes":  targets[b]["boxes"].cpu(),
                        "labels": targets[b]["labels"].cpu(),
                    })

        avg_loss = total_loss / len(data_loader)
        metrics  = compute_detection_metrics(
            predictions=all_preds,
            ground_truths=all_gts,
            num_classes=self.num_classes,
            box_format="cxcywh_norm",
            img_size=float(self.img_size),
        )
        metrics["loss"] = round(avg_loss, 6)
        return metrics

    # ------------------------------------------------------------------

    def predict(self, images, *args, **kwargs):
        self.model.eval()
        print(f"Predicting targets for {self.name}")
        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(images)

        prob   = outputs["pred_logits"].softmax(-1)
        scores, labels = prob[..., :-1].max(-1)
        boxes  = outputs["pred_boxes"]

        return [
            {
                "boxes":  boxes[i].cpu(),
                "scores": scores[i].cpu(),
                "labels": labels[i].cpu(),
            }
            for i in range(images.shape[0])
        ]

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