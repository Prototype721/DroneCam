from collections import defaultdict

import numpy as np
import torch


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def cxcywh_norm_to_xyxy(box, img_size: float):
    cx, cy, w, h = box
    cx, cy, w, h = cx * img_size, cy * img_size, w * img_size, h * img_size
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= thr
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return ap


def compute_detection_metrics(
    predictions: list,
    ground_truths: list,
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    box_format: str = "xyxy",       # "xyxy" | "xywh" | "cxcywh_norm"
    img_size: float = 640.0,
) -> dict:


    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def convert_box(box):
        if box_format == "xywh":
            return xywh_to_xyxy(box)
        if box_format == "cxcywh_norm":
            return cxcywh_norm_to_xyxy(box, img_size)
        return list(box)


    class_detections = defaultdict(list)
    class_gt_counts  = defaultdict(int)
    all_matched_ious = []

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = to_numpy(pred["boxes"])
        pred_scores = to_numpy(pred["scores"])
        pred_labels = to_numpy(pred["labels"]).astype(int)

        gt_boxes = to_numpy(gt["boxes"])
        gt_labels = to_numpy(gt["labels"]).astype(int)

        for lbl in gt_labels:
            class_gt_counts[int(lbl)] += 1

        keep = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        order = np.argsort(-pred_scores)
        pred_boxes  = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]

        for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
            pb_conv = convert_box(pb)
            best_iou, best_j = 0.0, -1

            for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if int(gl) != int(pl) or gt_matched[j]:
                    continue
                iou = box_iou_xyxy(pb_conv, convert_box(gb))
                if iou > best_iou:
                    best_iou, best_j = iou, j

            is_tp = best_iou >= iou_threshold and best_j >= 0
            class_detections[int(pl)].append((float(ps), int(is_tp)))
            if is_tp:
                gt_matched[best_j] = True
                all_matched_ious.append(best_iou)

    per_class_ap = []
    total_tp = total_fp = total_fn = 0

    for cls in range(num_classes):
        dets = class_detections.get(cls, [])
        n_gt = class_gt_counts.get(cls, 0)

        if n_gt == 0 and len(dets) == 0:
            continue

        if n_gt == 0:
            per_class_ap.append(0.0)
            total_fp += len(dets)
            continue

        dets_sorted = sorted(dets, key=lambda x: -x[0])
        tp_cum = np.cumsum([d[1] for d in dets_sorted])
        fp_cum = np.cumsum([1 - d[1] for d in dets_sorted])

        recalls = tp_cum / (n_gt + 1e-9)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        per_class_ap.append(compute_ap(recalls, precisions))

        tp = int(tp_cum[-1]) if len(tp_cum) else 0
        fp = int(fp_cum[-1]) if len(fp_cum) else 0
        fn = n_gt - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    mAP = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    return {
        "mAP": round(mAP, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "per_class_ap": [round(v, 4) for v in per_class_ap],
    }