import os

from ultralytics import YOLO

from src.models.model_interface import BaseModel
from src.evaluation.metrics import compute_detection_metrics


class Custom_YOLO11(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        yolo_cfg = cfg.get("yolo11", {})
        self.name = yolo_cfg["name"]
        self.init_name = yolo_cfg["init_name"]
        self.epochs = yolo_cfg["epochs"]
        self.device = yolo_cfg["device"]
        self.seed = yolo_cfg["seed"]
        self.data_yaml = yolo_cfg.get("data_yaml", "data/yolo/data.yaml")

        self.model = YOLO(self.init_name)

    def train(self, **kwargs):
        train_args = {
            "data": self.data_yaml,
            "epochs": self.epochs,
            "project": os.path.abspath(self.save_dir),
            "name": self.name,
            "device": self.device,
            "imgsz": self.img_size,
            "exist_ok": True,
        }
        train_args.update(kwargs)
        results = self.model.train(**train_args)

        if hasattr(results, "results_dict"):
            for epoch_idx in range(self.epochs):
                loss = results.results_dict.get("train/box_loss", 0.0)
                self.logger.log_epoch(epoch=epoch_idx + 1, loss=loss)

        self.save_model()
        metrics = self.evaluate()
        self.logger.log_epoch(epoch=self.epochs, loss=0.0, metrics=metrics)
        self.logger.save_plots()


    def evaluate(self, data_loader=None) -> dict:
        val_results = self.model.val(data=self.data_yaml, device=self.device,
                                     imgsz=self.img_size, project=os.path.abspath(self.save_dir))

        map50 = float(val_results.box.map50) if hasattr(val_results, "box") else 0.0
        precision = float(val_results.box.mp) if hasattr(val_results, "box") else 0.0
        recall = float(val_results.box.mr) if hasattr(val_results, "box") else 0.0
        f1 = (2 * precision * recall / (precision + recall + 1e-9))
        mean_iou = float(val_results.box.map75) if hasattr(val_results, "box") else 0.0
        per_class_ap = (
            [round(float(v), 4) for v in val_results.box.ap50]
            if hasattr(val_results, "box") and val_results.box.ap50 is not None
            else []
        )

        metrics = {
            "loss": 0.0,
            "mAP": round(map50,     4),
            "precision": round(precision,  4),
            "recall": round(recall,     4),
            "f1": round(f1,         4),
            "mean_iou": round(mean_iou,   4),
            "per_class_ap": per_class_ap,
        }
        return metrics


    def predict(self, *args, **kwargs):
        return self.model.predict(**kwargs)

    def save_model(self):
        best_weights_path = os.path.join(
            self.save_dir, self.name, "weights", "best.pt"
        )
        print(f"Model saved automatically in {best_weights_path}")

    def load_model(self, path):
        if os.path.exists(path):
            self.model = YOLO(path)
            print(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"Can't find file in {path}")