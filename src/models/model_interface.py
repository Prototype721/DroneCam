import os
from abc import ABC, abstractmethod

from src.utils.logger import TrainingLogger


class BaseModel(ABC):
    def __init__(self, cfg):
        self.model = None
        self.cfg = cfg
        self.save_dir        = cfg["data"]["model_save_dir"]
        self.img_size        = cfg["data"]["img_size"]
        self.num_classes     = cfg["data"]["num_classes"]
        self.results_save_dir = cfg["results"]["save_dir"]

        self._logger: TrainingLogger | None = None

    @property
    def logger(self) -> TrainingLogger:
        if self._logger is None:
            name = getattr(self, "name", "model")
            self._logger = TrainingLogger(
                results_dir=self.results_save_dir,
                model_name=name
            )
        return self._logger

    @abstractmethod
    def train(self):
        """Run the full training process."""
        pass

    @abstractmethod
    def evaluate(self, data_loader=None) -> dict:
        """
        Evaluate on the validation split and return a metrics dict with keys:
            loss, mAP, precision, recall, f1, mean_iou, per_class_ap
        """
        pass

    @abstractmethod
    def predict(self, **kwargs):
        """Return raw model predictions."""
        pass

    @abstractmethod
    def save_model(self):
        """Save model weights to disk."""
        pass

    @abstractmethod
    def load_model(self, path):
        """Load trained weights from disk."""
        pass