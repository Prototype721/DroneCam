import os

from ultralytics import YOLO

from src.models.model_interface import BaseModel

class Custom_YOLO11(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        yolo_cfg = cfg.get("yolo11", {})
        self.name = yolo_cfg["name"]
        self.init_name = yolo_cfg["init_name"]
        self.epochs = yolo_cfg["epochs"]
        self.device = yolo_cfg["device"]
        self.seed = yolo_cfg["seed"]

        self.model = YOLO(self.init_name)


    def train(self, **kwargs):
        train_args = {
            "epochs": self.epochs,
            "project": self.save_dir,
            "name": self.name,
            "exist_ok": True
        }
        train_args.update(kwargs)
        self.model.train(**train_args)

        self.save_model()


    def predict(self, *args, **kwargs):
        return self.model.predict(**kwargs)


    def save_model(self):
        best_weights_path = os.path.join(self.save_dir, self.name, 'weights', 'best.pt')
        print(f"Model saved automatic in {best_weights_path}")


    def load_model(self, path):
        if os.path.exists(path):
            self.model = YOLO(path)
            print(f"Model downloaded from {path}")
        else:
            raise FileNotFoundError(f"Can't find file in {path}")