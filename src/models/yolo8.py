import os

from ultralytics import YOLO

from src.utils.utils import load_config


def get_yolov8_model(typpe=0):
    types = ['n', 's', 'm', 'l']
    name_model = str(f"yolo8{types[typpe]}.pt") 
    model = Custom_YOLO8(name=name_model)
    return model


class Custom_YOLO8:

    def __init__(self, name="yolo8m.pt"):
        cfg=load_config()

        self.name = name
        self.model = YOLO(name)
        self.epochs = cfg["training"]["epochs"]
        self.project = cfg["data"]["model_save_dir"]

    def train(self, *args, **kwargs):
        self.model.train(
            epochs=self.epochs,
            project=self.project,
            name=self.name,
            exist_ok=True,
            **kwargs
        )
    
    def predict(self, *args, **kwargs):
        return self.model.predict(**kwargs)