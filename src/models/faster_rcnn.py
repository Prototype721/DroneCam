import os

import torch
from torchvision.models.detection.faster_rcnn import (
    fasterrcnn_resnet50_fpn_v2, 
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FastRCNNPredictor
)
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
)

from src.utils.utils import load_config


def get_faster_rcnn_model(typpe=0):
    types = {
        0: [fasterrcnn_resnet50_fpn_v2, 
           FasterRCNN_ResNet50_FPN_V2_Weights],
        1: [fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights]
    }
    model = Custom_Faster_RCNN(typpe=types[typpe])
    return model


class Custom_Faster_RCNN:

    def __init__(self, typpe, name="faster_rcnn"):
        cfg=load_config()

        self.name = name
        self.num_classes = cfg["data"]["num_classes"]
        self.input_shape = cfg["training"]["img_size"]
        self.epochs = cfg["training"]["epochs"]
        self.save_dir = cfg["data"]["model_save_dir"]
        self.model = typpe[0](
            weights=typpe[1].DEFAULT
        )
        in_features = self.model.roi_head.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )

    def train(self, *args, **kwargs):
        pass
    
    def predict(self, *args, **kwargs):
        with torch.no_grad():
            pass