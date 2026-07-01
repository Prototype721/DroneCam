from src.models.yolo8 import Custom_YOLO8
from src.models.yolo11 import Custom_YOLO11
from src.models.faster_rcnn import Custom_Faster_RCNN
from src.models.ssd import Custom_SSD
from src.models.detr import Custom_DETR

from src.utils.utils import seed_everything
class ModelFactory:
    
    models = {
        "yolo8": Custom_YOLO8,
        "yolo11": Custom_YOLO11,
        "faster_rcnn": Custom_Faster_RCNN,
        "ssd": Custom_SSD,
        "detr": Custom_DETR,
    }

    @classmethod
    def get_model(cls, model_name, cfg, **kwargs):
        
        model_class = cls.models.get(model_name.lower())
        
        if not model_class:
            valid_names = list(cls._models.keys())
            raise ValueError(f"Model '{model_name}' can't be find from: {valid_names}")
        
        seed_everything(cfg["data"]["seed"])
        
        return model_class(cfg, **kwargs)
