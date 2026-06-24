# Заглушка логики для main.py
import argparse
import yaml
from src.models.model_factory import ModelFactory

def load_config(config_path="configs/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        help="yolo8, yolo11, faster_rcnn, ssd, detr")
    args = parser.parse_args()
    
    cfg = load_config()
    num_classes = cfg["data"]["num_classes"]
    
    model = ModelFactory.get_model(args.model, num_classes=num_classes)
    print(f"Using {args.model} for train")
