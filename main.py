import argparse

from src.utils.utils import load_config
from src.training.train import train_model


VALID_MODELS = ["yolo8", "yolo11", "faster_rcnn", "ssd", "detr"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisDrone object detection training script",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        choices=VALID_MODELS,
        metavar="MODEL",
        help="Model to train. One of: " + ", ".join(VALID_MODELS)
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        metavar="PATH",
        help="Path to config YAML (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--evaluate-only", "-e",
        action="store_true",
        help="Skip training — run evaluate() on the validation split only"
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to pretrained weights to load before training/evaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    train_model(
        model_name=args.model,
        cfg=cfg,
        weights_path=args.weights,
        evaluate_only=args.evaluate_only,
    )