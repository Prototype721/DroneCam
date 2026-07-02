"""
    python main.py --model yolo8
    python main.py --model faster_rcnn --config configs/default.yaml
    python main.py --model detr --weights path/to/weights.pth --evaluate-only
"""

import time
import traceback

from src.models.factory import ModelFactory


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _print_metrics(metrics: dict):
    per_class = metrics.pop("per_class_ap", [])
    print("\n" + "=" * 50)
    print("  Evaluation results")
    print("=" * 50)
    for key, val in metrics.items():
        label = key.replace("_", " ").upper()
        print(f"  {label:<18} {val:.4f}")
    if per_class:
        print("  " + "-" * 46)
        print("  Per-class AP@0.5:")
        for i, ap in enumerate(per_class):
            print(f"    class {i:<4} {ap:.4f}")
    print("=" * 50 + "\n")
    metrics["per_class_ap"] = per_class


def train_model(
    model_name: str,
    cfg: dict,
    weights_path: str | None = None,
    evaluate_only: bool = False,
):
    """
    Args:
        model_name:    One of the keys registered in ModelFactory.
        cfg:           Full config dict loaded from default.yaml.
        weights_path:  Optional path to pre-trained weights.
        evaluate_only: If True, skip training and only run evaluate().
    """
    print(f"\n{'=' * 50}")
    print(f"  Model  : {model_name}")
    print(f"  Mode   : {'evaluate only' if evaluate_only else 'train + evaluate'}")
    if weights_path:
        print(f"  Weights: {weights_path}")
    print(f"{'=' * 50}\n")

    try:
        model = ModelFactory.get_model(model_name, cfg)
    except ValueError as e:
        print(f"[ERROR] Could not create model: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error while creating model:\n{traceback.format_exc()}")
        return

    if weights_path:
        try:
            model.load_model(weights_path)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
        except Exception as e:
            print(f"[ERROR] Failed to load weights:\n{traceback.format_exc()}")
            return

    if not evaluate_only:
        print(f"Starting training for '{model_name}'...\n")
        t_start = time.time()
        try:
            model.train()
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.")
        except Exception:
            print(f"[ERROR] Training failed:\n{traceback.format_exc()}")
            return

        elapsed = time.time() - t_start
        print(f"\n[INFO] Training finished in {_format_duration(elapsed)}.")

    print(f"\nRunning final evaluation for '{model_name}'...")
    try:
        metrics = model.evaluate()
        _print_metrics(metrics)
    except Exception:
        print(f"[ERROR] Evaluation failed:\n{traceback.format_exc()}")