"""
Logger — writes per-epoch JSON logs and saves training/eval plots.

Directory layout created automatically:
    results/
        logs/   <model_name>_log.json
        plots/  <model_name>_loss.png
                <model_name>_metrics.png
                <model_name>_per_class_ap.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on servers
import matplotlib.pyplot as plt


class TrainingLogger:
    """
    Accumulates epoch records and flushes them to disk after every epoch.

    Usage:
        logger = TrainingLogger(results_dir="results", model_name="faster_rcnn")
        logger.log_epoch(epoch=1, loss=0.42, metrics={...})
        logger.save_plots()          # call once at end of training
    """

    def __init__(self, results_dir: str, model_name: str):
        self.model_name  = model_name
        self.logs_dir    = os.path.join(results_dir, "logs")
        self.plots_dir   = os.path.join(results_dir, "plots")
        os.makedirs(self.logs_dir,  exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.log_path = os.path.join(self.logs_dir, f"{model_name}_log.json")
        self.records: list = []

        # Load existing log so we can append across resumed runs
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.records = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int, loss: float, metrics: dict | None = None):
        """
        Add one epoch record.  metrics should be the dict returned by
        compute_detection_metrics(), or None if evaluate() was not called.
        """
        record = {"epoch": epoch, "loss": round(float(loss), 6)}
        if metrics:
            record.update({k: v for k, v in metrics.items()
                           if k != "per_class_ap"})          # scalars only
            record["per_class_ap"] = metrics.get("per_class_ap", [])

        self.records.append(record)
        self._flush()
        self._print_epoch(record)

    def save_plots(self):
        """Generate and save all plots.  Call once after training is done."""
        if not self.records:
            return
        self._plot_loss()
        if "mAP" in self.records[0]:
            self._plot_metrics()
            self._plot_per_class_ap()
        print(f"[Logger] Plots saved to '{self.plots_dir}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush(self):
        with open(self.log_path, "w") as f:
            json.dump(self.records, f, indent=2)

    @staticmethod
    def _print_epoch(record: dict):
        parts = [f"Epoch {record['epoch']}",
                 f"loss={record['loss']:.4f}"]
        for key in ("mAP", "precision", "recall", "f1", "mean_iou"):
            if key in record:
                parts.append(f"{key}={record[key]:.4f}")
        print(" | ".join(parts))

    def _epochs(self):
        return [r["epoch"] for r in self.records]

    def _plot_loss(self):
        losses = [r["loss"] for r in self.records]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self._epochs(), losses, marker="o", linewidth=2,
                color="#e05c5c", label="Train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.model_name} — Training Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(self.plots_dir, f"{self.model_name}_loss.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Logger] Loss plot → {path}")

    def _plot_metrics(self):
        metrics = {
            "mAP":       ("#4c72b0", "mAP@0.5"),
            "precision": ("#55a868", "Precision"),
            "recall":    ("#c44e52", "Recall"),
            "f1":        ("#8172b2", "F1"),
            "mean_iou":  ("#ccb974", "Mean IoU"),
        }
        epochs = self._epochs()
        fig, ax = plt.subplots(figsize=(10, 5))
        for key, (color, label) in metrics.items():
            vals = [r.get(key) for r in self.records]
            if any(v is not None for v in vals):
                ax.plot(epochs, vals, marker="o", linewidth=2,
                        color=color, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"{self.model_name} — Evaluation Metrics")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(self.plots_dir, f"{self.model_name}_metrics.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Logger] Metrics plot → {path}")

    def _plot_per_class_ap(self):
        """Bar chart of per-class AP from the last logged epoch."""
        last = next(
            (r for r in reversed(self.records) if r.get("per_class_ap")),
            None
        )
        if last is None:
            return

        ap_vals = last["per_class_ap"]
        classes = [f"cls {i}" for i in range(len(ap_vals))]

        fig, ax = plt.subplots(figsize=(max(8, len(classes)), 4))
        bars = ax.bar(classes, ap_vals, color="#4c72b0", edgecolor="white")
        for bar, val in zip(bars, ap_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Class")
        ax.set_ylabel("AP@0.5")
        ax.set_title(f"{self.model_name} — Per-Class AP "
                     f"(epoch {last['epoch']})")
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(self.plots_dir,
                            f"{self.model_name}_per_class_ap.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Logger] Per-class AP plot → {path}")