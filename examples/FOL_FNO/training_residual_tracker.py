# training_residual_tracker.py

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


class TrainingResidualTracker:
    """
    Simple tracker for per-epoch training metrics.

    Usage:
        tracker = TrainingResidualTracker(out_dir, "otf_circular")
        for epoch in ...:
            tracker.log_epoch(epoch, total_loss, residual_rms_batch_mean)
        tracker.finalize()
    """

    def __init__(self, out_dir: str, tag: str = "train"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.tag = tag
        self.epochs = []
        self.total_losses = []
        self.residuals = []

        # CSV path
        self.csv_path = os.path.join(self.out_dir, f"{self.tag}_residual_rms.csv")

        # Write header immediately
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "total_loss", "residual_rms_batch_mean"])

    def log_epoch(self, epoch: int, total_loss: float, residual_rms: float):
        """Append one row to memory + CSV."""
        self.epochs.append(epoch)
        self.total_losses.append(total_loss)
        self.residuals.append(residual_rms)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, total_loss, residual_rms])

    def finalize(self):
        """Create a simple plot residual vs epoch (and optionally loss)."""
        if not self.epochs:
            return

        epochs = np.array(self.epochs)
        residuals = np.array(self.residuals)

        plt.figure(figsize=(6, 4))
        plt.semilogy(epochs, residuals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(r"residual\_rms\_batch\_mean")
        plt.title(f"Training residual RMS – {self.tag}")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        png_path = os.path.join(self.out_dir, f"{self.tag}_residual_rms.png")
        plt.savefig(png_path)
        plt.close()
