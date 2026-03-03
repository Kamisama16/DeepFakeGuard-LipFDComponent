"""
DFDC Benchmark — evaluate LipFD detector on the DFDC cropped dataset.

Usage:
    # Full benchmark (all 3293 videos)
    python benchmark_dfdc.py

    # Quick test (first N per class)
    python benchmark_dfdc.py --max-per-class 50

    # Custom paths
    python benchmark_dfdc.py --dataset DFDC_Dataset --weights weights/lipfd_ckpt.pth
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np


def gather_videos(dataset_dir: str, max_per_class: int | None = None):
    """Collect video paths and labels from DFDC dataset structure."""
    fake_dir = os.path.join(dataset_dir, "Fake")
    real_dir = os.path.join(dataset_dir, "Real")

    fakes = sorted([
        (os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir)
        if f.endswith((".mp4", ".avi", ".mkv"))
    ])
    reals = sorted([
        (os.path.join(real_dir, f), 0) for f in os.listdir(real_dir)
        if f.endswith((".mp4", ".avi", ".mkv"))
    ])

    if max_per_class:
        fakes = fakes[:max_per_class]
        reals = reals[:max_per_class]

    return fakes + reals


def compute_metrics(y_true, y_scores, threshold=0.5):
    """Compute classification metrics."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # AUROC (simple trapezoidal)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except ImportError:
        auroc = float("nan")
        ap = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "auroc": auroc,
        "ap": ap,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="DFDC Benchmark for LipFD")
    parser.add_argument("--dataset", default="DFDC_Dataset", help="Path to DFDC dataset")
    parser.add_argument("--weights", default="weights/lipfd_ckpt.pth", help="Path to LipFD weights")
    parser.add_argument("--max-per-class", type=int, default=None, help="Limit videos per class (for quick testing)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--output-csv", default=None, help="Save per-video results to CSV")
    args = parser.parse_args()

    from deepfake_guard.models.lipfd import LipFDDetector

    print("=" * 65)
    print("  DFDC Benchmark — LipFD Detector")
    print("=" * 65)

    # Gather videos
    videos = gather_videos(args.dataset, args.max_per_class)
    n_fake = sum(1 for _, l in videos if l == 1)
    n_real = sum(1 for _, l in videos if l == 0)
    print(f"\n  Dataset:    {args.dataset}")
    print(f"  Videos:     {len(videos)} ({n_fake} fake, {n_real} real)")
    print(f"  Weights:    {args.weights}")
    print(f"  Threshold:  {args.threshold}")
    print()

    # Load model
    print("Loading model...")
    det = LipFDDetector(weights_path=args.weights, threshold=args.threshold)
    print(f"  {det}\n")

    # Run inference
    y_true = []
    y_scores = []
    results = []
    errors = 0
    start_time = time.time()

    for i, (path, label) in enumerate(videos):
        fname = os.path.basename(path)
        folder = "Fake" if label == 1 else "Real"

        try:
            r = det.predict_video(path, batch_size=args.batch_size)
            score = r["overall_score"]
            pred_label = r["overall_label"]
        except Exception as e:
            score = 0.0
            pred_label = "ERROR"
            errors += 1

        y_true.append(label)
        y_scores.append(score)
        results.append({
            "file": fname,
            "folder": folder,
            "ground_truth": label,
            "pred_score": score,
            "pred_label": pred_label,
        })

        correct = (label == 1 and pred_label == "FAKE") or \
                  (label == 0 and pred_label == "REAL")

        # Progress update every 25 videos
        if (i + 1) % 25 == 0 or (i + 1) == len(videos):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(videos) - i - 1) / rate if rate > 0 else 0
            # Running accuracy
            running_pred = (np.array(y_scores) >= args.threshold).astype(int)
            running_acc = np.mean(running_pred == np.array(y_true))
            print(f"  [{i+1:>4}/{len(videos)}]  acc={running_acc:.3f}  "
                  f"rate={rate:.1f} vid/s  ETA={eta:.0f}s  "
                  f"last: {folder}/{fname} -> {pred_label} ({score:.4f})")

    total_time = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(y_true, y_scores, args.threshold)

    print(f"\n{'=' * 65}")
    print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  Total videos:    {metrics['total']}")
    print(f"  Errors:          {errors}")
    print(f"  Time:            {total_time:.1f}s ({total_time/len(videos):.2f}s/video)")
    print(f"")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  AUROC:           {metrics['auroc']:.4f}")
    print(f"  Avg Precision:   {metrics['ap']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")
    print(f"  FNR:             {metrics['fnr']:.4f}")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"                   Pred REAL    Pred FAKE")
    print(f"    True REAL       {metrics['tn']:>6}       {metrics['fp']:>6}")
    print(f"    True FAKE       {metrics['fn']:>6}       {metrics['tp']:>6}")
    print(f"{'=' * 65}")

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Per-video results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
