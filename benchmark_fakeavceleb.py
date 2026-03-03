"""
FakeAVCeleb Benchmark — evaluate LipFD detector on FakeAVCeleb v1.2.

Supports two dataset layouts:
  1. Full MP4 layout (official download):
       FakeAVCeleb_v1.2/
         RealVideo-RealAudio/  →  label=0 (REAL)
         FakeVideo-FakeAudio/  →  label=1 (FAKE)
         FakeVideo-RealAudio/  →  label=1 (FAKE, video deepfake / real audio)
         RealVideo-FakeAudio/  →  label=1 (FAKE, real video / audio deepfake)

  2. Frames layout (Kaggle version — extracted frames, no audio):
       FakeAVCeleb_v1.2/
         frames/
           RealVideo-RealAudio/<person>/<video_id>/*.jpg
           FakeVideo-FakeAudio/...
           ...
         moved/  (may contain some MP4s)

Usage:
    # Auto-detect layout and run full benchmark
    python benchmark_fakeavceleb.py --dataset FakeAVCeleb_v1.2

    # Quick test (50 videos per class)
    python benchmark_fakeavceleb.py --dataset FakeAVCeleb_v1.2 --max-per-class 50

    # Specific subset only (FakeVideo-FakeAudio vs RealVideo-RealAudio)
    python benchmark_fakeavceleb.py --dataset FakeAVCeleb_v1.2 --subset fakevideo_fakeaudio

    # Save results CSV
    python benchmark_fakeavceleb.py --dataset FakeAVCeleb_v1.2 --output-csv results_fakeavceleb.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

# ---------------------------------------------------------------------------
# Label mapping — folder name → (label, category)
# ---------------------------------------------------------------------------
FOLDER_LABELS = {
    "RealVideo-RealAudio": (0, "real"),
    "FakeVideo-FakeAudio": (1, "fake_both"),
    "FakeVideo-RealAudio": (1, "fake_video"),
    "RealVideo-FakeAudio": (1, "fake_audio"),
}

# Subset presets for --subset argument
SUBSET_PRESETS = {
    "all":                 list(FOLDER_LABELS.keys()),
    "fakevideo_fakeaudio": ["RealVideo-RealAudio", "FakeVideo-FakeAudio"],
    "fakevideo_realaudio": ["RealVideo-RealAudio", "FakeVideo-RealAudio"],
    "realvideo_fakeaudio": ["RealVideo-RealAudio", "RealVideo-FakeAudio"],
    "fake_only":           ["FakeVideo-FakeAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio"],
}


# ---------------------------------------------------------------------------
# Dataset structure detection
# ---------------------------------------------------------------------------
def resolve_dataset_root(dataset_dir: str) -> str:
    """
    Resolve the actual dataset root that contains the FakeAVCeleb category folders.
    Handles the Kaggle nested layout:
        FakeAVCeleb_v1.2/          ← user may point here
          FakeAVCeleb_v1.2/        ← actual root is one level deeper
            RealVideo-RealAudio/
            FakeVideo-FakeAudio/
            ...
    """
    d = Path(dataset_dir)
    # Check if any category folder exists directly
    for folder in FOLDER_LABELS:
        if (d / folder).exists():
            return str(d)
    # Check one level deeper
    for child in d.iterdir():
        if child.is_dir():
            for folder in FOLDER_LABELS:
                if (child / folder).exists():
                    return str(child)
    # Check inside frames/
    frames_sub = d / "frames"
    if frames_sub.exists():
        for folder in FOLDER_LABELS:
            if (frames_sub / folder).exists():
                return str(frames_sub)
    return str(d)


def detect_layout(dataset_dir: str) -> str:
    """Detect whether dataset is 'mp4', 'frames', or 'unknown'."""
    d = Path(dataset_dir)
    # Check for direct MP4 subdirs
    for folder in FOLDER_LABELS:
        p = d / folder
        if p.exists():
            mp4s = list(p.rglob("*.mp4"))
            if mp4s:
                return "mp4"
    # Check for frames/ subdirectory
    frames_dir = d / "frames"
    if frames_dir.exists():
        imgs = list(frames_dir.rglob("*.jpg")) + list(frames_dir.rglob("*.png"))
        if imgs:
            return "frames"
    # Check for moved/ or top-level MP4s under any subfolder
    moved_dir = d / "moved"
    if moved_dir.exists():
        mp4s = list(moved_dir.rglob("*.mp4"))
        if mp4s:
            return "moved_mp4"
    return "unknown"


# ---------------------------------------------------------------------------
# Gather items based on layout
# ---------------------------------------------------------------------------
def gather_mp4(dataset_dir: str, subfolders: list[str], max_per_class: int | None) -> list[tuple[str, int, str]]:
    """Collect (path, label, category) from MP4 layout."""
    items: list[tuple[str, int, str]] = []
    per_class_counts: dict[int, int] = {0: 0, 1: 0}
    d = Path(dataset_dir)
    for folder in subfolders:
        label, category = FOLDER_LABELS[folder]
        sub = d / folder
        if not sub.exists():
            continue
        # Sort by full path for determinism; skip .txt sidecar files
        mp4s = sorted(sub.rglob("*.mp4"))
        for p in mp4s:
            if max_per_class and per_class_counts[label] >= max_per_class:
                break
            items.append((str(p), label, category))
            per_class_counts[label] += 1
    return items


def gather_frames(dataset_dir: str, subfolders: list[str], max_per_class: int | None) -> list[tuple[str, int, str, str]]:
    """Collect (frame_dir, label, category, video_id) from frames layout."""
    items: list[tuple[str, int, str, str]] = []
    per_class_counts: dict[int, int] = {0: 0, 1: 0}
    d = Path(dataset_dir) / "frames"
    if not d.exists():
        d = Path(dataset_dir)

    for folder in subfolders:
        label, category = FOLDER_LABELS[folder]
        base = d / folder
        if not base.exists():
            continue
        # Find all leaf directories that contain image files
        for root, dirs, files in os.walk(str(base)):
            imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not imgs:
                continue
            if max_per_class and per_class_counts[label] >= max_per_class:
                break
            video_id = Path(root).name
            items.append((root, label, category, video_id))
            per_class_counts[label] += 1
    return items


def frames_to_mp4(frame_dir: str, out_path: str, fps: float = 25.0) -> bool:
    """Convert a directory of frames to a silent MP4 using ffmpeg."""
    imgs = sorted(
        [f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if not imgs:
        return False
    # Use glob pattern; frames usually named 0001.jpg, frame_001.jpg etc.
    ext = Path(imgs[0]).suffix
    # Write a list file for ffmpeg concat
    list_file = out_path + "_frames.txt"
    try:
        with open(list_file, "w") as f:
            for img in imgs:
                f.write(f"file '{os.path.join(frame_dir, img)}'\n")
                f.write(f"duration {1.0 / fps:.6f}\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_scores, threshold=0.5):
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

    try:
        import warnings
        from sklearn.metrics import roc_auc_score, average_precision_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auroc = roc_auc_score(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
    except (ImportError, ValueError):
        auroc = float("nan")
        ap = float("nan")

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "fpr": fpr, "fnr": fnr,
        "auroc": auroc, "ap": ap,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": total,
    }


def print_metrics(metrics: dict, total_time: float, errors: int, n_videos: int):
    sec_per_vid = total_time / n_videos if n_videos else 0
    print(f"\n{'=' * 65}")
    print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  Total videos:    {metrics['total']}")
    print(f"  Errors:          {errors}")
    print(f"  Time:            {total_time:.1f}s ({sec_per_vid:.2f}s/video)")
    print()
    print(f"  Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  AUROC:           {metrics['auroc']:.4f}")
    print(f"  Avg Precision:   {metrics['ap']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")
    print(f"  FNR:             {metrics['fnr']:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"                   Pred REAL    Pred FAKE")
    print(f"    True REAL       {metrics['tn']:>6}       {metrics['fp']:>6}")
    print(f"    True FAKE       {metrics['fn']:>6}       {metrics['tp']:>6}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------
def print_category_breakdown(results: list[dict], threshold: float):
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"y_true": [], "y_scores": []}
        categories[cat]["y_true"].append(r["ground_truth"])
        categories[cat]["y_scores"].append(r["pred_score"])

    if len(categories) <= 1:
        return

    print(f"\n  Per-Category Breakdown:")
    print(f"  {'Category':<25} {'N':>5}  {'Acc':>6}  {'AUROC':>6}  {'F1':>6}")
    print(f"  {'-'*55}")
    for cat, data in sorted(categories.items()):
        try:
            m = compute_metrics(data["y_true"], data["y_scores"], threshold)
            print(f"  {cat:<25} {m['total']:>5}  {m['accuracy']:>6.3f}  {m['auroc']:>6.3f}  {m['f1']:>6.3f}")
        except Exception:
            print(f"  {cat:<25}  (insufficient data)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FakeAVCeleb Benchmark for LipFD")
    parser.add_argument("--dataset", default="FakeAVCeleb_v1.2",
                        help="Path to FakeAVCeleb dataset root")
    parser.add_argument("--weights", default="weights/lipfd_ckpt.pth",
                        help="Path to LipFD weights")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Limit videos per class label (for quick testing)")
    parser.add_argument("--subset", default="all",
                        choices=list(SUBSET_PRESETS.keys()),
                        help="Which category subset to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Detection threshold")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-csv", default=None,
                        help="Save per-video results to CSV file")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="FPS used when reconstructing videos from frames")
    args = parser.parse_args()

    from deepfake_guard.models.lipfd import LipFDDetector

    print("=" * 65)
    print("  FakeAVCeleb Benchmark — LipFD Detector")
    print("=" * 65)

    dataset_dir = args.dataset
    if not os.path.exists(dataset_dir):
        print(f"\n[ERROR] Dataset directory not found: {dataset_dir}")
        print("  Use --dataset to specify the path to FakeAVCeleb_v1.2.")
        sys.exit(1)

    # Resolve actual root (handles Kaggle nested structure)
    resolved_dir = resolve_dataset_root(dataset_dir)
    if resolved_dir != dataset_dir:
        print(f"\n  Resolved nested dataset root: {resolved_dir}")
    dataset_dir = resolved_dir

    # Detect layout
    layout = detect_layout(dataset_dir)
    print(f"\n  Dataset:    {dataset_dir}")
    print(f"  Layout:     {layout}")
    print(f"  Subset:     {args.subset}")
    print(f"  Weights:    {args.weights}")
    print(f"  Threshold:  {args.threshold}")

    subfolders = SUBSET_PRESETS[args.subset]

    if layout == "mp4":
        items = gather_mp4(dataset_dir, subfolders, args.max_per_class)
        mode = "mp4"
    elif layout in ("frames", "unknown"):
        # Try frames layout (Kaggle version)
        frame_items = gather_frames(dataset_dir, subfolders, args.max_per_class)
        if frame_items:
            items = frame_items
            mode = "frames"
        else:
            print(f"\n[ERROR] No recognizable FakeAVCeleb content found in: {dataset_dir}")
            print("  Expected subdirectories: RealVideo-RealAudio, FakeVideo-FakeAudio, etc.")
            print("  Or a 'frames/' subdirectory with those categories.")
            sys.exit(1)
    elif layout == "moved_mp4":
        # Fall back: try moved/ for MP4s, still use frames structure for labels
        items = gather_frames(dataset_dir, subfolders, args.max_per_class)
        mode = "frames"
    else:
        print(f"[ERROR] Could not detect dataset layout.")
        sys.exit(1)

    n_fake = sum(1 for it in items if it[1] == 1)
    n_real = sum(1 for it in items if it[1] == 0)
    print(f"\n  Items:      {len(items)} ({n_fake} fake, {n_real} real)")
    print(f"  Mode:       {mode}")
    if mode == "frames":
        print(f"  Note:       Frames will be reconstructed into silent MP4s via ffmpeg.")
        print(f"              Audio pathway will produce blank mel-spectrograms.")
    print()

    # Load detector
    print("Loading model...")
    det = LipFDDetector(weights_path=args.weights, threshold=args.threshold)
    print(f"  {det}\n")

    # Temp dir for frames→MP4 conversion
    tmp_dir = tempfile.mkdtemp(prefix="fakeavceleb_bench_") if mode == "frames" else None

    y_true, y_scores = [], []
    results = []
    errors = 0
    start_time = time.time()

    for i, item in enumerate(items):
        if mode == "mp4":
            path, label, category = item
            fname = os.path.basename(path)
        else:
            frame_dir, label, category, video_id = item
            fname = video_id
            # Reconstruct video from frames
            out_mp4 = os.path.join(tmp_dir, f"{category}_{video_id}.mp4")
            if not frames_to_mp4(frame_dir, out_mp4, fps=args.fps):
                errors += 1
                y_true.append(label)
                y_scores.append(0.0)
                results.append({
                    "video_id": fname, "category": category,
                    "ground_truth": label, "pred_score": 0.0,
                    "pred_label": "ERROR",
                })
                continue
            path = out_mp4

        try:
            r = det.predict_video(path, batch_size=args.batch_size)
            score = r["overall_score"]
            pred_label = r["overall_label"]
        except Exception as e:
            score = 0.0
            pred_label = "ERROR"
            errors += 1

        # Clean up converted MP4 to save disk space
        if mode == "frames" and os.path.exists(path):
            os.remove(path)

        y_true.append(label)
        y_scores.append(score)
        results.append({
            "video_id": fname, "category": category,
            "ground_truth": label, "pred_score": score,
            "pred_label": pred_label,
        })

        # Progress every 25 items
        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(items) - i - 1) / rate if rate > 0 else 0
            running_pred = (np.array(y_scores) >= args.threshold).astype(int)
            running_acc = np.mean(running_pred == np.array(y_true))
            gt_str = "FAKE" if label == 1 else "REAL"
            print(f"  [{i+1:>5}/{len(items)}]  acc={running_acc:.3f}  "
                  f"rate={rate:.1f}/s  ETA={eta:.0f}s  "
                  f"[{category}] {fname} → {pred_label} ({score:.4f}) gt={gt_str}")

    # Clean up temp dir
    if tmp_dir and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_time = time.time() - start_time
    metrics = compute_metrics(y_true, y_scores, args.threshold)
    print_metrics(metrics, total_time, errors, len(items))
    print_category_breakdown(results, args.threshold)

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Per-video results saved to: {args.output_csv}")

    print()


if __name__ == "__main__":
    main()
