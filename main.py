"""
DeepFakeGuard — LipFD Detector Demo

Demonstrates how the LipFD detector (trained on AVLips dataset) integrates
with the DeepFakeGuard library as a fifth detector backend.

Usage:
    # Pipeline smoke test (no weights, no video)
    python main.py

    # With a real video (no weights — random model)
    python main.py --video path/to/video.mp4

    # With pretrained weights
    python main.py --video path/to/video.mp4 --weights weights/lipfd_ckpt.pth

    # Select CLIP architecture
    python main.py --video video.mp4 --arch "CLIP:ViT-B/32"
"""

from __future__ import annotations

import argparse
import json
import sys
import os

# Add src/ to path so the package is importable during development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def smoke_test() -> None:
    """Run a pipeline smoke test with synthetic data (no video needed)."""
    import torch
    import numpy as np

    print("=" * 60)
    print("  LipFD Detector — Pipeline Smoke Test")
    print("=" * 60)

    # 1. Test model instantiation
    print("\n[1/5] Building LipFD model (CLIP:ViT-L/14) ...")
    from deepfake_guard.models.lipfd.model import build_model
    model = build_model("CLIP:ViT-L/14")
    print(f"       ✓ Model created: {type(model).__name__}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       Total params:     {total_params:>12,}")
    print(f"       Trainable params: {trainable:>12,}")

    # 2. Test forward pass with dummy data
    print("\n[2/5] Running forward pass with synthetic tensors ...")
    model.eval()
    with torch.no_grad():
        # Simulate full composite image: (1, 3, 1120, 1120)
        dummy_img = torch.randn(1, 3, 1120, 1120)
        features = model.get_features(dummy_img)
        print(f"       ✓ CLIP features shape: {features.shape}")

        # Simulate crops: 3 scales × 5 frames, each (1, 3, 224, 224)
        dummy_crops = [
            [torch.randn(1, 3, 224, 224) for _ in range(5)]
            for _ in range(3)
        ]
        pred, w_max, w_org = model(dummy_crops, features)
        prob = torch.sigmoid(pred).item()
        print(f"       ✓ Prediction logit: {pred.item():.4f}")
        print(f"       ✓ Sigmoid prob:     {prob:.4f}")
        print(f"       ✓ Label:            {'FAKE' if prob >= 0.5 else 'REAL'}")

    # 3. Test RA Loss
    print("\n[3/5] Testing Region-Awareness Loss ...")
    from deepfake_guard.models.lipfd.model import get_loss
    ra_loss = get_loss()
    loss_val = ra_loss(w_max, w_org)
    print(f"       ✓ RA Loss value: {loss_val.item():.4f}")

    # 4. Test preprocessing utilities
    print("\n[4/5] Testing preprocessing utilities ...")
    from deepfake_guard.models.lipfd.preprocessing import (
        composite_to_tensors,
    )
    dummy_composite = np.random.randint(0, 255, (1000, 2500, 3), dtype=np.uint8)
    full_img, crops = composite_to_tensors(dummy_composite)
    print(f"       ✓ Full image shape:    {full_img.shape}")
    print(f"       ✓ Crop scales:         {len(crops)}")
    print(f"       ✓ Frames per scale:    {len(crops[0])}")
    print(f"       ✓ Crop tensor shape:   {crops[0][0].shape}")

    # 5. Test detector class
    print("\n[5/5] Testing LipFDDetector class ...")
    from deepfake_guard.models.lipfd import LipFDDetector
    det = LipFDDetector(device="cpu")
    print(f"       ✓ {det}")

    print("\n" + "=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60)


def detect_video(
    video_path: str,
    weights_path: str | None = None,
    arch: str = "CLIP:ViT-L/14",
    threshold: float = 0.5,
) -> None:
    """Run detection on a real video file."""
    from deepfake_guard.models.lipfd import LipFDDetector

    print("=" * 60)
    print("  LipFD Detector — Video Analysis")
    print("=" * 60)
    print(f"\n  Video:     {video_path}")
    print(f"  Arch:      {arch}")
    print(f"  Weights:   {weights_path or '(none — random model)'}")
    print(f"  Threshold: {threshold}")
    print()

    det = LipFDDetector(
        weights_path=weights_path,
        arch=arch,
        threshold=threshold,
    )

    print("Analysing video ...")
    result = det.predict_video(video_path)

    print(f"\n{'─' * 40}")
    print(f"  Label:  {result['overall_label']}")
    print(f"  Score:  {result['overall_score']}")
    print(f"{'─' * 40}")

    if result.get("errors"):
        print("\n  Warnings:")
        for err in result["errors"]:
            print(f"    ⚠ {err}")

    print("\nFull result:")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepFakeGuard — LipFD Detector Demo",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a video file to analyse.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to pretrained LipFD checkpoint (.pth).",
    )
    parser.add_argument(
        "--arch", type=str, default="CLIP:ViT-L/14",
        choices=["CLIP:ViT-B/32", "CLIP:ViT-B/16", "CLIP:ViT-L/14"],
        help="CLIP architecture (must match checkpoint).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold (0-1).",
    )
    args = parser.parse_args()

    if args.video:
        detect_video(args.video, args.weights, args.arch, args.threshold)
    else:
        smoke_test()


if __name__ == "__main__":
    main()
