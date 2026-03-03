# DeepFakeGuard — LipFD Component

Audio-visual lip-sync deepfake detector for the [DeepFakeGuard](https://github.com/aryanbiswas16/DeepFakeGuard) multimodal detection toolkit.

Implements the **LipFD** model from:

> **Liu et al.**, *"Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes"*, **NeurIPS 2024**.
> [arXiv:2401.15668](https://arxiv.org/abs/2401.15668) · [Original Code](https://github.com/AaronComo/LipFD)

## Architecture

```
Video ──► Frame extraction ──► Composite image (mel-spectrogram + frames)
                                    │
                     ┌──────────────┼──────────────┐
                     ▼                             ▼
              5×5 stride-5 Conv                Multi-scale crops
                     │                        (1.0×, 0.65×, 0.45×)
                     ▼                             │
              CLIP ViT-L/14                        ▼
            (frozen encoder)              Region-Aware ResNet-50
                     │                        (attention-weighted)
                     └──────────┬──────────────────┘
                                ▼
                       Concatenate + FC
                                ▼
                     Binary prediction (REAL/FAKE)
```

- **CLIP pathway**: Extracts global audio-visual features from the full composite image
- **Region-Aware pathway**: Processes multi-scale face crops with learned attention weights
- **RA-Loss**: Encourages the model to focus on the most discriminative facial region

## Project Structure

```
├── src/deepfake_guard/models/lipfd/
│   ├── __init__.py           # Package exports
│   ├── model.py              # LipFD network + RA-Loss
│   ├── region_awareness.py   # Region-Aware ResNet-50 backbone
│   ├── preprocessing.py      # Video → tensor pipeline
│   └── detector.py           # DeepFakeGuard-compatible wrapper
├── benchmark_dfdc.py          # DFDC dataset benchmark script
├── benchmark_fakeavceleb.py   # FakeAVCeleb dataset benchmark script
├── main.py                    # Demo & smoke test
├── weights/                   # Pretrained weights (download separately)
│   └── lipfd_ckpt.pth        # 1.68 GB — not included in repo
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Download pretrained weights (1.68 GB)
# Place as weights/lipfd_ckpt.pth
# Source: https://github.com/AaronComo/LipFD
```

## Usage

```python
from deepfake_guard.models.lipfd import LipFDDetector

detector = LipFDDetector(weights_path="weights/lipfd_ckpt.pth")
result = detector.predict_video("video.mp4")

print(result["overall_label"])  # "REAL" or "FAKE"
print(result["overall_score"])  # 0.0 (real) to 1.0 (fake)
```

```bash
# Smoke test (no weights needed)
python main.py

# Analyse a video
python main.py --video video.mp4 --weights weights/lipfd_ckpt.pth
```

## Benchmark Results

### FakeAVCeleb v1.2 (250 per class, balanced)

| Deepfake Type | Accuracy | AUROC | Recall | F1 |
|---|---|---|---|---|
| FakeVideo-FakeAudio (lip-sync + synth audio) | **91.2%** | **0.962** | 94.8% | 0.915 |
| FakeVideo-RealAudio (face-swap + real audio) | 75.2% | 0.821 | 62.8% | 0.717 |
| RealVideo-FakeAudio (real video + synth audio) | 49.6% | 0.513 | 11.6% | 0.187 |

### DFDC (800 videos, face-swap, no audio)

| Metric | Score |
|---|---|
| Accuracy | 72.3% |

### Wav2Lip Videos (3 lip-sync deepfakes)

| Metric | Score |
|---|---|
| Detection Rate | 0% (all scored as REAL) |

## Credits

- **LipFD Paper**: Liu et al., NeurIPS 2024 — [arXiv:2401.15668](https://arxiv.org/abs/2401.15668)
- **LipFD Code**: [AaronPeng920/LipFD](https://github.com/AaronComo/LipFD)
- **CLIP**: [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License)
- **ResNet-50**: torchvision (BSD-3-Clause License)
- **FakeAVCeleb Dataset**: [DASH-Lab/FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
- **DFDC Dataset**: [Deep Fake Detection Cropped Dataset](https://www.kaggle.com/datasets/ucimachinelearning/deep-fake-detection-cropped-dataset)
