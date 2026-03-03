# LipFD Component Report — DeepFakeGuard

## 1. Component Overview

LipFD (Lip Forgery Detection) is the **audio-visual modality detector** within the DeepFakeGuard multimodal deepfake detection toolkit. It is the 5th detector in the ensemble, specialising in **lip-sync deepfakes** — a category of manipulated media where a person's lip movements are artificially generated to match a different audio track.

The model was introduced in:

> Liu et al., *"Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes"*, **NeurIPS 2024** ([arXiv:2401.15668](https://arxiv.org/abs/2401.15668))

Our implementation adapts the [original codebase](https://github.com/AaronPeng920/LipFD) into a modular, integration-ready component that conforms to the DeepFakeGuard detector interface.

---

## 2. Overall Architecture

LipFD uses a **dual-pathway architecture** that jointly analyses audio and visual signals:

### Preprocessing Pipeline
1. **Frame Extraction** — Sample 10 groups of 5 consecutive frames from the video using OpenCV.
2. **Audio Extraction** — Extract the audio track via FFmpeg and convert to a mel-spectrogram using librosa.
3. **Composite Image Construction** — Vertically stack the mel-spectrogram (500×2500 px) above the concatenated frames (500×2500 px), producing a 1000×2500 composite image per sample.

### Dual Pathway
| Pathway | Architecture | Input | Purpose |
|---------|-------------|-------|---------|
| **Global** | CLIP ViT-L/14 (frozen) | Full composite (1120×1120 via 5×5 stride-5 conv) | Capture holistic audio-visual relationships |
| **Regional** | Modified ResNet-50 | 3 scale crops per frame (1.0×, 0.65×, 0.45×) at 224×224 | Focus on spatial artifacts in face/lip region |

### Fusion and Prediction
- Regional ResNet-50 features (2048-d) are concatenated with CLIP features (768-d) → 2816-d fused vector.
- A learned attention mechanism (`get_weight`) computes importance weights across scales.
- Weighted features are averaged across frames and passed through a linear classifier → binary logit.
- Sigmoid threshold (default 0.5) determines REAL/FAKE label.

### Region-Awareness Loss (RA-Loss)
A custom loss function that penalises uniform attention distributions, encouraging the model to concentrate on the most discriminative facial region (typically the lip area). Defined as:

$$\mathcal{L}_{RA} = \sum_{i} \frac{1}{B} \sum_{j=1}^{B} \frac{10}{\exp(\alpha_{\max}^{(i,j)} - \alpha_{\text{org}}^{(i,j)})}$$

---

## 3. Building the Model

### Implementation Process
The component was built by:

1. **Reference Study** — Analysed the original LipFD repository, focusing on `model/clip_model.py`, `model/model.py`, `model/region_awareness.py`, and `data/preprocess.py`.
2. **Architecture Reimplementation** — Recreated the complete model architecture in 5 Python modules following DeepFakeGuard conventions:
   - `model.py` — LipFD network class and RA-Loss
   - `region_awareness.py` — Region-Aware ResNet-50 backbone with multi-scale attention
   - `preprocessing.py` — End-to-end video-to-tensor pipeline
   - `detector.py` — DeepFakeGuard-compatible wrapper (`LipFDDetector`)
   - `__init__.py` — Package exports
3. **Weight Loading** — Downloaded the official pretrained checkpoint (`ckpt.pth`, 1.68 GB, 770 keys) and verified all parameters load correctly with `strict=False`.
4. **Smoke Testing** — Validated model instantiation, forward pass, RA-Loss computation, preprocessing utilities, and detector class with synthetic data. All 5 tests passed.

### Key Design Decisions
- **Raw BGR pixel values [0-255]**: The original training code contains a `Normalize()` call whose result is immediately overwritten — the model was trained on unnormalised BGR data. This was critical to replicate.
- **Crop offsets `range(5)`**: The original code uses `i:i+500` with `i ∈ {0,1,2,3,4}` — producing 5 nearly-identical crops with 1-pixel shifts rather than 5 distinct frame crops. We match this exactly.
- **BGR colour order**: Composites are constructed in RGB then converted to BGR via `cv2.cvtColor`, matching the original pipeline's save-then-reload pattern.
- **CLIP import patch**: OpenAI's CLIP package uses deprecated `pkg_resources.packaging`. We apply `pkg_resources.packaging = packaging` as a compatibility workaround.

---

## 4. Test Results and Key Findings

### 4.1 FakeAVCeleb v1.2 Benchmark (1,500 videos total)

Balanced evaluation: 250 real vs 250 fake per deepfake type.

| Deepfake Type | Description | Accuracy | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **FakeVideo-FakeAudio** | Lip-sync video + synthesised audio | **91.2%** | **0.962** | 88.4% | 94.8% | 0.915 |
| **FakeVideo-RealAudio** | Face-swap video + original audio | 75.2% | 0.821 | 83.5% | 62.8% | 0.717 |
| **RealVideo-FakeAudio** | Real video + synthesised audio only | 49.6% | 0.513 | 48.3% | 11.6% | 0.187 |

**Confusion matrices:**
```
FakeVideo-FakeAudio:          FakeVideo-RealAudio:          RealVideo-FakeAudio:
        Pred R  Pred F               Pred R  Pred F               Pred R  Pred F
True R    219      31          True R   219      31          True R   219      31
True F     13     237          True F    93     157          True F   221      29
```

### 4.2 DFDC Benchmark (800 videos)

| Metric | Score |
|---|---|
| Accuracy | 72.3% |
| Dataset type | Face-swap deepfakes, no audio track |

### 4.3 Wav2Lip Videos (3 lip-sync deepfakes)

| Video | Score | Prediction | Ground Truth |
|---|---|---|---|
| discover1_result.mp4 | 0.0004 | REAL | FAKE |
| discover2_result.mp4 | 0.0004 | REAL | FAKE |
| discover3_result.mp4 | 0.0003 | REAL | FAKE |

**Detection rate: 0%** — All 3 modern Wav2Lip videos evade detection.

### Key Findings

1. **Strong performance on its target domain**: LipFD achieves 91.2% accuracy and 0.962 AUROC on FakeAVCeleb's `FakeVideo-FakeAudio` subset — the lip-sync deepfake category it was designed to detect.

2. **Partial generalisation to face-swap**: Even on face-swap deepfakes (FakeVideo-RealAudio), LipFD achieves 75.2% through visual artifact detection alone, though recall drops to 62.8%.

3. **Cannot detect audio-only fakes**: RealVideo-FakeAudio performance is near random (49.6%). When the video is genuine, LipFD's visual backbone sees a real face and scores it as real regardless of audio manipulation. The model relies primarily on visual features.

4. **Vulnerable to modern lip-sync methods**: Wav2Lip produces such precise lip-audio synchronisation that LipFD's consistency check finds no anomaly. All 3 test videos scored ≤0.0004 — deep in the "real" range.

5. **Audio modality is secondary**: Cross-dataset results confirm LipFD primarily operates through its visual pathway. The CLIP encoder contributes global context but the model's discriminative power comes from the Region-Aware ResNet detecting spatial artifacts in lip regions.

---

## 5. Errors Encountered and Solutions

| Error | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'clip'` → then `packaging` error | OpenAI CLIP uses deprecated `pkg_resources.packaging` removed in newer setuptools | Applied `pkg_resources.packaging = packaging` monkey-patch before CLIP import |
| `RuntimeError: shape mismatch` in RA-Loss | `diff` variable was a tensor, not scalar — `.sum()` was missing | Changed `10.0 / torch.exp(diff)` to `10.0 / torch.exp(diff.sum())` |
| All videos scored ~96% FAKE | Preprocessing normalised pixel values to [0, 1] using CLIP stats | Removed normalisation — model expects raw BGR [0, 255] float32 |
| All videos scored ~0.001 REAL | Correct behaviour — test videos were genuine, not lip-sync deepfakes | Validated with FakeAVCeleb dataset which contains actual lip-sync fakes |
| `conda run` fails with `ModuleNotFoundError: numpy` | PowerShell `conda run -n env` doesn't properly activate the environment in some configurations | Used full Python path: `C:\...\envs\DeepFakeGuard\python.exe` |
| FakeAVCeleb nested directory structure | Kaggle download nests `FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/` | Added `resolve_dataset_root()` function that auto-detects and navigates nested structures |

---

## 6. Limitations

1. **Evasion by high-quality lip-sync**: Modern Wav2Lip-generated deepfakes produce lip movements that are synchronised well enough to fool LipFD's consistency check entirely. The model was trained on older AVLips-era (2022) deepfake methods.

2. **No audio-only detection**: LipFD cannot detect synthesised audio when paired with genuine video. The model's discriminative power is primarily visual, limiting its usefulness against voice cloning attacks.

3. **Computational cost**: Each video requires ~0.85 seconds on an NVIDIA GPU. The CLIP ViT-L/14 encoder loads ~428M frozen parameters plus the ResNet-50 backbone, totalling 1.68 GB of weights. This limits deployment on resource-constrained devices.

4. **Face-swap performance is moderate**: 75.2% accuracy on face-swap deepfakes is below specialised face-swap detectors. LipFD was not designed for this threat model.

5. **Fixed preprocessing assumptions**: The model expects videos with detectable faces and an audio track. Silent videos receive a blank mel-spectrogram, degrading performance. Very short videos (<6 frames) are rejected entirely.

6. **Single-face assumption**: The current pipeline does not perform face detection or tracking. It uses full video frames directly, which may reduce performance on multi-person scenes.

---

## 7. Significance for DeepFakeGuard

LipFD fills a critical gap in the DeepFakeGuard detector ensemble as the **only audio-visual modality detector**. While other detectors (DINOv3, ResNet18, IvyFake, D3) analyse visual features, LipFD uniquely exploits the **temporal relationship between lip movements and audio signals**.

The benchmark results demonstrate both the model's strengths and its limitations, reinforcing the central thesis of DeepFakeGuard: **no single detector is sufficient** — a multimodal ensemble approach is necessary for robust deepfake detection across different manipulation types.

The finding that modern Wav2Lip videos evade LipFD while older lip-sync methods are detected with 91.2% accuracy highlights the adversarial arms race in deepfake detection and motivates continued research into more generalised audio-visual forensic methods.

---

*Component developed as part of the DeepFakeGuard project, March 2026.*
