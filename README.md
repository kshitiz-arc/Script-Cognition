<div align="center">
   
```
   _____           _       __        ______                  _ __  _           
  / ___/__________(_)___  / /_      / ____/___  ____ _____  (_) /_(_)___  ____ 
  \__ \/ ___/ ___/ / __ \/ __/_____/ /   / __ \/ __ `/ __ \/ / __/ / __ \/ __ \
 ___/ / /__/ /  / / /_/ / /_/_____/ /___/ /_/ / /_/ / / / / / /_/ / /_/ / / / /
/____/\___/_/  /_/ .___/\__/      \____/\____/\__, /_/ /_/_/\__/_/\____/_/ /_/ 
                /_/                          /____/                            
```
   
### *Your handwriting knows how you feel. Now the model does too.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-FFD43B?style=for-the-badge&logo=python&logoColor=black)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Dataset: EMOTHAW](https://img.shields.io/badge/Dataset-EMOTHAW-A78BFA?style=for-the-badge)](https://archive.ics.uci.edu/dataset/520/emothaw)

<br/>

> **Depression. Anxiety. Stress.** — three invisible forces that shape lives.  
> This project reads them from the strokes of a pen.

<br/>

</div>

---

## 👁️ What Is This?

**EmotionDetectionUsingHandwriting** is a full machine learning pipeline that detects emotional states from online handwriting signals captured on digital tablets. Pen trajectory data — coordinates, pressure, velocity, tilt — gets rendered into grayscale images and fed into a CNN that classifies the writer's emotional state.

It uses the [**EMOTHAW**](https://archive.ics.uci.edu/dataset/520/emothaw) dataset (129 subjects, 7 handwriting tasks each) and labels derived from the **DASS-21** (Depression Anxiety Stress Scales) questionnaire.

```
  .svc file  ──▶  parse pen data  ──▶  render trajectory  ──▶  CNN  ──▶  emotion
```
## 🌐 Live Demo

👉 **Web Application:**  
https://script-cognition.streamlit.app/

> ⚠️ **Deployment Notice**  
> This Streamlit demo is intended for interface visualization and workflow demonstration only.  
> Due to deployment constraints (model size and training requirements), the hosted version does not include fully trained models.  
> 
> For accurate predictions and complete functionality, please run the application locally after training the models as described below.

---
---

## ✦ Features at a Glance

| | Feature | Detail |
|:---:|---|---|
| 🧠 | **Dual CNN Architectures** | EmotionCNN (2.7M params) · ResNet18 transfer learning (11.2M params) |
| 🎯 | **Three Emotion Dimensions** | Depression · Anxiety · Stress — train each independently |
| 📊 | **DASS-21 Labels** | Binary (Low / High) or full 5-class severity |
| 🔬 | **29+ Handwriting Features** | Velocity, pressure, curvature, stroke count, pen-lift frequency |
| 🌐 | **Streamlit Web App** | Upload `.svc` files · switch emotions · live trajectory viz |
| 🖥️ | **Powerful CLI** | Train · predict · visualize · extract — all from the terminal |
| ⚡ | **GPU Ready** | CUDA support out of the box |

---

## 📊 Dataset at a Glance

```
  ╔══════════════════════════════════════════════════════╗
  ║  EMOTHAW Dataset                                     ║
  ║                                                      ║
  ║  Subjects    ██████████████████████████████  129     ║
  ║  Tasks/user  ████████████████████           7        ║
  ║  Samples     ████████████████████████████   900+     ║
  ║  HW Features ████████████████████████████   29+      ║
  ║  Emotions    ██████████████                 3        ║
  ║                                                      ║
  ║  Format: .svc  →  X, Y, timestamp, pressure,         ║
  ║                    pen status, azimuth, altitude     ║
  ╚══════════════════════════════════════════════════════╝
```

**Emotion targets** (binary Low / High per DASS-21 thresholds):

- 🟣 **Depression** — Normal → Extremely Severe
- 🔵 **Anxiety** — Normal → Extremely Severe
- 🩷 **Stress** — Normal → Extremely Severe

---

## 🗂️ Project Structure

```
EmotionDetectionUsingHandwriting/
│
├── ⚙️  config.py                    central configuration
├── 🚀  main.py                      unified CLI entry point
├── 🌐  app.py                       Streamlit web interface
├── 🏃  run_app.py                   web app launcher helper
├── 📋  requirements.txt
│
├── 📂  preprocessing/
│   └──  svc_parser.py              .svc → NumPy arrays
│
├── 📂  data/
│   ├──  label_loader.py            DASS scores → class labels
│   └──  dataset.py                 PyTorch Dataset + DataLoader
│
├── 📂  features/
│   ├──  image_generator.py         trajectories → images
│   └──  signal_features.py         29+ handwriting features
│
├── 📂  models/
│   └──  cnn_model.py               EmotionCNN + EmotionResNet
│
├── 📂  training/
│   └──  train_cnn.py               training loop + early stopping
│
├── 📂  evaluation/
│   └──  evaluate.py                metrics, confusion matrix, plots
│
├── 📂  inference/
│   └──  predict.py                 run inference on new samples
│
├── 📂  utils/
│   └──  visualization.py           trajectory & dataset plots
│
└── 📂  outputs/
    ├──  models/                    saved checkpoints (.pth)
    ├──  plots/                     generated figures
    └──  image_cache/               cached trajectory images
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [EMOTHAW dataset](https://archive.ics.uci.edu/dataset/520/emothaw) at `../archive/DataEmothaw/`
- CUDA *(optional, for GPU acceleration)*

### Installation

```bash
# Clone the repo
git clone https://github.com/kshitiz-arc/EmotionDetectionUsingHandwriting.git
cd EmotionDetectionUsingHandwriting

# Install dependencies
pip install -r requirements.txt
```

Expected dataset layout:
```
../archive/DataEmothaw/
├── DASS_scores.xls
├── user00001/
├── user00002/
└── ...
```

---

## 🎮 Usage

### 1 · Train

```bash
# Single emotion
python main.py train --model cnn --emotion anxiety
python main.py train --model cnn --emotion depression
python main.py train --model cnn --emotion stress

# All three in one shot
python main.py train_all --model cnn

# Custom hyperparams
python main.py train --model resnet --emotion anxiety \
    --epochs 50 --batch_size 32 --lr 0.0001
```

Checkpoints save to `outputs/models/` as `best_cnn_anxiety.pth`, etc.

---

### 2 · Launch the Web App

```bash
python run_app.py
# or
streamlit run app.py
```

The app lets you switch emotion targets, upload `.svc` files, see confidence scores, and explore dataset statistics — all without touching the terminal again.

---

### 3 · Predict from the CLI

```bash
python main.py predict --svc_path "path/to/handwriting.svc" --emotion anxiety
```

```
══════════════════════════════════════════════════════
  HANDWRITING EMOTION PREDICTION
══════════════════════════════════════════════════════

  Input file  : u00001s00001_hw00001.svc
  Data points : 1474
  Dimension   : Anxiety

──────────────────────────────────────────────────────
  PREDICTED   : LOW
  CONFIDENCE  : 84.12%
──────────────────────────────────────────────────────

  Low   ████████████████████████████  0.8412
  High  ████                          0.1588
```

---

### 4 · Visualize

```bash
python main.py visualize --statistics     # class distribution
python main.py visualize --trajectory 1   # specific user's trajectory
python main.py visualize --all            # everything
```

### 5 · Extract Features

```bash
python main.py extract --svc_path file.svc   # single file
python main.py extract --dataset              # full dataset
```

### 6 · Python API

```python
from handwriting_emotion_detection import (
    load_all_svc_files, load_labels, get_dataloaders,
    EmotionCNN, extract_signal_features, plot_trajectory
)

samples  = load_all_svc_files(DATASET_ROOT)
labels   = load_labels(DASS_SCORES_PATH, target_emotion="anxiety")
train_loader, val_loader, test_loader = get_dataloaders(samples, labels)

model    = EmotionCNN(num_classes=2)
features = extract_signal_features(samples[0]["data"])
```

---

## 🏗️ Model Architectures

### EmotionCNN — fast & lean
```
Input (224×224)
    │
    ▼
Conv Block ×4  (BatchNorm + ReLU + MaxPool + Dropout)
    │
    ▼
Fully Connected  →  Softmax  →  [Low, High]

Parameters : ~2.7M
Best for   : small datasets, fast iteration
```

### EmotionResNet — powerful & accurate
```
Input (224×224)
    │
    ▼
ResNet18 backbone  (pretrained on ImageNet)
    │
    ▼
Fine-tuned FC head  →  Softmax  →  [Low, High]

Parameters : ~11.2M
Best for   : maximum accuracy, transfer learning
```

**Training setup** — Adam optimizer · ReduceLROnPlateau scheduler · Cross-Entropy with class weights · Early stopping · 70 / 15 / 15 train/val/test split

---

## ⚙️ Configuration

All tunable params live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `TARGET_EMOTION` | `"anxiety"` | `depression` · `anxiety` · `stress` |
| `USE_BINARY` | `True` | `True` = Low/High · `False` = 5-class |
| `IMAGE_SIZE` | `224` | Trajectory render resolution (px) |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `30` | Max training epochs |
| `LEARNING_RATE` | `1e-4` | Initial Adam LR |
| `EARLY_STOPPING_PATIENCE` | `7` | Patience before stopping |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `No trained model found` | Run `python main.py train --model cnn --emotion anxiety` first |
| `Dataset not found` | Check `config.py` paths, ensure `DASS_scores.xls` exists |
| `streamlit: not found` | `pip install streamlit` |
| Out of GPU memory | Lower `BATCH_SIZE` in `config.py` or switch to CPU |
| Poor accuracy | More epochs · lower LR · switch to `--model resnet` |

---

## 🤝 Contributing

```bash
git checkout -b feature/your-idea
# make changes, write tests, update docs
git commit -m "feat: your idea"
git push origin feature/your-idea
# open a Pull Request
```

Please follow PEP 8 and add docstrings to any new functions.

---

## 📚 References

- [EMOTHAW Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/520/emothaw)
- [DASS-21 Questionnaire](https://www.psych.unsw.edu.au/dass)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [kshitiz-arc](https://github.com/kshitiz-arc)**

*The pen is mightier than the word — and now, smarter too.*

</div>
