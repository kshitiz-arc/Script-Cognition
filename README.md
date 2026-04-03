# Handwriting Emotion Detection — EMOTHAW

A comprehensive Python machine learning pipeline that detects **emotional states** (Depression, Anxiety, Stress) from **handwriting signals** using the **EMOTHAW** (EMOTional HAndWriting) dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-orange.svg)](https://streamlit.io/)

## 🎯 Overview

This project analyzes online handwriting data captured from digital tablets and classifies the writer's emotional state using a **CNN-based image classification** approach. Pen trajectory coordinates are rendered into grayscale images, then classified using a Convolutional Neural Network.

### How It Works

```
.svc Handwriting File → Parse Pen Data → Render Trajectory Image → CNN → Emotion Prediction
```

### Emotional States (DASS-21)

Labels are derived from the **DASS (Depression Anxiety Stress Scales)** questionnaire:
- **Depression** — Normal / Mild / Moderate / Severe / Extremely Severe
- **Anxiety** — Normal / Mild / Moderate / Severe / Extremely Severe
- **Stress** — Normal / Mild / Moderate / Severe / Extremely Severe

Binary mode (default): **Low** vs **High** severity.

## ✨ Features

- 🧠 **CNN Models**: Lightweight EmotionCNN and ResNet-based architectures
- 🌐 **Interactive Web App**: Streamlit interface with emotion dropdown
- 📊 **Multi-Emotion Support**: Train and predict for Depression, Anxiety, or Stress
- 📈 **Visualization**: Trajectory plots, feature extraction, and dataset statistics
- 🔧 **CLI Tools**: Comprehensive command-line interface for training, prediction, and analysis
- 📚 **Dataset Explorer**: Built-in tools to understand the EMOTHAW dataset
- 🎯 **Real-time Prediction**: Upload .svc files and get instant emotion classification
- 📥 **Feature Extraction**: 29+ handwriting signal features (velocity, pressure, curvature, etc.)

## 📁 Project Structure

```
handwriting_emotion_detection/
├── config.py                      # Central configuration
├── main.py                        # Unified CLI entry point
├── app.py                         # Streamlit web interface
├── run_app.py                     # Helper script to launch web app
├── requirements.txt               # Python dependencies
├── README.md                      # This documentation
│
├── preprocessing/
│   └── svc_parser.py             # Parse .svc files → NumPy arrays
├── data/
│   ├── label_loader.py           # Load DASS scores → labels
│   └── dataset.py                # PyTorch Dataset & DataLoader
├── features/
│   ├── image_generator.py        # Trajectories → images
│   └── signal_features.py        # Handwriting signal features
├── models/
│   └── cnn_model.py              # EmotionCNN + EmotionResNet
├── training/
│   └── train_cnn.py              # Training loop with early stopping
├── evaluation/
│   └── evaluate.py               # Metrics, confusion matrix, plotting
├── inference/
│   └── predict.py                # Inference on new samples
├── utils/
│   └── visualization.py          # Trajectory & statistics plotting
└── outputs/
    ├── models/                   # Trained model checkpoints
    ├── plots/                    # Generated visualizations
    └── image_cache/              # Cached trajectory images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- The EMOTHAW dataset (expected at `../archive/DataEmothaw/`)
- CUDA (optional, for GPU acceleration)

### Installation

1. **Navigate to project directory**:
```bash
cd handwriting_emotion_detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify dataset**:
The project expects:
- Dataset at: `../archive/DataEmothaw/`
- DASS scores file: `../archive/DataEmothaw/DASS_scores.xls`
- User folders: `user00001/`, `user00002/`, etc.
- Handwriting files: `.svc` format

## 🎮 Usage

### 1. Train Models

**Train for a single emotion**:
```bash
# Default: anxiety
python main.py train --model cnn

# Specify emotion
python main.py train --model cnn --emotion depression
python main.py train --model cnn --emotion stress

# Custom parameters
python main.py train --model cnn --emotion anxiety --epochs 50 --batch_size 32 --lr 0.0001
```

**Train all emotions at once**:
```bash
python main.py train_all --model cnn
```

### 2. Launch Web Interface

**Start the interactive web app**:
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

**Web App Features**:
- 🎯 **Emotion Dropdown**: Switch between Depression, Anxiety, and Stress
- 📁 **Upload .svc Files**: Analyze your own handwriting samples
- 📊 **Dataset Selection**: Pick samples from the EMOTHAW dataset
- 📈 **Real-time Analysis**: Get emotion predictions with confidence scores
- 📈 **Trajectory Visualization**: See pen movement patterns
- 🔍 **Feature Extraction**: View 29+ handwriting features
- 📚 **Dataset Explorer**: Understand the data distribution and statistics

### 3. Make Predictions (CLI)

```bash
# Predict from file
python main.py predict --svc_path "path/to/handwriting.svc"

# Specify emotion (if model supports it)
python main.py predict --svc_path "file.svc" --emotion stress
```

**Expected output**:
```
══════════════════════════════════════════════════════
  HANDWRITING EMOTION PREDICTION
══════════════════════════════════════════════════════

Input file: u00001s00001_hw00001.svc
Data points: 1474
Target dimension: Anxiety

──────────────────────────────────────────────────────
  PREDICTED EMOTION: LOW
  CONFIDENCE: 0.8412 (84.12%)
──────────────────────────────────────────────────────

Probability Distribution:
  Low                 : 0.8412  ████████████████████████████
  High                : 0.1588  ████
```

### 4. Generate Visualizations

```bash
# Dataset class distribution
python main.py visualize --statistics

# Specific user's trajectory
python main.py visualize --trajectory 1

# All visualizations
python main.py visualize --all
```

### 5. Extract Handwriting Features

```bash
# Signal features from a single file
python main.py extract --svc_path file.svc

# Extract from entire dataset
python main.py extract --dataset
```

### 6. View Project Information

```bash
python main.py info
```

## 📊 Understanding Results

### Emotion Classification
The model classifies handwriting into two categories:
- **Low**: Normal to Mild severity
- **High**: Moderate to Extremely Severe

Based on DASS-21 thresholds for the selected emotion.

### Model Architectures
- **EmotionCNN**: Lightweight custom CNN (~2.7M parameters)
- **EmotionResNet**: ResNet18 with transfer learning (~11.2M parameters)

### Handwriting Features
The system extracts 29+ features including:
- Velocity metrics (mean, std, max)
- Pressure statistics
- Trajectory properties (bounding box, length, coverage)
- Pen status analysis (lift frequency, stroke count)
- Curvature and smoothness measures

## 🔧 Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_EMOTION` | `"anxiety"` | DASS dimension: depression, anxiety, stress |
| `USE_BINARY` | `True` | Binary (Low/High) vs. 5-class severity |
| `IMAGE_SIZE` | `224` | Rendered trajectory image size |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `30` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `EARLY_STOPPING_PATIENCE` | `7` | Epochs before early stop |

## 📈 Training Details

### Dataset Split
- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

### Hyperparameters
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Loss**: Cross-Entropy with class weights
- **Early Stopping**: Based on validation loss

### Model Checkpoints
Models are saved in `outputs/models/` with emotion-specific names:
- `best_cnn_anxiety.pth`
- `best_cnn_depression.pth`
- `best_cnn_stress.pth`

## 🌐 Deployment

### Local Development
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place EMOTHAW dataset in `../archive/DataEmothaw/`
4. Train models: `python main.py train_all --model cnn`
5. Launch web app: `python run_app.py`

### GitHub Repository
The project is designed to be fully portable:
- Source code and configuration are tracked
- Generated models and large datasets are ignored
- Anyone can clone, install deps, and run locally

### Cloud Deployment
The Streamlit app can be deployed to:
- **Streamlit Cloud**: Connect GitHub repo, auto-deploy
- **Heroku**: Use `streamlit` buildpack
- **AWS/Heroku**: Containerize with Docker

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README for new features
- Test on multiple Python versions

## 📄 License

This project is open-source. Please check the LICENSE file for details.

## 📚 References

- **EMOTHAW Dataset**: Online handwriting emotional database
- **DASS-21**: Depression Anxiety Stress Scales questionnaire
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework for ML

## 🆘 Troubleshooting

### Common Issues

**"No trained model found"**
- Train a model first: `python main.py train --model cnn --emotion anxiety`

**"Dataset not found"**
- Ensure EMOTHAW data is at `../archive/DataEmothaw/`
- Check file paths in `config.py`

**"Streamlit not found"**
- Install: `pip install streamlit`

**Low GPU memory**
- Reduce batch size in `config.py` or use CPU

### Getting Help

- Check the [Issues](https://github.com/yourusername/handwriting_emotion_detection/issues) page
- Review the code comments and docstrings
- Run `python main.py info` for project details

---

**Happy handwriting analysis! ✍️**

### 2. Launch Web Interface ⭐ NEW

**Start the interactive web app:**
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

**Features:**
- 📁 Upload .svc files or select from dataset
- 🎯 Click buttons to analyze handwriting
- 📊 View emotion predictions with confidence scores
- 📈 Visualize pen trajectories
- 🔍 Extract 29+ handwriting features
- 📚 Explore dataset and learn more

### 3. Make Predictions (CLI)

```bash
python main.py predict --svc_path "path/to/handwriting.svc"
```

**Expected output**:
```
══════════════════════════════════════════════════════
  HANDWRITING EMOTION PREDICTION
══════════════════════════════════════════════════════

Input file: u00001s00001_hw00001.svc
Data points: 1474
Target dimension: Anxiety

──────────────────────────────────────────────────────
  PREDICTED EMOTION: LOW
  CONFIDENCE: 0.8412 (84.12%)
──────────────────────────────────────────────────────

Probability Distribution:
  Low                 : 0.8412  ████████████████████████████
  High                : 0.1588  ████
```

### 4. Generate Visualizations

```bash
# Dataset class distribution
python main.py visualize --statistics

# Specific user's trajectory
python main.py visualize --trajectory 1

# All visualizations
python main.py visualize --all
```

### 4. Extract Handwriting Features

```bash
# Signal features from a single file
python main.py extract --svc_path file.svc

# Extract from entire dataset
python main.py extract --dataset
```

### 5. View Project Information

```bash
python main.py info
```

### 6. Train Models for All Emotions

The web interface includes a **Target Emotion** dropdown; you must have a trained
model for each emotion you want to try. There are two convenient ways to build
them:

```bash
# train a model for a single dimension (override config)
python main.py train --model cnn --emotion anxiety
python main.py train --model cnn --emotion depression
python main.py train --model cnn --emotion stress

# or train all three in one shot – files will be saved with emotion tags
python main.py train_all --model cnn
```

Each checkpoint is stored in `outputs/models/` with names like
`best_cnn_anxiety.pth`, so the Streamlit app automatically loads the correct
file when you change the dropdown.

Once the models exist you can launch the app and switch the target emotion in
the sidebar to see results for depression, anxiety or stress without editing
any code.

> **Note:** if you trained a model prior to adding emotion-specific filenames,
> the app will still load it thanks to a fallback mechanism.  The generic
> checkpoints (`best_cnn_model.pth`, etc.) will be used for whatever emotion is
> currently selected.  However it's cleaner to retrain or rename those files to
> include the emotion (e.g. copy `best_cnn_model.pth` → `best_cnn_depression.pth`).


## Detailed Usage

### Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET_EMOTION` | `"anxiety"` | DASS dimension: depression, anxiety, stress |
| `USE_BINARY` | `True` | Binary (Low/High) vs. 5-class severity |
| `IMAGE_SIZE` | `224` | Rendered trajectory image size |
| `BATCH_SIZE` | `16` | Training batch size |
| `NUM_EPOCHS` | `30` | Maximum training epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `EARLY_STOPPING_PATIENCE` | `7` | Epochs before early stop |

### Python API

Use as a Python package:

```python
from handwriting_emotion_detection import (
    load_all_svc_files,
    load_labels,
    get_dataloaders,
    EmotionCNN,
    extract_signal_features,
    plot_trajectory
)

# Load data
samples = load_all_svc_files(DATASET_ROOT)
labels = load_labels(DASS_SCORES_PATH, target_emotion="anxiety")
train_loader, val_loader, test_loader = get_dataloaders(samples, labels)

# Create model
model = EmotionCNN(num_classes=2)

# Extract features
features = extract_signal_features(samples[0]["data"])
```

## Model Architectures

### EmotionCNN (Lightweight - Recommended)
- 4 convolutional blocks with batch normalization
- 2.7 million parameters
- Dropout regularization
- Best for small datasets

### EmotionResNet (Powerful - Transfer Learning)
- ResNet18 backbone pretrained on ImageNet
- Fine-tuned on EMOTHAW data
- 11.2 million parameters
- Better feature extraction

## Dataset Information

**EMOTHAW** contains:
- **129 subjects** across 2 data collections
- **7 handwriting tasks** per subject
- **~900+ samples** with emotional labels
- **.svc format**: X, Y, timestamp, pen pressure, pen status, azimuth, altitude

## Technologies

- **PyTorch** — Deep learning framework
- **OpenCV** — Image generation from trajectories
- **scikit-learn** — Data splitting, evaluation metrics
- **NumPy** — Numerical computation
- **Matplotlib** — Visualization
- **xlrd** — Excel file parsing (DASS scores)

## Troubleshooting

### Q: Slow training
**A:** Enable image caching (automatic), use GPU, reduce batch size

### Q: No trained model found
**A:** Train first: `python main.py train`

### Q: Poor accuracy  
**A:** Try more epochs, adjust learning rate, use ResNet model

### Q: Dataset loading fails
**A:** Verify paths in `config.py`, check DASS_scores.xls exists

## License

This project is for research and educational purposes.

---

**Last Updated**: March 2025  
**Version**: 1.0.0  
**Python**: 3.10+  
**PyTorch**: 2.0+
