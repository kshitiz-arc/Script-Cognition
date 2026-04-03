"""
Streamlit Web Interface for Handwriting Emotion Detection.

Provides an interactive web app for:
- Uploading .svc handwriting files
- Testing emotion detection
- Visualizing results
- Exploring the dataset
"""
import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATASET_ROOT, DASS_SCORES_PATH, MODEL_DIR, TARGET_EMOTION,
    USE_BINARY, CLASS_NAMES, IMAGE_SIZE
)
from preprocessing.svc_parser import parse_svc, load_all_svc_files
from data.label_loader import load_labels
from features.image_generator import trajectory_to_image
from features.signal_features import extract_signal_features
from models.cnn_model import get_model
from inference.predict import predict, load_trained_model
from utils.visualization import plot_trajectory, plot_dataset_statistics


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Handwriting Emotion Detection",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        font-size: 1rem;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .emotion-low {
        background-color: #d4edda;
        color: #155724;
    }
    .emotion-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE & CACHING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
# cache keyed by emotion so different models are stored separately
def load_model_cached(emotion):
    """Load trained model for a specific emotion and cache it."""
    try:
        model, info = load_trained_model(target_emotion=emotion)
        return model, info, None
    except FileNotFoundError as e:
        return None, None, str(e)


@st.cache_resource
# emotion is passed through to ensure cache is keyed by the selected target
# (Streamlit will automatically memoize based on the function argument).
def load_dataset_cached(emotion):
    """Load dataset once and cache it for a given target emotion."""
    try:
        samples = load_all_svc_files(DATASET_ROOT)
        labels = load_labels(DASS_SCORES_PATH, emotion, USE_BINARY)
        return samples, labels, None
    except Exception as e:
        return None, None, str(e)


def initialize_session():
    """Initialize session state variables."""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "svc_data" not in st.session_state:
        st.session_state.svc_data = None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit app."""
    initialize_session()

    # emotion selector in sidebar
    EMOTIONS = ["depression", "anxiety", "stress"]
    default_index = EMOTIONS.index(TARGET_EMOTION) if TARGET_EMOTION in EMOTIONS else 0
    selected_emotion = st.sidebar.selectbox("🎯 Target Emotion", EMOTIONS, index=default_index)
    st.session_state.selected_emotion = selected_emotion

    # Header
    st.title("✍️ Handwriting Emotion Detection")
    st.markdown("### Detect emotional states from handwriting patterns using AI")

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📌 Pages")
        page = st.radio(
            "Select a page:",
            ["🏠 Home", "📊 Dataset Explorer", "🧪 Test Handwriting", "ℹ️ Information"]
        )

    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Dataset Explorer":
        show_dataset_explorer()
    elif page == "🧪 Test Handwriting":
        show_test_handwriting()
    elif page == "ℹ️ Information":
        show_information()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════

def show_home():
    """Home page with overview and quick info."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## Welcome! 👋

        This application detects **emotional states** from **handwriting** using a trained CNN model.

        ### How It Works
        1. **Upload** a handwriting file (.svc format)
        2. **Click** the "Analyze" button
        3. **Get** emotion classification results

        ### What Can It Detect?
        The model classifies handwriting based on the **DASS-21** scale:
        - **Depression** (or Anxiety/Stress)
        - **Severity Level**: Low vs High

        ### Key Features
        - Real-time emotion prediction
        - Confidence scores
        - Probability distribution
        - Trajectory visualization
        - Feature extraction
        """)

    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. Go to **Test Handwriting** page
        2. Upload a `.svc` file
        3. Click **Analyze Handwriting**
        4. View results

        ### 📚 Learn More
        - Visit **Dataset Explorer** to understand the data
        - Check **Information** page for technical details
        """)

    # Show model status
    st.markdown("---")
    st.markdown("### 📊 Model Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
        model, info, error = load_model_cached(emotion)
        if model is not None:
            st.success("✅ Model Loaded")
            st.caption(f"Type: {info['model_type'].upper()}")
        else:
            st.error("❌ Model Not Found")
            st.caption("Please train a model first")

    with col2:
        emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
        samples, labels, error = load_dataset_cached(emotion)
        if samples is not None:
            st.success("✅ Dataset Found")
            st.caption(f"{len(samples)} samples loaded")
        else:
            st.error("❌ Dataset Not Found")

    with col3:
        st.info("ℹ️ Target Emotion")
        emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
        st.caption(f"{emotion.capitalize()}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def show_dataset_explorer():
    """Dataset exploration and visualization."""
    st.markdown("## 📊 Dataset Explorer")

    emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
    samples, labels, error = load_dataset_cached(emotion)

    if error:
        st.error(f"Could not load dataset: {error}")
        return

    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", len(labels))

    with col2:
        st.metric("Total Samples", len(samples))

    with col3:
        unique_users = len(set(s["user_id"] for s in samples if s["user_id"] in labels))
        st.metric("Users w/ Data", unique_users)

    with col4:
        st.metric("Tasks/User", "Up to 7")

    # Class distribution
    st.markdown("---")
    st.markdown("### Class Distribution")

    label_counts = {}
    for label in labels.values():
        cls_name = CLASS_NAMES[label]
        label_counts[cls_name] = label_counts.get(cls_name, 0) + 1

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(label_counts.keys())
        values = list(label_counts.values())
        colors = ["#90EE90", "#FFB6C6"][:len(names)]
        ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.5)
        ax.set_ylabel("Number of Users", fontsize=12)
        emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
        ax.set_title(f"Class Distribution - {emotion.capitalize()}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
        st.pyplot(fig)

    with col2:
        st.markdown("### Statistics")
        for cls_name, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            st.write(f"**{cls_name}**: {count} users ({percentage:.1f}%)")

    # Sample handwriting trajectories
    st.markdown("---")
    st.markdown("### Sample Handwriting Trajectories")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input("Select User ID:", min_value=1, value=1, step=1)

    with col2:
        task_id = st.number_input("Select Task ID:", min_value=1, max_value=7, value=1, step=1)

    if st.button("📈 Display Trajectory"):
        user_samples = [s for s in samples if s["user_id"] == user_id]

        if not user_samples:
            st.warning(f"No samples found for user {user_id}")
        else:
            sample = user_samples[0]
            data = sample["data"]

            # Plot trajectory
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            x = data[:, 0]
            y = data[:, 1]
            pen_status = data[:, 3]
            pressure = data[:, 6]

            # Plot 1: Pen status
            pen_down = pen_status == 1
            axes[0].scatter(x[pen_down], -y[pen_down], c="navy", s=1, alpha=0.8, label="Pen Down")
            axes[0].set_title(f"User {user_id} - Task {task_id}\n(Pen Status)", fontweight="bold")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].set_aspect("equal")
            axes[0].grid(True, alpha=0.2)
            axes[0].legend()

            # Plot 2: Pressure
            valid = pen_down & (pressure > 0)
            scatter = axes[1].scatter(x[valid], -y[valid], c=pressure[valid], cmap="viridis", s=2, alpha=0.8)
            axes[1].set_title(f"User {user_id} - Task {task_id}\n(Pressure)", fontweight="bold")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].set_aspect("equal")
            axes[1].grid(True, alpha=0.2)
            plt.colorbar(scatter, ax=axes[1], label="Pressure")

            st.pyplot(fig)

            # Statistics
            st.markdown("#### Handwriting Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", len(data))
            with col2:
                st.metric("X Range", f"{x.max() - x.min():.0f}")
            with col3:
                st.metric("Y Range", f"{y.max() - y.min():.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TEST HANDWRITING
# ═══════════════════════════════════════════════════════════════════════════

def show_test_handwriting():
    """Main testing interface for emotion detection."""
    st.markdown("## 🧪 Test Handwriting Emotion Detection")

    emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
    model, model_info, model_error = load_model_cached(emotion)

    # Check if model is loaded
    if model is None:
        st.error("⚠️ No trained model found for the selected emotion!")
        st.info("""
        You need to train a model for the emotion you chose. Examples:
        ```bash
        # single emotion (e.g. anxiety)
        python main.py train --model cnn --emotion anxiety

        # train all three at once
        python main.py train_all --model cnn
        ```
        After training, refresh this page or restart the app.
        """)
        return

    # make sure user knows which emotion is currently active
    current_emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
    trained_emotion = model_info.get("target_emotion") if model_info else None
    if trained_emotion and trained_emotion != current_emotion:
        st.warning(
            f"⚠️ The loaded model was trained for '{trained_emotion}', "
            f"but you have selected '{current_emotion}'. "
            "Dataset labels will follow the selection, so predictions may not align."
        )

    # Upload file or select from dataset
    st.markdown("### 📁 Input Source")
    input_mode = st.radio("Choose input method:", ["Upload .svc File", "Select from Dataset"])

    svc_data = None
    filename = None

    if input_mode == "Upload .svc File":
        # File upload
        uploaded_file = st.file_uploader("Upload a .svc handwriting file", type=["svc"])

        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                os.makedirs("/tmp", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                svc_data = parse_svc(temp_path)
                filename = uploaded_file.name
                st.success(f"✅ File loaded: {filename} ({len(svc_data)} data points)")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    else:
        # Select from dataset
        emotion = st.session_state.get("selected_emotion", TARGET_EMOTION)
        samples, labels, dataset_error = load_dataset_cached(emotion)

        if dataset_error:
            st.error(f"Could not load dataset: {dataset_error}")
            return

        col1, col2 = st.columns(2)

        with col1:
            user_id = st.number_input("User ID:", min_value=1, value=1, step=1)

        with col2:
            task_id = st.number_input("Task ID:", min_value=1, max_value=7, value=1, step=1)

        # Find sample
        matching_samples = [
            s for s in samples
            if s["user_id"] == user_id and s["task_id"] == task_id and s["user_id"] in labels
        ]

        if matching_samples:
            svc_data = matching_samples[0]["data"]
            filename = f"user_{user_id:05d}_task_{task_id:05d}.svc"
            st.success(f"✅ Selected: {filename} ({len(svc_data)} data points)")
            # Show actual label for validation
            actual_label = labels[user_id]
            st.info(f"**Actual Label (from dataset)**: {CLASS_NAMES[actual_label]}")
        else:
            st.warning(f"No sample found for User {user_id}, Task {task_id}")

    # Processing buttons
    st.markdown("---")
    st.markdown("### 🔄 Processing")

    col1, col2, col3 = st.columns(3)

    with col1:
        analyze_button = st.button("🎯 Analyze Handwriting", key="analyze", use_container_width=True)

    with col2:
        visualize_button = st.button("📈 Visualize Trajectory", key="visualize", use_container_width=True)

    with col3:
        extract_button = st.button("🔍 Extract Features", key="extract", use_container_width=True)

    # ─── Analyze Handwriting ───────────────────────────────────────────
    if analyze_button and svc_data is not None:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        try:
            # Create temporary file for prediction
            temp_path = "/tmp/temp_handwriting.svc"
            os.makedirs("/tmp", exist_ok=True)

            # Write .svc format
            with open(temp_path, "w") as f:
                f.write(f"{len(svc_data)}\n")
                for row in svc_data:
                    f.write(" ".join(f"{v:.1f}" for v in row) + "\n")

            # Make prediction
            result = predict(temp_path, model=model, model_info=model_info)

            if "error" in result:
                st.error(f"Prediction error: {result['error']}")
            else:
                # Main result
                pred_class = result["predicted_class"]
                confidence = result["confidence"]
                probs = result["all_probabilities"]

                # Color based on prediction
                emotion_color = "emotion-high" if pred_class == "High" else "emotion-low"

                st.markdown(f"""
                <div class="emotion-box {emotion_color}">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">
                        Predicted Emotion: <strong>{pred_class.upper()}</strong>
                    </div>
                    <div style="font-size: 1.2rem;">
                        Confidence: <strong>{confidence:.2%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Data Points", result["num_data_points"])

                with col2:
                    # display the emotion associated with the loaded model
                    st.metric("Target Emotion", result["target_emotion"].capitalize())

                with col3:
                    st.metric("Model Type", result.get("model_type", "Unknown"))

                # Probability distribution
                st.markdown("#### Probability Distribution")

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    classes = list(probs.keys())
                    probabilities = list(probs.values())
                    colors = ["#90EE90" if c == "Low" else "#FFB6C6" for c in classes]
                    bars = ax.barh(classes, probabilities, color=colors, edgecolor="black", linewidth=1.5)
                    ax.set_xlabel("Probability", fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.grid(True, alpha=0.3, axis="x")

                    # Add percentage labels
                    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                        ax.text(prob + 0.02, i, f"{prob:.2%}", va="center", fontweight="bold")

                    st.pyplot(fig)

                with col2:
                    for cls_name, prob in probs.items():
                        st.write(f"**{cls_name}**: {prob:.4f}")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
            import traceback
            st.write(traceback.format_exc())

    # ─── Visualize Trajectory ──────────────────────────────────────────
    if visualize_button and svc_data is not None:
        st.markdown("---")
        st.markdown("### 📈 Handwriting Trajectory Visualization")

        try:
            data = svc_data
            x = data[:, 0]
            y = data[:, 1]
            pen_status = data[:, 3]
            pressure = data[:, 6]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot 1: Pen status
            pen_down = pen_status == 1
            axes[0].scatter(x[pen_down], -y[pen_down], c="navy", s=1, alpha=0.8, label="Pen Down")
            axes[0].set_title("Handwriting Trajectory\n(Pen Status)", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].set_aspect("equal")
            axes[0].grid(True, alpha=0.2)
            axes[0].legend()

            # Plot 2: Pressure heatmap
            valid = pen_down & (pressure > 0)
            scatter = axes[1].scatter(x[valid], -y[valid], c=pressure[valid], cmap="viridis", s=2, alpha=0.8)
            axes[1].set_title("Handwriting Trajectory\n(Pressure Heatmap)", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].set_aspect("equal")
            axes[1].grid(True, alpha=0.2)
            cbar = plt.colorbar(scatter, ax=axes[1], label="Pressure")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error visualizing trajectory: {e}")

    # ─── Extract Features ──────────────────────────────────────────────
    if extract_button and svc_data is not None:
        st.markdown("---")
        st.markdown("### 🔍 Extracted Handwriting Features")

        try:
            features = extract_signal_features(svc_data)

            # Display features in columns
            feature_dict = {k: v for k, v in features.items()}

            # Create tabs for different feature categories
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Velocity", "Pressure", "Trajectory", "Other"]
            )

            # Velocity features
            with tab1:
                vel_features = {
                    k: v for k, v in feature_dict.items()
                    if "velocity" in k.lower() or "speed" in k.lower()
                }
                if vel_features:
                    for name, value in vel_features.items():
                        st.metric(name.replace("_", " ").title(), f"{value:.4f}")
                else:
                    st.write("No velocity features extracted")

            # Pressure features
            with tab2:
                press_features = {
                    k: v for k, v in feature_dict.items()
                    if "pressure" in k.lower()
                }
                if press_features:
                    for name, value in press_features.items():
                        st.metric(name.replace("_", " ").title(), f"{value:.4f}")
                else:
                    st.write("No pressure features extracted")

            # Trajectory features
            with tab3:
                traj_features = {
                    k: v for k, v in feature_dict.items()
                    if any(word in k.lower() for word in ["bounding", "length", "coverage", "curvature"])
                }
                if traj_features:
                    for name, value in traj_features.items():
                        st.metric(name.replace("_", " ").title(), f"{value:.4f}")
                else:
                    st.write("No trajectory features extracted")

            # Other features
            with tab4:
                other_features = {
                    k: v for k, v in feature_dict.items()
                    if k not in vel_features and k not in press_features and k not in traj_features
                }
                if other_features:
                    for name, value in other_features.items():
                        st.metric(name.replace("_", " ").title(), f"{value:.4f}")

            # Download features as CSV
            import pandas as pd
            df = pd.DataFrame([features])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Features as CSV",
                data=csv,
                file_name="handwriting_features.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error extracting features: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: INFORMATION
# ═══════════════════════════════════════════════════════════════════════════

def show_information():
    """Information and help page."""
    st.markdown("## ℹ️ Information & Help")

    with st.expander("📚 About This Project", expanded=True):
        st.markdown("""
        This is a **machine learning project** for detecting emotional states from handwriting.

        ### EMOTHAW Dataset
        - **Source**: Online handwriting captured from digital tablets
        - **Subjects**: 129 participants
        - **Tasks**: 7 handwriting tasks per person
        - **Labels**: DASS-21 psychological scales (Depression, Anxiety, Stress)

        ### Model Architecture
        - **Type**: Convolutional Neural Network (CNN)
        - **Input**: 224×224 grayscale images of pen trajectories
        - **Output**: Emotion classification (Low/High severity)
        - **Training**: ~30 epochs with early stopping

        ### How It Works
        1. Parse .svc handwriting file
        2. Render pen trajectory as grayscale image
        3. Feed to trained CNN
        4. Output emotion prediction with confidence
        """)

    with st.expander("🔧 How to Use"):
        st.markdown("""
        ### Step-by-Step Guide

        1. **Navigate to "Test Handwriting" page**
        2. **Choose input**:
           - Upload a `.svc` file, OR
           - Select from the dataset
        3. **Click a button**:
           - **Analyze Handwriting** → Get emotion prediction
           - **Visualize Trajectory** → See pen movement
           - **Extract Features** → Get 29+ handwriting features
        4. **View results** and download if needed

        ### What is a .svc File?
        A text file containing pen trajectory data:
        - X, Y coordinates
        - Timestamp
        - Pen pressure
        - Pen status (up/down)
        - Azimuth & altitude angles

        Example format:
        ```
        1474
        150.5 200.3 1000 1 45.2 30.5 125.0
        151.2 201.1 1010 1 45.5 30.6 128.0
        ...
        ```
        """)

    with st.expander("📊 Understanding Results"):
        st.markdown("""
        ### Emotion Classification
        The model classifies handwriting into two categories:
        - **Low**: Normal to Mild severity
        - **High**: Moderate to Extremely Severe

        Based on DASS-21 thresholds for the selected emotion.

        ### Confidence Score
        - Range: 0.0 to 1.0 (0% to 100%)
        - Higher = More confident
        - >0.85 = Very confident
        - 0.5-0.85 = Moderate confidence
        - <0.5 = Low confidence

        ### Probability Distribution
        Shows model's % probability for EACH class:
        - Should sum to 100%
        - Wider gap between classes = More confident
        """)

    with st.expander("🎯 Extracted Features (29 total)"):
        st.markdown("""
        ### Categories

        **Velocity Features** (4)
        - Mean, max, std, median velocity

        **Acceleration Features** (3)
        - Mean, max, std acceleration

        **Pressure Features** (4)
        - Mean, max, std, variance, entropy

        **Curvature Features** (3)
        - Mean, max, std curvature

        **Stroke Features** (2)
        - Total length, mean segment length

        **Spatial Features** (5)
        - Bounding width, height, area, aspect ratio

        **Pen Behavior** (3)
        - Pen lifts, mean pause, max pause, total pause

        **Efficiency** (1)
        - Writing ratio

        These features encode handwriting dynamics that correlate with emotional state.
        """)

    with st.expander("❓ FAQ"):
        st.markdown("""
        **Q: What format should the file be?**
        A: Must be `.svc` format (text file with space-separated values)

        **Q: Can I use my own handwriting?**
        A: Yes! Write on a digital tablet and export as .svc format

        **Q: How accurate is the model?**
        A: ~70-74% accuracy on test set (depends on emotion & training duration)

        **Q: What emotions can it detect?**
        A: Depression, Anxiety, or Stress (configured in settings)

        **Q: Can I retrain the model?**
        A: Yes, use: `python main.py train --model cnn`

        **Q: Is GPU required?**
        A: No, but it's faster with GPU. Model runs on CPU too.
        """)

    with st.expander("🔗 Project Links"):
        st.markdown("""
        **Project Files:**
        - `main.py` - CLI interface
        - `config.py` - Configuration
        - `README.md` - Documentation
        - `test_project.py` - Test suite

        **Command Line Usage:**
        ```bash
        python main.py info                 # View project info
        python main.py train --model cnn    # Train model
        python main.py predict --svc_path <file>    # Predict
        python main.py visualize --all      # Generate plots
        ```
        """)


# ═══════════════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
