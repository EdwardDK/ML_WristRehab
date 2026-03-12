# ML_WristRehab

**Real-time wrist rehab feedback using machine learning**

---

## Overview

**ML_WristRehab** is a Python-based system that analyzes wrist flexion and extension rehabilitation exercises and provides **real-time feedback** on patient form. The system uses machine learning to detect deviations from correct movement patterns, helping patients recover safely and effectively.

* **Accuracy:** 90.62% on validation data
* **Latency:** ~0.01s per frame for model inference
* **Input:** Webcam-based hand tracking
* **Output:** Real-time corrective feedback via overlay on video

This project won **1st Place in the Medical & Health Sciences category at the Connecticut STEM Competition**.

---

## Features

* Collect wrist exercise data with a webcam
* Train a LSTM + Conv1D ML model for movement classification
* Provide **real-time feedback** to guide form correction
* Visual metrics including confusion matrix and latency charts

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ML_WristRehab.git
cd ML_WristRehab
```

2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

* Python 3.10
* OpenCV (`cv2`)
* Mediapipe (`mediapipe`)
* TensorFlow/Keras (`tensorflow`)
* NumPy, Matplotlib, Seaborn, Pandas
* scikit-learn (`sklearn`)

---

## Usage

### 1. Data Collection

Use the `data_collection.py` script to record wrist movement sequences. Enter a label for each recording (`bad`, `okay`, `perfect`) and press `r` to start capturing a sequence of frames.

```bash
python data_collection.py
```

*(Add image showing the data collection window)*
![Data Collection Placeholder](path/to/data_collection_image.png)

---

### 2. Training the Model

Use `train_model.py` to train the machine learning model:

```bash
python train_model.py
```

**Model Architecture:**

* Input → Conv1D → BatchNormalization → LSTM → Dropout → LSTM → Dense → Dropout → Dense (Softmax)
* Uses data augmentation (adding Gaussian noise) for robustness
* EarlyStopping and ModelCheckpoint to prevent overfitting

*(Add plot showing training accuracy/loss over epochs)*
![Training Plot Placeholder](path/to/training_plot.png)

---

### 3. Real-Time Feedback

Run the system to receive live feedback on wrist exercises:

```bash
python real_time_feedback.py
```

* Webcam input with MediaPipe hand tracking
* Smooths landmark positions to reduce jitter
* Displays feedback messages and color-coded cues over the video

*(Add GIF demonstrating real-time feedback)*
![Feedback Demo Placeholder](path/to/feedback_demo.gif)

---

### 4. Model Evaluation

Use `evaluate_model.py` to generate metrics and visualizations:

* Confusion Matrix
* Precision, Recall, F1-Score heatmaps
* Latency analysis for MediaPipe extraction, model inference, and feedback calculation

```bash
python evaluate_model.py
```

*(Add images of confusion matrix, classification metrics, and latency chart)*
<img src="https://github.com/EdwardDK/ML_WristRehab/blob/main/confusion_matrix.png?raw=true" width="500" height="250">
![Classification Metrics Placeholder](path/to/classification_metrics.png)
![Latency Chart Placeholder](path/to/latency_chart.png)

---

## Results

* **Classification Accuracy:** 90.62%
* **Real-Time Latency:** ~0.01 seconds per inference
* **Majority Vote Smoothing:** Stabilizes predictions for more reliable feedback

---

## Project Structure

```
ML_WristRehab/
├─ data_collection.py      # Script to collect labeled exercise sequences
├─ train_model.py          # Script to train LSTM + Conv1D model
├─ real_time_feedback.py   # Script to run real-time feedback
├─ evaluate_model.py       # Script to analyze model performance
├─ utils.py                # Helper functions (landmark processing, feedback)
├─ dataset/                # Folder containing recorded sequences
├─ model/                  # Saved ML model (wrist_model.h5)
├─ README.md               # Project documentation
└─ requirements.txt        # Python dependencies
```

---

## Future Work

* Extend to other wrist or hand exercises
* Improve feedback visuals for patients
* Explore mobile or AR integration for at-home rehab

---

## License

No license applied. For educational and portfolio purposes only.

---
