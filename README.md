# ML_WristRehab

**Real-time wrist rehab feedback using machine learning**

---

## Overview

**ML_WristRehab** is a Python-based system that analyzes wrist flexion and extension rehabilitation exercises and provides **real-time feedback** on patient form. Using machine learning, the system detects deviations from correct movement patterns, guiding patients toward safer and more effective recovery.

* Achieves **90.62% classification accuracy**
* Processes feedback with an **average response time of 0.01 seconds**
* Uses a webcam interface to provide instantaneous guidance

---

## Features

* Real-time wrist exercise evaluation through camera feed
* Automatic detection of improper form
* Feedback on movement quality and correction suggestions
* Supports flexion and extension exercises
* Pre-recorded datasets for training and testing

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
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* Python 3.10
* OpenCV (`cv2`)
* Mediapipe (`mediapipe`)
* TensorFlow/Keras (`tensorflow`)
* NumPy (`numpy`)
* Matplotlib, Seaborn, Pandas
* scikit-learn (`sklearn`)

---

## Usage

### 1. Data Collection

Pre-record exercise sequences using the `data_collection.py` script. This generates labeled data for training the ML model.

```bash
python data_collection.py
```

*(Replace with screenshot of data collection window)*
![Data Collection Placeholder](path/to/data_collection_image.png)

---

### 2. Training the Model

Train the ML model using the processed dataset:

```bash
python train_model.py
```

* The model uses LSTM and Conv1D layers for sequence learning.
* Early stopping and checkpointing are included to optimize training.

*(Add a plot showing training loss/accuracy if desired)*
![Training Placeholder](path/to/training_plot.png)

---

### 3. Real-Time Feedback

Run the real-time feedback system:

```bash
python real_time_feedback.py
```

* Uses webcam input to detect wrist landmarks with Mediapipe
* Provides corrective feedback on form through the camera window

*(Add GIF showing live feedback in action)*
![Real-Time Feedback Placeholder](path/to/feedback_demo.gif)

---

## Results / Metrics

* **Classification Accuracy:** 90.62%
* **Average Response Time:** 0.01 seconds per frame

*(Add confusion matrix or other visualizations here)*
![Confusion Matrix](https://github.com/EdwardDK/ML_WristRehab/blob/main/confusion_matrix.png?raw=true)

---

## Future Work

* Extend support to additional wrist or hand exercises
* Improve feedback clarity and visualization
* Explore integration with mobile devices or AR for at-home rehab

---

## License

No license is applied. For educational and portfolio purposes only.

---
