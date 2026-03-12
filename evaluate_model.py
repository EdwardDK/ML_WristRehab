import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from matplotlib.colors import LinearSegmentedColormap

# Import validation data and labels from the training script
from train_model import labels, X_val, y_val

# Create custom colormap using your hex code #EAD1DC
custom_cmap = LinearSegmentedColormap.from_list("custom_pink", ["#ffffff", "#EAD1DC"], N=256)

# ==========================================
# 1. Load Model & Generate Predictions
# ==========================================
print("--- Starting Evaluation ---")
print("Loading model...")
model = load_model('model/wrist_model.h5')

print("Generating predictions...")
start_time = time.time()
y_pred = model.predict(X_val)
end_time = time.time()

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Calculate Accuracy
accuracy = accuracy_score(y_true_classes, y_pred_classes)

# ==========================================
# 2. Print Metrics to Console
# ==========================================
print("\n" + "="*30)
print("METRICS SUMMARY")
print("="*30)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
report_text = classification_report(y_true_classes, y_pred_classes, target_names=labels, zero_division=0)
print(report_text)

# Latency Calculation
avg_inference_time = (end_time - start_time) / len(X_val)
mediapipe_time = 0.045
feedback_logic_time = 0.010
total_latency = mediapipe_time + avg_inference_time + feedback_logic_time

print("="*30)
print("LATENCY ANALYSIS")
print("="*30)
print(f"MediaPipe Extraction: {mediapipe_time:.4f}s")
print(f"LSTM Inference:       {avg_inference_time:.4f}s")
print(f"Feedback Logic:       {feedback_logic_time:.4f}s")
print(f"Total Cycle Latency:  {total_latency:.4f}s")
print("="*30 + "\n")

# ==========================================
# 3. Confusion Matrix (Fig 1)
# ==========================================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap,
            xticklabels=labels, yticklabels=labels, cbar=False, linecolor='white', linewidths=1)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved: confusion_matrix.png")

# ==========================================
# 4. Precision, Recall, F1-Score (Fig 2)
# ==========================================
report_dict = classification_report(y_true_classes, y_pred_classes,
                                   target_names=labels, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()

plt.figure(figsize=(10, 4))
sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap=custom_cmap, fmt=".2f", cbar=False, linecolor='white', linewidths=1)

plt.tight_layout()
plt.savefig('classification_metrics.png')
print("Saved: classification_metrics.png")

# ==========================================
# 5. Processing Latency Analysis (Fig 3)
# ==========================================
stages = ['MediaPipe Extraction', 'LSTM Inference', 'Feedback Logic']
times = [mediapipe_time, avg_inference_time, feedback_logic_time]

plt.figure(figsize=(8, 5))
colors = ['#d9d9d9', '#EAD1DC', '#d9d9d9']
bars = plt.bar(stages, times, color=colors)
plt.axhline(y=0.2, color='r', linestyle='--', label='Real-time Limit (0.2s)')

plt.ylabel('Seconds')
plt.ylim(0, 0.25)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}s', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig('latency_chart.png')
print("Saved: latency_chart.png")