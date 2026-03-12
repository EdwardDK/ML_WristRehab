import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from utils import SEQ_LENGTH, NUM_FEATURES

labels = ['bad', 'okay', 'perfect']


def load_raw_dataset():
    """Loads only the original .npy files without augmentation."""
    X, y = [], []
    for idx, label in enumerate(labels):
        folder = f'dataset/{label}'
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found.")
            continue
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(folder, file))
                X.append(seq)
                y.append(idx)
    return np.array(X), np.array(y)


# 1. Load the "Clean" data
X_raw, y_raw = load_raw_dataset()

# 2. Split into Train and Validation sets FIRST
# This prevents identical movements from appearing in both sets
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# 3. Augment ONLY the training data
X_train = []
y_train = []

for i in range(len(X_train_raw)):
    # Original sequence
    X_train.append(X_train_raw[i])
    y_train.append(y_train_raw[i])

    # Augmented sequence (adding noise)
    # This helps the model become invariant to camera jitter
    noise = np.random.normal(0, 0.015, X_train_raw[i].shape)
    X_train.append(X_train_raw[i] + noise)
    y_train.append(y_train_raw[i])

X_train = np.array(X_train)
y_train = to_categorical(np.array(y_train), num_classes=3)
X_val = np.array(X_val_raw)
y_val = to_categorical(y_val_raw, num_classes=3)

# 4. Define Model Architecture
model = Sequential([
    Input(shape=(SEQ_LENGTH, NUM_FEATURES)),
    Conv1D(64, 3, activation='relu'),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Training with EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[
        ModelCheckpoint('model/wrist_model.h5', save_best_only=True),
        early_stop
    ]
)