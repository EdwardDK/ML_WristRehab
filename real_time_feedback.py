import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from utils import landmarks_to_array, get_detailed_feedback, SEQ_LENGTH

model = load_model('model/wrist_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQ_LENGTH)
pred_history = deque(maxlen=10)

smoothed_landmarks = None
alpha = 0.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            raw = landmarks_to_array(hand_landmarks.landmark)
            smoothed_landmarks = raw if smoothed_landmarks is None else (alpha * raw + (1 - alpha) * smoothed_landmarks)
            sequence.append(smoothed_landmarks)

            if len(sequence) == SEQ_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred_history.append(np.argmax(res))


                stable_idx = max(set(pred_history), key=list(pred_history).count)
                if list(pred_history).count(stable_idx) >= 5:
                    msg, color = get_detailed_feedback(res, hand_landmarks.landmark)
                else:
                    msg, color = "Stabilizing...", (128, 128, 128)

                cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
                cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Feedback", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()