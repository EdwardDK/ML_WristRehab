import cv2
import mediapipe as mp
from utils import landmarks_to_array, save_sequence
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
SEQ_LENGTH = 30

cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQ_LENGTH)
seq_id = 0

label = input("Enter label for this recording (bad/okay/perfect): ")

print("Press 'r' to start recording a repetition sequence")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):

        temp_seq = deque(maxlen=SEQ_LENGTH)
        print("Recording...")
        while len(temp_seq) < SEQ_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    temp_seq.append(landmarks_to_array(hand_landmarks.landmark))
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1)
        if len(temp_seq) == SEQ_LENGTH:
            save_sequence(temp_seq, label, seq_id)
            seq_id += 1
            print("Sequence saved.")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
