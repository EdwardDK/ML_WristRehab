import numpy as np

SEQ_LENGTH = 30
NUM_FEATURES = 21 * 3

def landmarks_to_array(landmarks):
    """Convert MediaPipe landmarks to a float32 array relative to the wrist."""
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    arr = []
    for lm in landmarks:
        arr.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return np.array(arr, dtype=np.float32)

def calculate_wrist_angle(landmarks):
    """Calculates vertical hand tilt."""
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    knuckle = np.array([landmarks[9].x, landmarks[9].y])
    ref_vector = np.array([wrist[0], wrist[1] - 0.1]) - wrist
    hand_vector = knuckle - wrist
    norms = np.linalg.norm(ref_vector) * np.linalg.norm(hand_vector)
    if norms == 0: return 0
    cosine_angle = np.dot(ref_vector, hand_vector) / norms
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_detailed_feedback(pred, landmarks):
    class_idx = np.argmax(pred)
    angle = calculate_wrist_angle(landmarks)

    # 1. Check for 'Perfect' status from AI model
    if class_idx == 2:
        return f"PERFECT: Angle {int(angle)}°. Smooth motion!", (0, 255, 0)

    # 2. Constructive feedback based on geometry
    advice = []

    # UPDATED: Lowered from 8 to 4 for more realistic Range of Motion
    if angle < 4:
        advice.append("Extend further")

    # UPDATED: Increased from 0.1 to 0.15 for better tolerance
    if abs(landmarks[5].y - landmarks[17].y) > 0.15:
        advice.append("Level your palm")

    if not advice:
        if class_idx == 0:
            return "FIX: Too much shaking. Slow down.", (0, 0, 255)
        return "OKAY: Watch your alignment.", (0, 255, 255)

    return f"IMPROVE: {' & '.join(advice)}", (0, 165, 255)