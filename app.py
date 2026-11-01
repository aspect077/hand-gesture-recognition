import cv2, time
import numpy as np
from collections import deque, Counter
import mediapipe as mp
from utils import CLASSES, normalize_landmarks

# Load the trained MLP weights (3 dense layers) and run it with NumPy
w = np.load("gesture_weights.npz")
W1, b1, W2, b2, W3, b3 = [w[f"arr_{i}"] for i in range(6)]

def predict(x):
    h = np.maximum(x @ W1 + b1, 0)       # Dense 128, ReLU
    h = np.maximum(h @ W2 + b2, 0)       # Dense 64, ReLU
    z = h @ W3 + b3                      # Dense 5
    e = np.exp(z - z.max())              # softmax
    return e / e.sum()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

history = deque(maxlen=7)
cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    label, conf, detected = "-", 0.0, False
    if result.multi_hand_landmarks:
        detected = True
        hl = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        probs = predict(normalize_landmarks(hl))
        history.append(int(np.argmax(probs)))
        idx = Counter(history).most_common(1)[0][0]
        label, conf = CLASSES[idx], float(probs[idx])
    else:
        history.clear()

    now = time.time(); fps = 1 / (now - prev); prev = now
    cv2.putText(frame, f"Hand: {'YES' if detected else 'NO'}", (10, 30), 0, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Gesture: {label}", (10, 60), 0, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Confidence: {conf*100:.1f}%", (10, 90), 0, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 120), 0, 0.8, (0,255,0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()