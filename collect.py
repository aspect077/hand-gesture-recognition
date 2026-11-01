import cv2, csv, os
import mediapipe as mp
from utils import CLASSES, normalize_landmarks

SESSION = 1          # CHANGE THIS for each session: 1, 2, 3
OUT_FILE = f"session{SESSION}.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

keys = {ord(str(i + 1)): i for i in range(len(CLASSES))}
recording = None
counts = {c: 0 for c in CLASSES}

new_file = not os.path.exists(OUT_FILE)
f = open(OUT_FILE, "a", newline="")
writer = csv.writer(f)
if new_file:
    writer.writerow([f"f{i}" for i in range(63)] + ["label", "session"])

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                       # mirror view
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        if recording is not None:
            writer.writerow(list(normalize_landmarks(hl)) + [CLASSES[recording], SESSION])
            counts[CLASSES[recording]] += 1

    status = f"REC: {CLASSES[recording]}" if recording is not None else "Not recording"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for i, c in enumerate(CLASSES):
        cv2.putText(frame, f"{i+1}:{c} {counts[c]}", (10, 60 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.imshow("Collect", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('0'):
        recording = None
    elif k in keys:
        recording = keys[k]

f.close(); cap.release(); cv2.destroyAllWindows()