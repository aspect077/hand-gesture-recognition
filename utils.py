import numpy as np

CLASSES = ["OPEN_PALM", "FIST", "THUMBS_UP", "PEACE", "ONE_FINGER"]

def normalize_landmarks(hand_landmarks):
    """21 landmarks -> 63 normalized features."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)          # (21, 3)
    pts -= pts[0]                              # wrist becomes origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale                           # size-invariant
    return pts.flatten()                       # (63,)