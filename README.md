# Real-Time Hand Gesture Detection and Recognition Using Computer Vision and Deep Learning

A webcam application that detects a hand, recognizes one of five gestures with a custom-trained neural network, and shows the prediction and confidence in real time.

**Gestures:** `OPEN_PALM`, `FIST`, `THUMBS_UP`, `PEACE`, `ONE_FINGER`

## How it works

```
Webcam -> MediaPipe Hands -> 21 landmarks -> normalize (63 features) -> MLP -> smoothed label
```

1. **OpenCV** captures webcam frames.
2. **MediaPipe Hands** (pretrained) detects the hand and returns 21 3D landmarks.
3. Landmarks are made position- and size-invariant: the wrist becomes the origin and all points are scaled by the largest wrist-to-landmark distance (`utils.py`).
4. A **Multilayer Perceptron** (63 -> 128 -> 64 -> 5, ReLU, softmax; 16,773 parameters) classifies the gesture. It was trained in Google Colab with TensorFlow/Keras.
5. A 7-frame majority vote smooths the prediction to reduce flicker.

For real-time inference, the trained weights are exported to `gesture_weights.npz` and the same network runs in NumPy, so TensorFlow is not needed to run the app.

## Dataset

Custom dataset recorded with `collect.py` (8,624 samples, about 550-620 per class per session):

| Session | Conditions |
|---|---|
| 1 | Indoors, lights on, camera close, right hand |
| 2 | Balcony (daylight), right hand |
| 3 | Lights off, camera farther away, left hand |

The data is in `session1.csv`. The `session` column was assigned afterwards by splitting the file into its three consecutive recording blocks.

## Training and evaluation

- Train/validation: sessions 1-2 (80/20 split). Test: session 3, a whole held-out session with a different hand, lighting and distance.
- Mirror augmentation (negating the x coordinates) was applied to the training data.
- Optimizer: Adam. Loss: sparse categorical cross-entropy. Early stopping on validation loss.
- The deployed model was retrained on all three sessions.

### Results on the held-out test session (2,890 samples)

| Model | Test accuracy |
|---|---|
| MLP (with mirror augmentation) | about 99.4% (2,872 / 2,890) |
| MLP (without mirror augmentation) | 99.41% |
| Random Forest (baseline) | about 99% |

Mirror augmentation made no measurable difference on this test set. Validation accuracy (about 99.9%) is optimistic because consecutive frames from the same recording are near-duplicates; the held-out session is the honest result.

All 18 errors were PEACE or ONE_FINGER predicted as FIST/THUMBS_UP (see `confusion_matrix.png`).

## Run it

Requires **Python 3.11** (MediaPipe 0.10.14 has no builds for 3.13).

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Press `q` to quit. Do not install TensorFlow in the same environment: its protobuf requirement conflicts with this MediaPipe version.

### Files

| File | Purpose |
|---|---|
| `utils.py` | Class names and landmark normalization (shared by collection and the app) |
| `collect.py` | Record labelled landmark samples from the webcam |
| `app.py` | Real-time recognition app |
| `gesture_weights.npz` | Trained MLP weights used by the app |
| `gesture_mlp.keras` | Trained Keras model |
| `session1.csv` | Recorded dataset |

To record new data: set `SESSION` in `collect.py`, run it, press `1`-`5` to record a gesture and `0` to stop.

## Limitations

- Data comes from one person, one webcam and one laptop; it may not generalize to other people.
- Single hand only.
- No "no gesture" class: any hand pose is assigned one of the five labels.
- Depends on MediaPipe detecting the hand; thin fingers at a distance or in the dark gave the few errors seen.
