# Photo to Measurements

Proof of concept for estimating human body measurements (chest, hips, inseam, sleeve length, etc.) from:
- 1 front photo
- 1 side photo  
- Declared body height

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install opencv-python mediapipe numpy

# 3. Run on a subject
python src/run.py
```

---

## 📁 Project Structure

```
PhotoToMeasurements/
├── src/
│   └── run.py              # Main processing script
├── input/
│   └── subject_01/         # Input images go here
│       ├── front_side.JPG  # Front-facing photo
│       └── meta.json       # Subject metadata
├── output/
│   └── subject_01/         # Generated outputs
│       ├── front_pose.jpg       # Image with skeleton overlay
│       ├── front_mask.png       # Person segmentation mask
│       ├── front_mask_overlay.jpg  # Segmentation visualization
│       └── quality.json         # Detection quality metrics
└── README.md
```

---

## 🔄 How It Works (Step by Step)

### Input
You provide a front-facing photo of a person.

### Step 1: Pose Detection
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Your Photo    │ ──▶ │  MediaPipe Pose  │ ──▶ │  33 Body Points │
│   (front.jpg)   │     │  Neural Network  │     │  (x, y coords)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

The pose model finds **33 keypoints** on the body:
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists
- Torso: hips
- Lower body: knees, ankles, feet

**Output:** `front_pose.jpg` - your photo with a skeleton drawn on it

### Step 2: Person Segmentation
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Your Photo    │ ──▶ │ Selfie Segmenter │ ──▶ │  Binary Mask    │
│   (front.jpg)   │     │  Neural Network  │     │  (person=white) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

The segmentation model creates a **mask** showing where the person is:
- White pixels = person
- Black pixels = background

**Outputs:**
- `front_mask.png` - the raw mask
- `front_mask_overlay.jpg` - green overlay showing detected person area

### Step 3: Quality Check
```json
{
  "pose_has_landmarks": true,   // Was a person detected?
  "segmentation_mean": 207.7    // How much of image is person
}
```

---

## 🧠 What Happens Inside the Neural Networks?

```
        YOUR IMAGE                    NEURAL NETWORK                    OUTPUT
    ┌───────────────┐            ┌─────────────────────┐         ┌──────────────┐
    │               │            │  Layer 1: Features  │         │              │
    │  RGB Pixels   │            │  Layer 2: Patterns  │         │  Keypoints   │
    │  3024 x 4032  │  ──────▶   │  Layer 3: Body Parts│  ────▶  │     or       │
    │  = 12M pixels │            │  Layer 4: Positions │         │    Mask      │
    │               │            │         ...         │         │              │
    └───────────────┘            └─────────────────────┘         └──────────────┘
         
    Image becomes                Network learned from              Structured
    a number array               millions of photos                predictions
```

1. **Image → Tensor**: Photo converted to numbers (RGB values 0-255)
2. **Forward Pass**: Numbers flow through network layers
3. **Prediction**: Network outputs coordinates or pixel classifications

---

## 🎯 Core Design Principles

| Principle | Description |
|-----------|-------------|
| **Fail Fast** | Reject bad images before attempting measurements |
| **Geometry First** | Use actual coordinates, not black-box guessing |
| **Confidence Scores** | Every measurement has a reliability score |
| **Debuggable** | All steps produce visual outputs you can inspect |

---

## 📊 Current Capabilities

- [x] Pose landmark detection (33 body keypoints)
- [x] Person segmentation (foreground/background separation)
- [x] Quality metrics output
- [ ] Actual body measurements (coming soon)
- [ ] Side photo processing (coming soon)
- [ ] Height calibration (coming soon)

---

## 🛠️ Technical Details

### Dependencies
- **OpenCV** - Image reading/writing and processing
- **MediaPipe** - Google's ML models for pose and segmentation
- **NumPy** - Array operations

### Models Used
| Model | File | Purpose |
|-------|------|---------|
| Pose Landmarker | `pose_landmarker.task` | Detects 33 body keypoints |
| Selfie Segmenter | `selfie_segmenter.tflite` | Separates person from background |

Models are automatically downloaded on first run (~5MB total).