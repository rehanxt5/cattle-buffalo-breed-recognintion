# 🐄 Cattle & Buffalo Breed Recognition

A deep learning image classification project that identifies cattle and buffalo breeds from images using **EfficientNetB2** transfer learning.

---

## 📌 Overview

This project uses a convolutional neural network (CNN) based on EfficientNetB2 to classify images into **6 categories**:

| Class | Description |
|-------|-------------|
| `HolsteinFriesian` | High-yield dairy cattle breed (black & white markings) |
| `Jersey` | Small dairy cattle breed (tan/brown color) |
| `gir` | Indian Zebu cattle breed |
| `hariana` | Indian draft & dairy cattle breed |
| `murrah` | High-yielding buffalo breed |
| `NotCattle` | Non-cattle / background images |

---

## ✨ Features

- 🔁 **Transfer Learning** — Fine-tuned EfficientNetB2 pre-trained on ImageNet
- ⚖️ **Class Imbalance Handling** — Balanced class weights during training
- 🔄 **Data Augmentation** — Rotation, shifting, shearing, zoom, and flips
- 📉 **Two-Phase Training** — Head training followed by full fine-tuning
- 📊 **Dynamic LR Scheduling** — ReduceLROnPlateau callback
- 💾 **Model Checkpointing** — Saves the best model based on validation loss
- 🛡️ **Data Validation** — Automatically removes corrupted images before training

---

## 🏗️ Model Architecture

```
Input Image (260 × 260 × 3)
        ↓
EfficientNetB2 (ImageNet weights, frozen in Phase 1)
        ↓
GlobalAveragePooling2D
        ↓
Dense(6, activation="softmax")
        ↓
Predicted Breed (6 classes)
```

### Training Strategy

**Phase 1 — Head Training (20 epochs)**
- EfficientNetB2 base layers are **frozen**
- Only the custom classification head is trained
- Optimizer: Adam (default learning rate)

**Phase 2 — Fine-Tuning (30 epochs)**
- All layers are **unfrozen**
- Entire network is fine-tuned with a low learning rate (`1e-5`)
- Optimizer: Adam (`lr=1e-5`)

### Callbacks
| Callback | Configuration |
|----------|---------------|
| `EarlyStopping` | Monitors `val_loss`, patience=7, restores best weights |
| `ReduceLROnPlateau` | Monitors `val_loss`, factor=0.2, patience=3 |
| `ModelCheckpoint` | Saves `best_efficientnet_cattle_b2.h5` on best `val_loss` |

---

## 📁 Project Structure

```
cattle-buffalo-breed-recognition/
├── train-model.ipynb          # Full training pipeline
├── main-model.ipynb           # Inference / prediction notebook
├── class_indices.json         # Class name → index mapping
└── test.jpg                   # Sample test image
```

> **Note:** The dataset and trained model weights are hosted externally on Google Drive and are downloaded at runtime.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install tensorflow efficientnet pandas scikit-learn pillow numpy
```

Or install inside the notebooks (as done in `train-model.ipynb`):

```bash
pip install -q efficientnet pandas scikit-learn
```

---

## 🚀 Usage

### 1. Training the Model

Open and run **`train-model.ipynb`**:

1. The notebook downloads the dataset from Google Drive automatically.
2. Invalid/corrupted images are removed before training.
3. Data generators are created with augmentation (train) and normalization (val).
4. The model trains in two phases with early stopping and checkpointing.

**Outputs:**
- `best_efficientnet_cattle_b2.h5` — best model weights
- `final_efficientnet_cattle_b2.h5` — final model weights
- `class_indices.json` — class name to index mapping

### 2. Running Inference

Open and run **`main-model.ipynb`**:

1. The pre-trained model is downloaded from Google Drive.
2. Supply the path to an image for classification.

**Example Prediction Code:**

```python
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 260
MODEL_PATH = "best_efficientnet_cattle_b2.h5"
MAPPING_JSON = "class_indices.json"

model = load_model(MODEL_PATH)
idx_to_class = {v: k for k, v in json.load(open(MAPPING_JSON)).items()}

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

x = prepare_image("test.jpg")
probs = model.predict(x, verbose=0)[0]
breed = idx_to_class[int(np.argmax(probs))]
confidence = np.max(probs) * 100

print(f"Prediction : {breed}")
print(f"Confidence : {confidence:.1f}%")
```

**Sample Output:**
```
Prediction : murrah
Confidence : 99.5%
```

---

## 📊 Dataset

- **Format:** JPEG images organized in class subdirectories
- **Split:** `dataset/train/` and `dataset/val/`
- **Image Size:** 260 × 260 pixels
- **Batch Size:** 16
- **Source:** Downloaded from Google Drive during training

> The dataset is not included in this repository. Run `train-model.ipynb` to download it automatically.

---

## 🗂️ Class Indices

```json
{
  "HolsteinFriesian": 0,
  "Jersey": 1,
  "NotCattle": 2,
  "gir": 3,
  "hariana": 4,
  "murrah": 5
}
```

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| TensorFlow / Keras | Deep learning framework |
| EfficientNetB2 | Pre-trained CNN backbone |
| scikit-learn | Class weight computation |
| Pandas | Data manipulation |
| PIL / Pillow | Image loading & validation |
| NumPy | Numerical operations |
| gdown | Google Drive file download |

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.
