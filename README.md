# 🔬 DermaScan — AI Skin Disease Detection

A web-based application that uses a Convolutional Neural Network (CNN) trained on the
HAM10000 dataset to classify skin lesions into 7 categories.

Built by **Ubika Chaudhary**, **Moksha**, and **Muskan Mehra** @ MMDU, Ambala.

---

## 📁 Folder Structure

```
DermaScan/
│
├── Backend/
│   ├── app.py                  ← Flask API server
│   ├── train_model.py          ← CNN training script
│   ├── prepare_ham10000.py     ← Dataset preparation script
│   ├── requirements.txt        ← Python dependencies
│   │
│   ├── models/                 ← Trained .h5 model files go here
│   │   └── .gitkeep
│   │
│   ├── uploads/                ← Temp image storage (auto-cleared)
│   │   └── .gitkeep
│   │
│   └── dataset/
│       ├── train/
│       │   ├── Melanoma/
│       │   ├── Melanocytic_nevus/
│       │   ├── Basal_cell_carcinoma/
│       │   ├── Actinic_keratosis/
│       │   ├── Benign_keratosis/
│       │   ├── Dermatofibroma/
│       │   └── Vascular_lesion/
│       ├── validation/         ← same 7 subfolders
│       └── test/               ← same 7 subfolders
│
├── Frontend/
│   ├── index.html              ← Main webpage
│   ├── style.css               ← Styles
│   ├── main.js                 ← Frontend logic & API calls
│   └── Images/                 ← Skin type icons & illustrations
│       └── .gitkeep
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Install Python dependencies

```bash
cd Backend
pip install -r requirements.txt
```

> **No GPU?** Replace `tensorflow` with `tensorflow-cpu` in requirements.txt

---

### Step 2 — Run the server (Demo Mode, no training needed)

```bash
cd Backend
python app.py
```

The server starts at **http://localhost:5000**  
Open `Frontend/index.html` in your browser and start scanning!

> In demo mode, predictions are random. Train the model (Step 3) for real results.

---

### Step 3 — Train the Model (for real predictions)

#### 3a. Download the HAM10000 Dataset
1. Go to https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
2. Download and extract to `Backend/dataset/train/archive/`

#### 3b. Prepare the dataset
```bash
cd Backend
# Edit the paths inside prepare_ham10000.py first, then:
python prepare_ham10000.py
```

#### 3c. Train the model
```bash
python train_model.py
# Choose option 2 (Transfer Learning) — recommended
```

Training saves the model to `Backend/models/`.  
Restart `app.py` and it will automatically load the trained model.

---

## 🧠 Model Details

| Property       | Value                          |
|---------------|-------------------------------|
| Architecture  | MobileNetV2 (Transfer Learning)|
| Input size    | 224 × 224 × 3 (RGB)           |
| Output        | 7-class softmax                |
| Dataset       | HAM10000 (~10,000 images)      |
| Target accuracy | 90%+                        |
| Training time | ~2–4 hours (GPU) / longer CPU |

### Classes Detected

| Code   | Disease                  | Severity  |
|--------|--------------------------|-----------|
| mel    | Melanoma                 | 🔴 Critical |
| akiec  | Actinic Keratosis        | 🟠 High    |
| bcc    | Basal Cell Carcinoma     | 🟠 High    |
| nv     | Melanocytic Nevus        | 🔵 Moderate |
| bkl    | Benign Keratosis         | 🟢 Low     |
| df     | Dermatofibroma           | 🟢 Low     |
| vasc   | Vascular Lesion          | 🟢 Low     |

---

## 🌐 API Endpoints

| Method | Endpoint        | Description                    |
|--------|----------------|-------------------------------|
| GET    | `/`             | Server status & model info    |
| POST   | `/api/predict`  | Upload image → get prediction |
| GET    | `/api/classes`  | List all detectable classes   |
| GET    | `/api/health`   | Health check                  |

### Example predict request
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@your_skin_image.jpg"
```

### Example response
```json
{
  "success": true,
  "prediction": "Actinic_keratosis",
  "confidence": "87.43%",
  "test_mode": false,
  "all_predictions": [
    { "condition": "Actinic_keratosis", "confidence": 0.8743 },
    { "condition": "Basal_cell_carcinoma", "confidence": 0.0821 },
    { "condition": "Melanoma", "confidence": 0.0312 }
  ],
  "recommendations": {
    "description": "...",
    "care_tips": ["..."],
    "when_to_see_doctor": "..."
  }
}
```

---

## ⚠️ Disclaimer

This tool is for **educational and screening purposes only**.  
It does **not** replace professional medical diagnosis.  
Always consult a qualified dermatologist for any skin concerns.

---

## 📄 License

Built for academic submission at Maharishi Markandeshwar Deemed University (MMDU), Mullana, Ambala.
