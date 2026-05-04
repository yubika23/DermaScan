import os
import gdown

FILE_ID = "1GY2_qjKf4vDw1ja04z4uF8zNRO56i-sa"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_disease_model_transfer_best.h5")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}&confirm=t",
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model downloaded! Size: {size_mb:.1f} MB")
else:
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model already exists. Size: {size_mb:.1f} MB")
