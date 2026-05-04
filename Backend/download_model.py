"""
download_model.py
Runs once at startup — downloads the trained .h5 model from Google Drive
if it doesn't already exist locally.
"""

import os
import gdown

# ── Paste your Google Drive file ID here ──────────────────
# Get it from your shareable link:
# https://drive.google.com/file/d/YOUR_FILE_ID_HERE/view
FILE_ID = os.environ.get("MODEL_FILE_ID", "YOUR_GOOGLE_DRIVE_FILE_ID_HERE")

# ── Where to save the model ───────────────────────────────
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_disease_model_transfer_best.h5")

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at {MODEL_PATH} — skipping download.")
        return

    if FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        print("⚠️  No Google Drive FILE_ID set. Skipping model download.")
        print("    Set MODEL_FILE_ID environment variable in Railway dashboard.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📥 Downloading model from Google Drive (ID: {FILE_ID}) ...")

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"✅ Model downloaded successfully! ({size_mb:.1f} MB)")
        else:
            print("❌ Download failed — file not found after download.")
    except Exception as e:
        print(f"❌ Download error: {e}")
        print("   App will start in demo mode.")

if __name__ == "__main__":
    download_model()
