"""
DermaScan — Skin Disease Detection API
Flask backend with TensorFlow/Keras CNN support
Falls back to demo mode if no model file is found.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import io, os, random, logging, json, smtplib
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename

# ─── CONFIG ────────────────────────────────────────────
IMG_SIZE       = 224
UPLOAD_FOLDER  = 'uploads'
MAX_FILE_SIZE  = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXT    = {'png', 'jpg', 'jpeg'}
MESSAGES_FILE  = 'messages.json'

# ─── EMAIL CONFIG ──────────────────────────────────────
# Fill these in with your Gmail credentials.
# Use a Gmail App Password (NOT your real password):
#   1. Go to myaccount.google.com → Security → 2-Step Verification → ON
#   2. Then go to myaccount.google.com → Security → App Passwords
#   3. Create one for "Mail" → copy the 16-character password here
SENDER_EMAIL    = 'immoksha7@gmail.com'         # ← Gmail you send FROM
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', '')        # ← 16-char Gmail App Password (replace this!)
RECIPIENT_EMAILS = [
    'immoksha7@gmail.com',
    'yubikachaudhary@gmail.com',
]

# Paths to look for a saved model (priority order)
MODEL_PATHS = [
    'models/skin_disease_model_transfer_best.h5',
    'models/skin_disease_model_transfer.h5',
    'models/skin_disease_model_best.h5',
    'models/skin_disease_model.h5',
]

CLASS_NAMES = [
    'Melanoma',
    'Melanocytic_nevus',
    'Basal_cell_carcinoma',
    'Actinic_keratosis',
    'Benign_keratosis',
    'Dermatofibroma',
    'Vascular_lesion',
]

# ─── FLASK SETUP ────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─── LOAD TENSORFLOW MODEL ─────────────────────────────
model             = None
tensorflow_available = False

def load_model():
    """Try to load TensorFlow and a saved model."""
    global model, tensorflow_available
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.get_logger().setLevel('ERROR')    # suppress TF logs
        log.info(f"TensorFlow {tf.__version__} loaded.")

        for path in MODEL_PATHS:
            if os.path.exists(path):
                log.info(f"Loading model from {path} …")
                model = keras.models.load_model(path)
                tensorflow_available = True
                log.info("✅ Model loaded successfully!")
                return

        # No saved model — try building MobileNetV2 as a baseline
        # (weights only; accuracy will be random until fine-tuned)
        log.warning("No saved model found. Building bare MobileNetV2 (untrained head).")
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights='imagenet'
        )
        base.trainable = False
        x = base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
        model = tf.keras.Model(inputs=base.input, outputs=out)
        tensorflow_available = True
        log.warning("⚠️  Using untrained head — predictions NOT accurate. Train the model first.")

    except ImportError:
        log.warning("TensorFlow not installed. Running in demo mode.")
    except Exception as e:
        log.error(f"Model loading failed: {e}")

load_model()

# ─── IMAGE PREPROCESSING ───────────────────────────────
def preprocess_image(file_path: str) -> np.ndarray:
    """Load, resize, and normalise an image for CNN inference."""
    img = Image.open(file_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # [0,1]
    return np.expand_dims(arr, 0)                    # (1, H, W, 3)

# ─── PREDICTION ────────────────────────────────────────
def predict_image(file_path: str) -> dict:
    """Run inference and return structured prediction dict."""

    if tensorflow_available and model is not None:
        x     = preprocess_image(file_path)
        probs = model.predict(x, verbose=0)[0]        # shape (7,)
        top_idx  = int(np.argmax(probs))
        top_conf = float(probs[top_idx])

        # Top-3 sorted predictions
        ranked = sorted(
            [(CLASS_NAMES[i], float(p)) for i, p in enumerate(probs)],
            key=lambda t: t[1], reverse=True
        )
        top3 = [{'condition': c, 'confidence': round(p, 4)} for c, p in ranked[:3]]

        return {
            'prediction': CLASS_NAMES[top_idx],
            'confidence': f"{top_conf * 100:.2f}%",
            'all_predictions': top3,
            'test_mode': False,
        }

    # ── Demo / fallback ──
    predicted = random.choice(CLASS_NAMES)
    confidence = random.uniform(0.70, 0.95)
    others = [c for c in CLASS_NAMES if c != predicted]
    random.shuffle(others)
    top3 = [
        {'condition': predicted,   'confidence': round(confidence, 4)},
        {'condition': others[0],   'confidence': round(random.uniform(0.03, 0.18), 4)},
        {'condition': others[1],   'confidence': round(random.uniform(0.01, 0.08), 4)},
    ]
    return {
        'prediction': predicted,
        'confidence': f"{confidence * 100:.2f}%",
        'all_predictions': top3,
        'test_mode': True,
    }

# ─── RECOMMENDATIONS ───────────────────────────────────
RECOMMENDATIONS = {
    'Melanoma': {
        'description': 'Melanoma is the most serious type of skin cancer, arising from melanocytes. Early detection is critical for survival.',
        'care_tips': [
            'See a dermatologist IMMEDIATELY for a proper biopsy',
            'Avoid all unprotected sun exposure; use SPF 50+ sunscreen',
            'Document the lesion with photos to track changes',
            'Do not attempt self-treatment under any circumstances',
        ],
        'when_to_see_doctor': 'URGENTLY — This condition requires same-day or next-day medical evaluation.',
    },
    'Melanocytic_nevus': {
        'description': 'A melanocytic nevus (common mole) is a benign growth of melanocytes. Most are harmless, but regular monitoring is important.',
        'care_tips': [
            'Use the ABCDE rule: watch for Asymmetry, Border irregularity, Colour change, Diameter > 6mm, Evolution',
            'Apply broad-spectrum SPF 30+ sunscreen daily',
            'Avoid tanning beds and excessive UV exposure',
            'Schedule annual full-body skin checks',
        ],
        'when_to_see_doctor': 'If the mole changes in size, shape, colour, or becomes itchy or painful.',
    },
    'Basal_cell_carcinoma': {
        'description': 'The most common skin cancer. Basal cell carcinoma grows slowly and rarely spreads, but it must be treated to prevent local tissue destruction.',
        'care_tips': [
            'Consult a dermatologist to discuss excision, Mohs surgery, or topical therapy',
            'Protect skin with SPF 30+ sunscreen and protective clothing',
            'Avoid peak UV hours (10 am – 4 pm)',
            'Attend all follow-up appointments as recurrence is possible',
        ],
        'when_to_see_doctor': 'As soon as possible — within the next few weeks for proper treatment.',
    },
    'Actinic_keratosis': {
        'description': 'Rough, scaly patches caused by cumulative sun (UV) damage. Actinic keratoses are pre-cancerous and can progress to squamous cell carcinoma if untreated.',
        'care_tips': [
            'See a dermatologist for topical fluorouracil, cryotherapy, or photodynamic therapy',
            'Apply high-SPF sunscreen every morning',
            'Wear wide-brimmed hats and UPF-rated clothing outdoors',
            'Avoid peak sun hours and tanning beds',
        ],
        'when_to_see_doctor': 'Soon — within 2–4 weeks to prevent potential cancer progression.',
    },
    'Benign_keratosis': {
        'description': 'Benign keratoses (seborrhoeic keratoses) are harmless, age-related skin growths with a waxy, stuck-on appearance. They are not contagious and do not become cancerous.',
        'care_tips': [
            'No treatment is medically necessary',
            'Removal is available (cryotherapy, curettage) for cosmetic concerns or if irritated',
            'Moisturise to reduce mild itching',
            'Mention any rapidly changing lesion to your doctor',
        ],
        'when_to_see_doctor': 'If the lesion becomes irritated, bleeds, or changes rapidly in appearance.',
    },
    'Dermatofibroma': {
        'description': 'A dermatofibroma is a common, firm, benign skin nodule thought to be a reaction to minor injury. It dimples inward when pinched (Fitzpatrick sign).',
        'care_tips': [
            'No treatment is required in most cases',
            'Avoid picking or scratching the lesion',
            'Surgical excision is an option if the lesion is bothersome or cosmetically concerning',
            'Monitor for any rapid change in size or colour',
        ],
        'when_to_see_doctor': 'If the lesion grows rapidly, becomes painful, or changes in appearance.',
    },
    'Vascular_lesion': {
        'description': 'Vascular lesions are abnormalities of blood vessels in the skin, including haemangiomas, port-wine stains, and spider naevi. Most are benign.',
        'care_tips': [
            'Most vascular lesions require no treatment',
            'Laser therapy (pulsed-dye laser) is effective for cosmetic removal',
            'Protect the area from sun exposure with SPF 30+ sunscreen',
            'Use a gentle, fragrance-free skincare routine',
        ],
        'when_to_see_doctor': 'If the lesion bleeds easily, grows rapidly, or changes in appearance.',
    },
}

DEFAULT_REC = {
    'description': 'A skin condition was detected. Please consult a qualified dermatologist for an accurate diagnosis.',
    'care_tips': ['Consult a dermatologist for proper evaluation and treatment'],
    'when_to_see_doctor': 'As soon as possible for professional evaluation.',
}

# ─── HELPERS ───────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def validate_image(path: str) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False

# ─── CONTACT HELPERS ───────────────────────────────────

def save_message(data: dict) -> bool:
    """Append a contact message to messages.json."""
    try:
        messages = []
        if os.path.exists(MESSAGES_FILE):
            with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        messages.append(data)
        with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        log.info(f"Message saved — total messages: {len(messages)}")
        return True
    except Exception as e:
        log.error(f"Failed to save message: {e}")
        return False


def send_email(data: dict) -> bool:
    """Send contact form submission to both recipient emails via Gmail SMTP."""
    try:
        subject = f"📬 DermaScan Contact — {data['name']}"
        body = f"""
New contact form submission on DermaScan
─────────────────────────────────────────
Name    : {data['name']}
Email   : {data['email']}
Phone   : {data.get('phone', 'Not provided')}
Time    : {data['timestamp']}
─────────────────────────────────────────
Message :

{data['message']}

─────────────────────────────────────────
Reply directly to: {data['email']}
        """.strip()

        msg = MIMEMultipart()
        msg['From']    = SENDER_EMAIL
        msg['To']      = ', '.join(RECIPIENT_EMAILS)
        msg['Subject'] = subject
        msg['Reply-To'] = data['email']
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())

        log.info(f"Email sent to {RECIPIENT_EMAILS}")
        return True
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return False


# ─── ROUTES ────────────────────────────────────────────
@app.route('/app')
def frontend():
    return send_from_directory('../Frontend', 'index.html')

@app.route('/app/<path:filename>')
def frontend_files(filename):
    return send_from_directory('../Frontend', filename)

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'DermaScan Skin Disease Detection API',
        'version': '2.0',
        'tensorflow_available': tensorflow_available,
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if not validate_image(filepath):
            return jsonify({'error': 'The uploaded file is not a valid image.'}), 400

        result = predict_image(filepath)
        rec    = RECOMMENDATIONS.get(result['prediction'], DEFAULT_REC)

        response = {
            'success': True,
            **result,
            'recommendations': rec,
        }
        return jsonify(response), 200

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'skin_diseases': CLASS_NAMES,
        'count': len(CLASS_NAMES),
        'tensorflow_available': tensorflow_available,
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'tensorflow': tensorflow_available,
        'model': model is not None,
    })


@app.route('/api/contact', methods=['POST'])
def contact():
    """Receive contact form, save to file and send email."""
    try:
        body = request.get_json()
        if not body:
            return jsonify({'error': 'No data provided'}), 400

        name    = (body.get('name')    or '').strip()
        email   = (body.get('email')   or '').strip()
        phone   = (body.get('phone')   or '').strip()
        message = (body.get('message') or '').strip()

        if not name or not email or not message:
            return jsonify({'error': 'Name, email and message are required.'}), 400

        data = {
            'name':      name,
            'email':     email,
            'phone':     phone or 'Not provided',
            'message':   message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        saved  = save_message(data)
        emailed = send_email(data)

        if not saved and not emailed:
            return jsonify({'error': 'Failed to process your message. Please try again.'}), 500

        return jsonify({
            'success': True,
            'saved':   saved,
            'emailed': emailed,
            'message': 'Your message has been received! We\'ll get back to you soon.',
        }), 200

    except Exception as e:
        log.error(f"Contact error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Return all saved contact messages (for internal use)."""
    try:
        if not os.path.exists(MESSAGES_FILE):
            return jsonify({'messages': [], 'count': 0}), 200
        with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        return jsonify({'messages': messages, 'count': len(messages)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16 MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404

# ─── ENTRY POINT ───────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '='*60)
    if tensorflow_available and model is not None:
        print('✅  TensorFlow loaded — real predictions enabled!')
    else:
        print('⚠️   TensorFlow not available — running in DEMO mode.')
        print('     Install TensorFlow and train a model to enable real predictions.')
        print('     Run:  pip install tensorflow')
        print('     Then: python train_model.py')
    print('='*60)
    print('🌐  Starting DermaScan API on http://localhost:5000\n')
    app.run(debug=True, host='0.0.0.0', port=5000)