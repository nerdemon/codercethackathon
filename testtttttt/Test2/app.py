import os
import io
import uuid
import whisper
import torch
from PIL import Image
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', 'cetproject-467019-a3ebc3657680.json')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyC3Wzl41LVFAk2DGSDK4xDMN0eqMiuS-9A")
    STORAGE_BUCKET = os.getenv('STORAGE_BUCKET', "cetproject-467019.firebasestorage.app")
    MODEL_NAME = os.getenv('MODEL_NAME', "gemini-1.5-flash")

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize Firebase
try:
    cred = credentials.Certificate(Config.FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        'storageBucket': Config.STORAGE_BUCKET
    })
    db = firestore.client()
    storage_bucket = storage.bucket()
    print("✅ Firebase initialized successfully")
except Exception as e:
    print(f"❌ Firebase initialization failed: {str(e)}")
    exit(1)

# Configure Gemini
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=Config.MODEL_NAME)
    print("✅ Gemini AI configured successfully")
except Exception as e:
    print(f"❌ Gemini AI configuration failed: {str(e)}")
    exit(1)

# Initialize Whisper
try:
    whisper_model = whisper.load_model("base").to(device)
    print("✅ Whisper model loaded successfully")
except Exception as e:
    print(f"❌ Whisper model loading failed: {str(e)}")
    exit(1)

def save_to_firestore(user_input, ai_response, image_url=None, audio_transcription=None):
    """Save interaction to Firestore with all relevant data"""
    session_id = request.cookies.get('session_id', str(uuid.uuid4()))
    chat_ref = db.collection('chat_logs').document()
    
    chat_data = {
        'timestamp': datetime.now(),
        'user_input': user_input,
        'ai_response': ai_response,
        'session_id': session_id,
        'has_image': image_url is not None,
        'image_url': image_url,
        'has_audio': audio_transcription is not None,
        'audio_transcription': audio_transcription,
        'device_type': request.user_agent.platform,
        'browser': request.user_agent.browser
    }
    
    chat_ref.set(chat_data)
    return chat_data

def upload_to_storage(file, file_type='image'):
    """Upload file to Firebase Storage and return public URL"""
    try:
        # Generate unique filename
        filename = f"uploads/{file_type}/{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        blob = storage_bucket.blob(filename)
        
        if file_type == 'image':
            # Process image
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            blob.upload_from_file(img_bytes, content_type='image/jpeg')
        else:
            # Process audio
            blob.upload_from_file(file, content_type=file.content_type)
            
        blob.make_public()
        return blob.public_url
    except Exception as e:
        print(f"File upload error: {str(e)}")
        return None

@app.route('/')
def index():
    resp = make_response(render_template('index.html', answer1=None))
    if not request.cookies.get('session_id'):
        resp.set_cookie('session_id', str(uuid.uuid4()), max_age=60*60*24*30)  # 30 days
    return resp

@app.route('/que', methods=['POST'])
def ai_response():
    file = request.files.get('bill')
    audio_file = request.files.get('audio')
    question = request.form.get('question', '').strip()
    
    # Process inputs
    transcribed_text = ""
    user_input = ""
    image_url = None
    audio_url = None
    
    # Handle audio file
    if audio_file and audio_file.filename != '':
        audio_path = f"temp_audio_{uuid.uuid4()}.wav"
        audio_file.save(audio_path)
        result = whisper_model.transcribe(audio_path, fp16=False)
        transcribed_text = result['text']
        user_input = transcribed_text
        audio_url = upload_to_storage(audio_file, 'audio')
        os.remove(audio_path)  # Clean up temp file
        
    # Handle text input
    if question:
        user_input = f"{user_input}. {question}" if user_input else question
        
    # Handle image file
    if file and file.filename != '':
        image_url = upload_to_storage(file, 'image')
        if not image_url:
            return jsonify({"success": False, "error": "Failed to process image"}), 400
    
    # Validate at least one input method
    if not any([user_input, image_url]):
        return jsonify({"success": False, "error": "Missing input - please provide text, audio, or image"}), 400
    
    try:
        # Generate response based on input type
        if image_url:
            # Multimodal request with image
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            
            response = model.generate_content([
                build_prompt(user_input),
                {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}
            ])
        else:
            # Text-only request
            response = model.generate_content(build_prompt(user_input))
        
        # Save to Firestore
        chat_data = save_to_firestore(
            user_input=user_input,
            ai_response=response.text,
            image_url=image_url,
            audio_transcription=transcribed_text if audio_url else None
        )
        
        return jsonify({
            "success": True,
            "answer": response.text,
            "image_url": image_url,
            "audio_url": audio_url,
            "transcription": transcribed_text,
            "timestamp": chat_data['timestamp'].isoformat(),
            "session_id": chat_data['session_id']
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def build_prompt(user_input):
    """Construct the system prompt for Gemini"""
    return f"""
You are SpendView — a smart, multilingual voice assistant built to help users understand and manage their **offline spending** in a clear, structured, and accessible way.

The user just said or submitted:
"{user_input if user_input else "Analyze this image"}"

🎯 Your core mission:
Process offline receipts — whether spoken, typed, or photographed — and convert them into clean financial insights. You should always extract relevant data like items, quantities, amounts, GST, and totals in a structured manner, even if the input is unstructured or partial.

[Rest of your existing prompt...]
"""

@app.route('/chat-history')
def chat_history():
    try:
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({"success": False, "error": "No session ID found"}), 400

        chats = db.collection('chat_logs') \
                 .where('session_id', '==', session_id) \
                 .order_by('timestamp', direction=firestore.Query.DESCENDING) \
                 .limit(20) \
                 .stream()
        
        history = []
        for chat in chats:
            chat_data = chat.to_dict()
            chat_data['id'] = chat.id
            chat_data['timestamp'] = chat_data['timestamp'].isoformat()
            history.append(chat_data)
        
        return jsonify({"success": True, "history": history})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "services": {
            "firebase": "active",
            "gemini": "active",
            "whisper": "active",
            "storage": "active"
        }
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6069, debug=True)