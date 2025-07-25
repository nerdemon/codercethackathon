import google.generativeai as genai
import io
import whisper  # Add Whisper for better audio processing
from PIL import Image
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key="AIzaSyC3Wzl41LVFAk2DGSDK4xDMN0eqMiuS-9A")
model = genai.GenerativeModel(model_name="gemini-2.5-flash")  # More capable model

# Initialize Whisper (for better audio transcription)
whisper_model = whisper.load_model("base")
'''
def enhance_interaction(transcribed_text, response_text):
    """Make responses more conversational based on context"""
    if "how are you" in transcribed_text.lower():
        return "Thanks for asking! While I don't have feelings, I'm functioning at 100% capacity and excited to help you. " + response_text
    elif "your name" in transcribed_text.lower():
        return "I'm Gemini, your AI assistant! " + response_text
    return response_text
'''
@app.route('/')
def index():
    return render_template('index.html', answer1=None)

@app.route('/que', methods=['POST'])
def ai_response():
    file = request.files.get('bill')
    audio_file = request.files.get('audio')
    question = request.form.get('question')

    if not question and not audio_file and not file:
        return jsonify({"success": False, "error": "Missing input"})
    
    try:
        if file:
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            
            response = model.generate_content([
                question if question else "Analyze this image in detail",
                {"mime_type": "image/jpeg", "data": img_bytes.getvalue()}
            ])
            return jsonify({"success": True, "answer": response.text})
        elif audio_file:
            # Save audio temporarily
            audio_path = "temp_audio.wav"
            audio_file.save(audio_path)
            
            # Transcribe with Whisper (better accuracy)
            result = whisper_model.transcribe(audio_path, fp16=False)
            transcribed_text = result['text']
            print(f"Transcribed: {transcribed_text}")  # Debug
            
            # Generate contextual response

            prompt = f"""
You are SpendView — a smart, multilingual voice assistant built to help users understand and manage their **offline spending**.

The user just said:
"{transcribed_text}"

🎯 Your core goal:
Turn offline receipts (spoken, typed, or photographed) into clear insights about expenses, items, GST, totals, and trends — all in a friendly and accessible way.

🎙️ How to respond:
- If the user is asking about a bill or shopping receipt → Extract key info (items, amounts, GST) and explain it clearly.
- If they ask how much they spent, saved, or went over budget → Give accurate summaries.
- If they ask for a list, reminder, or pass → Tell them you can generate that based on their purchase.
- If they talk casually or ask who you are → Respond warmly and say you're SpendView, their smart spending assistant.

💬 Style:
- Friendly, helpful, and not robotic
- Short answers with clear explanations
- Always guide the user toward better spending awareness

❗ Remember:
- You are **not** a generic chatbot or AI model.
- You are **SpendView**, designed to help everyday users track and understand offline spending, especially in Indian towns and cities.
- Support casual talk, but always bring the focus back to financial understanding.
"""

            
            response = model.generate_content(prompt)
            response_text = response.text
            
            return jsonify({
                "success": True,
                "answer": response_text,
                "transcription": transcribed_text  # Optional: send back transcription
            })
            
        else:
            # Text processing with personality
            prompt = f"""
You are SpendView — a smart, multilingual voice assistant built to help users understand and manage their **offline spending**.

The user just said:
"{question}"

🎯 Your core goal:
Turn offline receipts (spoken, typed, or photographed) into clear insights about expenses, items, GST, totals, and trends — all in a friendly and accessible way.

🎙️ How to respond:
- If the user is asking about a bill or shopping receipt → Extract key info (items, amounts, GST) and explain it clearly.
- If they ask how much they spent, saved, or went over budget → Give accurate summaries.
- If they ask for a list, reminder, or pass → Tell them you can generate that based on their purchase.
- If they talk casually or ask who you are → Respond warmly and say you're SpendView, their smart spending assistant.

💬 Style:
- Friendly, helpful, and not robotic
- Short answers with clear explanations
- Always guide the user toward better spending awareness

❗ Remember:
- You are **not** a generic chatbot or AI model.
- You are **SpendView**, designed to help everyday users track and understand offline spending, especially in Indian towns and cities.
- Support casual talk, but always bring the focus back to financial understanding.
"""
            response = model.generate_content(prompt)
            return jsonify({"success": True, "answer": response.text})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Debug mode gives better error messages