import google.generativeai as genai
import io
from PIL import Image
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key="AIzaSyAV4g4LIPbwTIb6zMr1xBaHkRwwPdlDY5I")  # replace with your actual Gemini API key
model = genai.GenerativeModel(model_name="gemini-2.5-flash")

@app.route('/')
def index():
    return render_template('index.html', answer1=None)

@app.route('/que', methods=['POST'])
def ai_response():
    # Check for image upload
    file = request.files.get('bill')
    # Check for audio upload
    audio_file = request.files.get('audio')
    question = request.form.get('question')

    if not question and not audio_file and not file:
        return jsonify({"success": False, "error": "Missing input. Please provide text, audio, or image input."})
    
    try:
        if file:
            # Process with image
            img = Image.open(file).convert("RGB")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            image_data = img_bytes.getvalue()
            
            response = model.generate_content([
                question if question else "Describe this image",
                {"mime_type": "image/jpeg", "data": image_data}
            ])
            
        elif audio_file:
            # Process audio file
            audio_bytes = audio_file.read()
            
            # Save the audio file (optional)
            with open("recorded_audio.wav", "wb") as f:
                f.write(audio_bytes)
            
            # Send to Gemini for processing
            audio_part = {
                "mime_type": "audio/wav",
                "data": audio_bytes
            }
            
            prompt = question if question else "Answer the question in this audio message "
            response = model.generate_content([prompt, audio_part])
        else:
            # Process text only
            response = model.generate_content(question)
            
        return jsonify({"success": True, "answer": response.text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)