#!/usr/bin/env python3
"""
Web-based VoiceBot Demo for Remote Access
Provides a web interface with audio streaming for remote users
"""

import os
import sys
import time
import threading
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import tempfile

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    # Use the same import pattern as your working demo_chatbot.py
    import whisper
    print(f"Whisper imported successfully, version: {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
    import torch
    from TTS.api import TTS
    import requests
    import soundfile as sf
    import numpy as np
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install openai-whisper torch TTS requests soundfile numpy flask python-dotenv")
    print("Note: Make sure to install 'openai-whisper' not 'whisper' to avoid conflicts with Graphite whisper")
    sys.exit(1)

# Load environment variables (optional)
try:
    load_dotenv()
except:
    pass  # Continue without .env file

app = Flask(__name__)

class WebVoiceBot:
    def __init__(self):
        print("Initializing Web VoiceBot...")
        
        # Initialize AI components
        self.whisper_model = None
        self.tts = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "orca2:7b"
        
        # Audio settings
        self.sample_rate = 16000
        
        print("Web VoiceBot initialized successfully!")
    
    def load_models(self):
        """Load AI models"""
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            print("Loading Whisper model...")
            # Debug: Check if whisper has load_model attribute
            if not hasattr(whisper, 'load_model'):
                print(f"Error: whisper module does not have 'load_model' attribute")
                print(f"Available attributes: {[attr for attr in dir(whisper) if not attr.startswith('_')]}")
                return False
            
            # Use the exact same pattern as test_pipeline.py
            self.whisper_model = whisper.load_model("base", device=device)
            print("Whisper loaded successfully")
            
            print("Loading TTS model...")
            self.tts = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                gpu=torch.cuda.is_available(),
                progress_bar=False
            )
            print("TTS loaded successfully")
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def text_to_speech(self, text):
        """Convert text to speech using TTS"""
        try:
            if not self.tts:
                return None
            
            # Generate audio (using same method as test_pipeline.py)
            wav = self.tts.tts(text=text)
            
            # Convert to numpy array if needed
            if isinstance(wav, list):
                wav = np.array(wav)
            
            return wav
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def generate_response(self, user_input):
        """Generate response using Ollama LLM"""
        try:
            prompt = f"""You are a helpful AI assistant. Please provide a concise, helpful response to the user's question.

User: {user_input}

Assistant:"""
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I'm sorry, I couldn't generate a response.")
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {e}"
    
    def process_request(self, user_input):
        """Process user input and return response with audio"""
        try:
            # Generate response
            response = self.generate_response(user_input)
            
            # Generate audio
            audio_data = self.text_to_speech(response)
            
            if audio_data is not None:
                # Save audio to temporary file
                timestamp = int(time.time())
                filename = f"response_{timestamp}.wav"
                filepath = os.path.join(tempfile.gettempdir(), filename)
                sf.write(filepath, audio_data, 22050)
                
                return {
                    "success": True,
                    "response": response,
                    "audio_file": filename,
                    "filepath": filepath
                }
            else:
                return {
                    "success": False,
                    "response": response,
                    "error": "Failed to generate audio"
                }
                
        except Exception as e:
            return {
                "success": False,
                "response": f"Error: {e}",
                "error": str(e)
            }

# Global voicebot instance
voicebot = WebVoiceBot()

@app.route('/')
def index():
    return render_template('voicebot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"success": False, "error": "Empty message"})
        
        # Process the request
        result = voicebot.process_request(user_input)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/audio/<filename>')
def get_audio(filename):
    """Serve audio files"""
    try:
        filepath = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/api/status')
def status():
    """Check system status"""
    try:
        # Check if models are loaded
        models_loaded = voicebot.whisper_model is not None and voicebot.tts is not None
        
        # Check Ollama connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_running = response.status_code == 200
        except:
            ollama_running = False
        
        return jsonify({
            "models_loaded": models_loaded,
            "ollama_running": ollama_running,
            "status": "ready" if models_loaded and ollama_running else "initializing"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/health')
def health():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Web VoiceBot is running"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main function"""
    print("Starting Web VoiceBot Demo...")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Ollama is not running. Please start it first:")
            print("   ollama serve")
            return
        print("Ollama connection verified")
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        print("Please start it first: ollama serve")
        return
    
    # Load models
    print("Loading AI models...")
    if not voicebot.load_models():
        print("Failed to load AI models. Exiting.")
        return
    
    print("Models loaded successfully!")
    print("Starting web server...")
    print("="*60)
    print("🌐 Web Interface URL: http://10.2.9.10:5000")
    print("🎤 VoiceBot is ready for chat!")
    print("="*60)
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down web server...")
    except Exception as e:
        print(f"Error starting web server: {e}")

if __name__ == "__main__":
    main()
