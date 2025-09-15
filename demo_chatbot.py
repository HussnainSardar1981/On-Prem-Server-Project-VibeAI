#!/usr/bin/env python3
"""
Interactive VoiceBot Demo for Meeting
Uses existing AI pipeline: STT -> LLM (Orca2) -> TTS
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    # Import OpenAI Whisper specifically (not the Graphite whisper package)
    import openai_whisper as whisper
    print(f"OpenAI Whisper imported successfully, version: {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
    import torch
    from TTS.api import TTS
    import requests
    import soundfile as sf
    import numpy as np
    import webrtcvad
    import pyaudio
    import wave
    import io
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install openai-whisper torch TTS requests soundfile numpy webrtcvad pyaudio python-dotenv")
    print("Note: Make sure to install 'openai-whisper' not 'whisper' to avoid conflicts with Graphite whisper")
    sys.exit(1)

# Load environment variables (optional)
try:
    load_dotenv()
except:
    pass  # Continue without .env file

class VoiceBotDemo:
    def __init__(self):
        print("Initializing VoiceBot Demo...")
        
        # Initialize AI components
        self.whisper_model = None
        self.tts = None
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "orca2:7b"
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # VAD settings
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Demo mode flag
        self.demo_mode = True
        
        print("VoiceBot Demo initialized successfully!")
    
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
    
    def play_audio(self, audio_data):
        """Play audio through speakers"""
        try:
            if audio_data is None:
                return False
            
            # Convert to 16-bit PCM (same as test_pipeline.py)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Play audio with TTS sample rate
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=22050,  # TTS sample rate from test_pipeline.py
                output=True
            )
            
            stream.write(audio_int16.tobytes())
            stream.stop_stream()
            stream.close()
            
            return True
        except Exception as e:
            print(f"Audio playback error: {e}")
            return False
    
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
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        print(f"Speaking: {text}")
        
        # Generate audio
        audio_data = self.text_to_speech(text)
        if audio_data is not None:
            # Play audio
            self.play_audio(audio_data)
        else:
            print("Failed to generate speech")
    
    def run_demo(self):
        """Run the interactive demo"""
        print("\n" + "="*60)
        print("VOICEBOT DEMO - Interactive Chat")
        print("="*60)
        print("Type your questions and press Enter")
        print("The AI will respond with voice")
        print("Press Ctrl+C to exit")
        print("="*60)
        
        # Load models
        if not self.load_models():
            print("Failed to load AI models. Exiting.")
            return
        
        print("\nReady! Ask me anything...")
        
        try:
            while True:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Generate response
                print("AI: Thinking...")
                response = self.generate_response(user_input)
                
                # Speak response
                self.speak_text(response)
                
                # Print response for reference
                print(f"AI: {response}")
                
        except KeyboardInterrupt:
            print("\n\nDemo ended. Goodbye!")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'audio'):
                self.audio.terminate()
        except:
            pass

def main():
    """Main function"""
    print("Starting VoiceBot Demo...")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Ollama is not running. Please start it first:")
            print("   ollama serve")
            return
    except:
        print("Cannot connect to Ollama. Please start it first:")
        print("   ollama serve")
        return
    
    # Create and run demo
    demo = VoiceBotDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
