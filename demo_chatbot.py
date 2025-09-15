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
    import whisper
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
    print("Missing dependency: {}".format(e))
    print("Please install: pip install whisper torch TTS requests soundfile numpy webrtcvad pyaudio python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

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
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("Whisper loaded")
            
            print("Loading TTS model...")
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            print("TTS loaded")
            
            return True
        except Exception as e:
            print("Error loading models: {}".format(e))
            return False
    
    def text_to_speech(self, text):
        """Convert text to speech using TTS"""
        try:
            if not self.tts:
                return None
            
            # Generate audio
            audio_data = self.tts.tts(text)
            
            # Convert to numpy array
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
        except Exception as e:
            print("TTS Error: {}".format(e))
            return None
    
    def play_audio(self, audio_data):
        """Play audio through speakers"""
        try:
            if audio_data is None:
                return False
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Play audio
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=22050,  # TTS sample rate
                output=True
            )
            
            stream.write(audio_int16.tobytes())
            stream.stop_stream()
            stream.close()
            
            return True
        except Exception as e:
            print("Audio playback error: {}".format(e))
            return False
    
    def generate_response(self, user_input):
        """Generate response using Ollama LLM"""
        try:
            prompt = (
                "You are a helpful AI assistant. Please provide a concise, helpful response to the user's question.\n\n"
                "User: {}\n\n"
                "Assistant:".format(user_input)
            )
            
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
                return "Error: {}".format(response.status_code)
                
        except Exception as e:
            return "Error generating response: {}".format(e)
    
    def speak_text(self, text):
        """Convert text to speech and play it"""
        print("Speaking: {}".format(text))
        
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
                print("AI: {}".format(response))
                
        except KeyboardInterrupt:
            print("\n\nDemo ended. Goodbye!")
        except Exception as e:
            print("\nError: {}".format(e))
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
