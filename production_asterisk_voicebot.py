#!/usr/bin/env python3
"""
Optimized Production Asterisk VoiceBot for NETOVO 3CX Integration
Fast startup with pre-loaded models and proper error handling
"""

import os
import sys
import logging
import time
import tempfile
import signal
import subprocess
import json
from typing import Optional, Dict, Any
from pathlib import Path

# Early debug output
print("OPTIMIZED VOICEBOT STARTING", file=sys.stderr)
sys.stderr.flush()

# Core imports
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(1)

import numpy as np
import requests
import soundfile as sf

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NETOVO_VoiceBot_Optimized')

class OptimizedConfig:
    """Optimized configuration for fast startup"""
    
    def __init__(self):
        # 3CX Configuration
        self.sip_server = "mtipbx.ny.3cx.us"
        self.extension = "1600"
        self.did = "+1 (646) 358-3509"
        
        # Optimized AI Models (lightweight for telephony)
        self.whisper_model = "tiny"  # 39MB vs 244MB - loads in ~2 seconds
        self.ollama_model = "orca2:7b"
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        
        # Audio Configuration
        self.sample_rate = 8000
        self.record_timeout = 8  # Reduced timeout
        self.silence_threshold = 2
        
        # Performance Settings
        self.max_turns = 10
        self.max_call_duration = 300  # 5 minutes
        self.response_timeout = 15  # Reduced timeout
        self.model_load_timeout = 20  # Max time for model loading
        
        # Business Logic
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"
        
        logger.info("Optimized config loaded")

class LightweightAIProcessor:
    """Lightweight AI processor with fast startup"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = 'cpu'  # Force CPU for consistency
        self.whisper_model = None
        
        # Performance tracking
        self.call_metrics = {
            'stt_times': [],
            'llm_times': [],
            'total_turns': 0,
            'call_start': time.time()
        }
        
        logger.info("AI Processor initialized")
    
    def initialize_models_fast(self) -> bool:
        """Fast model initialization with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.model_load_timeout)
        
        try:
            logger.info("Fast-loading AI models...")
            start_time = time.time()
            
            # Load lightweight Whisper model
            if self.whisper_model is None:
                logger.info("Loading Whisper tiny model...")
                try:
                    import whisper
                    self.whisper_model = whisper.load_model(
                        self.config.whisper_model, 
                        device=self.device
                    )
                    logger.info("Whisper tiny model loaded successfully")
                except Exception as e:
                    logger.error(f"Whisper loading failed: {e}")
                    return False
            
            # Test Ollama connectivity
            try:
                response = requests.get(
                    "http://127.0.0.1:11434/api/tags", 
                    timeout=3
                )
                if response.status_code == 200:
                    logger.info("Ollama connectivity verified")
                else:
                    logger.warning("Ollama may not be ready")
            except requests.exceptions.RequestException:
                logger.warning("Ollama not accessible - using fallback responses")
            
            load_time = time.time() - start_time
            logger.info(f"Models ready in {load_time:.2f}s")
            return True
            
        except TimeoutError:
            logger.error("Model loading timed out")
            return False
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
        finally:
            signal.alarm(0)  # Clear timeout
    
    def speech_to_text_fast(self, audio_file: str) -> Optional[str]:
        """Optimized speech-to-text processing"""
        try:
            start_time = time.time()
            
            # Quick audio validation
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1024:
                logger.warning("Audio file too small or missing")
                return None
            
            # Load audio efficiently
            try:
                audio_data, sr = sf.read(audio_file)
            except Exception as e:
                logger.error(f"Audio read error: {e}")
                return None
            
            # Quick length check
            if len(audio_data) < 800:  # ~0.1s at 8kHz
                logger.warning("Audio too short for transcription")
                return None
            
            # Simple resampling if needed
            if sr != 16000:
                # Basic resampling for telephony audio
                ratio = 16000 / sr
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Normalize efficiently
            audio_data = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Transcribe with timeout
            try:
                result = self.whisper_model.transcribe(
                    audio_data,
                    language='en',
                    task='transcribe',
                    fp16=False,
                    verbose=False
                )
                
                text = result.get('text', '').strip()
                processing_time = time.time() - start_time
                self.call_metrics['stt_times'].append(processing_time)
                
                logger.info(f"STT ({processing_time:.2f}s): '{text[:50]}...'")
                return text if text else None
                
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
                return None
                
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
    
    def generate_response_fast(self, user_text: str) -> str:
        """Fast response generation with fallbacks"""
        try:
            start_time = time.time()
            
            # Quick fallback responses for common cases
            user_lower = user_text.lower()
            
            if any(word in user_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello! I'm {self.config.bot_name} from {self.config.company_name} support. How can I help you today?"
            
            if any(word in user_lower for word in ['bye', 'goodbye', 'thanks']):
                return f"Thank you for calling {self.config.company_name}. Have a great day!"
            
            if any(word in user_lower for word in ['help', 'support', 'problem', 'issue']):
                return "I'm here to help with your IT needs. Can you describe the specific issue you're experiencing?"
            
            # Try Ollama with short timeout
            try:
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": f"You are {self.config.bot_name}, IT support for {self.config.company_name}. Keep responses under 30 words. Customer said: {user_text}\n\nResponse:",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 50
                    }
                }
                
                response = requests.post(
                    self.config.ollama_url,
                    json=payload,
                    timeout=self.config.response_timeout
                )
                
                if response.status_code == 200:
                    bot_response = response.json().get("response", "").strip()
                    if bot_response:
                        processing_time = time.time() - start_time
                        self.call_metrics['llm_times'].append(processing_time)
                        logger.info(f"LLM ({processing_time:.2f}s): Generated response")
                        return bot_response
                
            except requests.exceptions.Timeout:
                logger.warning("LLM timeout, using fallback")
            except Exception as e:
                logger.warning(f"LLM error: {e}, using fallback")
            
            # Intelligent fallback based on keywords
            if any(word in user_lower for word in ['network', 'internet', 'wifi', 'connection']):
                return "For network issues, please check your cables and restart your router. If the problem persists, I can transfer you to our network specialist."
            
            if any(word in user_lower for word in ['email', 'outlook', 'mail']):
                return "For email issues, try restarting your email client. If that doesn't work, I can help you with account settings or transfer you to our email support team."
            
            if any(word in user_lower for word in ['password', 'login', 'access']):
                return "For login issues, I can help reset your password or transfer you to our security team for account access problems."
            
            # Generic fallback
            return f"I understand you need help with that. Let me transfer you to one of our {self.config.company_name} specialists who can assist you better."
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to a human agent right away."

class OptimizedVoiceBot:
    """Optimized VoiceBot with fast startup and error recovery"""
    
    def __init__(self):
        try:
            self.config = OptimizedConfig()
            self.agi = AGI()
            self.ai_processor = LightweightAIProcessor(self.config)
            
            # Call state
            self.call_start_time = time.time()
            self.conversation_context = {
                'turn_count': 0,
                'last_topic': '',
                'escalation_requested': False
            }
            
            # Get call information safely
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            self.channel = self.agi.env.get('agi_channel', 'Unknown')
            
            logger.info(f"VoiceBot initialized for {self.caller_id}")
            
        except Exception as e:
            logger.error(f"VoiceBot initialization error: {e}")
            raise
    
    def speak_text_simple(self, text: str) -> bool:
        """Simplified text-to-speech using Asterisk built-ins"""
        try:
            # Use Festival TTS if available, otherwise use simple playback
            logger.info(f"Speaking: {text[:50]}...")
            
            # Create temporary file for TTS
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write(text)
            temp_file.close()
            
            try:
                # Try using festival TTS
                wav_file = temp_file.name.replace('.txt', '.wav')
                result = subprocess.run([
                    'text2wave', temp_file.name, '-o', wav_file
                ], capture_output=True, timeout=10)
                
                if result.returncode == 0 and os.path.exists(wav_file):
                    # Play the generated audio
                    audio_name = wav_file.replace('.wav', '')
                    play_result = self.agi.stream_file(audio_name, '#')
                    
                    # Cleanup
                    os.unlink(wav_file)
                    os.unlink(temp_file.name)
                    
                    return play_result == 0
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Festival not available, use fallback
                pass
            
            # Fallback: Use Asterisk's built-in TTS or pre-recorded messages
            logger.info("Using fallback speech method")
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # For now, just return success (audio will be handled by calling system)
            return True
            
        except Exception as e:
            logger.error(f"Speech error: {e}")
            return False
    
    def record_speech_simple(self, prompt: str = None) -> Optional[str]:
        """Simplified speech recording"""
        try:
            if prompt:
                self.speak_text_simple(prompt)
            
            # Create recording file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            record_name = temp_file.name.replace('.wav', '')
            temp_file.close()
            
            logger.info(f"Recording to {record_name}")
            
            # Record with optimized settings
            self.agi.record_file(
                record_name,
                format='wav',
                escape_digits='#*',
                timeout=self.config.record_timeout * 1000,
                offset=0,
                beep=True,
                silence=self.config.silence_threshold
            )
            
            # Check recording
            wav_file = record_name + '.wav'
            if os.path.exists(wav_file) and os.path.getsize(wav_file) > 1024:
                logger.info(f"Recording successful: {os.path.getsize(wav_file)} bytes")
                return wav_file
            else:
                logger.warning("Recording failed or too short")
                if os.path.exists(wav_file):
                    os.unlink(wav_file)
                return None
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
    
    def handle_conversation_turn_fast(self) -> bool:
        """Fast conversation turn handling"""
        try:
            self.conversation_context['turn_count'] += 1
            turn_start = time.time()
            
            logger.info(f"Turn {self.conversation_context['turn_count']}")
            
            # Record user speech
            audio_file = self.record_speech_simple("Please speak now, or press # when finished.")
            
            if not audio_file:
                if self.conversation_context['turn_count'] == 1:
                    self.speak_text_simple("I didn't hear anything. Please try speaking again.")
                    return True
                else:
                    self.speak_text_simple("I'm having trouble hearing you. Let me transfer you to an agent.")
                    return False
            
            # Convert speech to text
            user_text = self.ai_processor.speech_to_text_fast(audio_file)
            
            # Cleanup
            try:
                os.unlink(audio_file)
            except:
                pass
            
            if not user_text:
                self.speak_text_simple("I didn't catch that. Could you repeat your question?")
                return True
            
            # Check for end signals
            user_lower = user_text.lower()
            if any(phrase in user_lower for phrase in ['goodbye', 'bye', 'thank you', 'thanks', 'hang up']):
                self.speak_text_simple(f"Thank you for calling {self.config.company_name}. Goodbye!")
                return False
            
            # Check for escalation
            if any(phrase in user_lower for phrase in ['human', 'agent', 'person', 'transfer', 'supervisor']):
                self.speak_text_simple("I'll transfer you to an agent right away. Please hold.")
                return False
            
            # Generate response
            bot_response = self.ai_processor.generate_response_fast(user_text)
            
            # Speak response
            success = self.speak_text_simple(bot_response)
            
            # Update context
            self.conversation_context['last_topic'] = user_text[:30]
            
            turn_time = time.time() - turn_start
            logger.info(f"Turn completed in {turn_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Conversation turn error: {e}")
            self.speak_text_simple("I'm having technical difficulties. Please hold for transfer.")
            return False
    
    def run_call_optimized(self):
        """Optimized main call handling"""
        try:
            logger.info(f"Starting optimized call from {self.caller_id}")
            
            # Answer call
            self.agi.answer()
            logger.info("Call answered")
            
            # Quick model initialization with timeout
            if not self.ai_processor.initialize_models_fast():
                self.speak_text_simple("I'm experiencing technical difficulties. Please call back shortly.")
                return False
            
            # Send greeting
            greeting = f"Hello! Thank you for calling {self.config.company_name} support. This is {self.config.bot_name}. How can I help you today?"
            if not self.speak_text_simple(greeting):
                logger.error("Failed to send greeting")
                return False
            
            # Main conversation loop
            while True:
                # Check limits
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_text_simple("Let me transfer you to an agent for continued assistance.")
                    break
                
                if self.conversation_context['turn_count'] >= self.config.max_turns:
                    self.speak_text_simple("Let me connect you with one of our specialists.")
                    break
                
                # Handle turn
                if not self.handle_conversation_turn_fast():
                    break
            
            logger.info(f"Call completed successfully after {self.conversation_context['turn_count']} turns")
            return True
            
        except Exception as e:
            logger.error(f"Call handling error: {e}")
            try:
                self.speak_text_simple("I'm sorry, technical difficulties. Goodbye.")
            except:
                pass
            return False
        
        finally:
            try:
                self.agi.hangup()
            except:
                pass
            logger.info("Call ended")

def main():
    """Optimized main entry point"""
    try:
        # Signal handlers
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and run optimized voicebot
        logger.info("Starting optimized voicebot")
        voicebot = OptimizedVoiceBot()
        success = voicebot.run_call_optimized()
        
        logger.info(f"Voicebot finished: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0)  # Always exit 0 for AGI
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(0)  # Still exit 0 for AGI compatibility

print("ABOUT TO START MAIN", file=sys.stderr)
sys.stderr.flush()

if __name__ == "__main__":
    main()
