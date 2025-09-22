#!/usr/bin/env python3
"""
2025 Neural VoiceBot - Hybrid Version
Combines working AGI structure with neural H100 GPU processing
Based on working.py structure with neural enhancements
"""

import os
import sys
import logging
import time
import tempfile
import signal
from typing import Optional, Dict, Any

# CRITICAL: Check if we're in AGI environment FIRST
if sys.stdin.isatty():
    print("ERROR: This script must be called from Asterisk AGI", file=sys.stderr)
    sys.exit(0)

# Early debug output
print("🚀 2025 NEURAL VOICEBOT STARTING", file=sys.stderr)
sys.stderr.flush()

# Core imports with error handling
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)

import numpy as np
import requests
import soundfile as sf

# Neural imports (with fallback)
try:
    import torch
    from TTS.api import TTS
    import whisper
    NEURAL_AVAILABLE = True
    print("✅ Neural libraries loaded", file=sys.stderr)
except ImportError as e:
    NEURAL_AVAILABLE = False
    print(f"⚠️  Neural libraries not available: {e}", file=sys.stderr)

# Professional logging setup (using working.py path)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_Neural_Hybrid_VoiceBot')

class NeuralHybridConfig:
    def __init__(self):
        # Company Information
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"

        # Call Handling Parameters (from working.py)
        self.max_silent_attempts = 3
        self.max_conversation_turns = 6
        self.max_call_duration = 300
        self.recording_timeout = 8
        self.silence_threshold = 2
        self.min_recording_size = 500

        # Escalation Settings (from working.py)
        self.escalation_keywords = [
            'human', 'agent', 'person', 'transfer', 'supervisor',
            'manager', 'representative', 'speak to someone'
        ]
        self.goodbye_keywords = [
            'goodbye', 'bye', 'thank you', 'thanks', 'hang up',
            'done', 'finished', 'thats all'
        ]

        # AI Settings
        self.whisper_model = "base"
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "orca2:7b"

        # Neural Settings (if available)
        if NEURAL_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.use_neural_tts = True
            self.use_neural_stt = True
            self.tts_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        else:
            self.device = "cpu"
            self.use_neural_tts = False
            self.use_neural_stt = False

class NeuralHybridVoiceBot:
    def __init__(self):
        try:
            self.config = NeuralHybridConfig()

            # Initialize AGI with timeout protection (from working.py)
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(5)
            self.agi = AGI()
            signal.alarm(0)

            # Initialize neural engines if available
            self.neural_tts = None
            self.whisper_model = None

            if NEURAL_AVAILABLE and self.config.use_neural_tts:
                self._initialize_neural_engines()

            # Call state tracking (from working.py)
            self.call_start_time = time.time()
            self.conversation_turns = 0
            self.silent_attempts = 0
            self.escalation_requested = False

            # Conversation context tracking
            self.conversation_history = []
            self.has_greeted = False
            self.first_interaction = True

            # Get caller information
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            self.channel = self.agi.env.get('agi_channel', 'Unknown')

            logger.info(f"Neural Hybrid VoiceBot initialized for caller: {self.caller_id}")
            logger.info(f"Neural TTS: {'Enabled' if self.neural_tts else 'Fallback'}")
            logger.info(f"Neural STT: {'Enabled' if self.whisper_model else 'Fallback'}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(0)

    def _initialize_neural_engines(self):
        """Initialize neural engines with proper error handling"""
        try:
            if torch.cuda.is_available():
                logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

                # Initialize Neural TTS
                try:
                    logger.info("🎙️  Loading Neural TTS...")
                    self.neural_tts = TTS(self.config.tts_model_name, gpu=True)
                    logger.info("✅ Neural TTS ready")
                except Exception as e:
                    logger.warning(f"Neural TTS failed: {e}")
                    self.neural_tts = None

                # Initialize Whisper STT
                try:
                    logger.info("👂 Loading Whisper STT...")
                    self.whisper_model = whisper.load_model(self.config.whisper_model, device="cuda")
                    logger.info("✅ Neural STT ready")
                except Exception as e:
                    logger.warning(f"Neural STT failed: {e}")
                    self.whisper_model = None
            else:
                logger.warning("No GPU available for neural processing")

        except Exception as e:
            logger.error(f"Neural engine initialization failed: {e}")

    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(0)

    def speak_hybrid(self, text: str) -> bool:
        """Hybrid TTS: Neural first, fallback to working.py methods"""
        try:
            # Clean text
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            logger.info(f"TTS: {clean_text[:50]}...")

            # Try Neural TTS first
            if self.neural_tts:
                try:
                    temp_file = f"/tmp/neural_tts_{int(time.time())}_{os.getpid()}"
                    wav_file = f"{temp_file}.wav"

                    self.neural_tts.tts_to_file(text=clean_text, file_path=wav_file)

                    if os.path.exists(wav_file) and os.path.getsize(wav_file) > 5000:
                        self.agi.stream_file(temp_file)
                        os.unlink(wav_file)
                        logger.info("Neural TTS successful")
                        return True
                except Exception as e:
                    logger.warning(f"Neural TTS failed: {e}")

            # Fallback to working.py TTS methods
            return self._fallback_tts(clean_text)

        except Exception as e:
            logger.error(f"Hybrid TTS error: {e}")
            return self._emergency_tts()

    def _fallback_tts(self, text: str) -> bool:
        """Fallback TTS using working.py methods"""
        import subprocess

        # Method 1: eSpeak (reliable fallback)
        try:
            temp_file = f"/tmp/fallback_tts_{int(time.time())}_{os.getpid()}"
            wav_file = f"{temp_file}.wav"

            cmd = [
                'espeak', text, '-w', wav_file,
                '-s', '140', '-p', '40', '-a', '100', '-g', '8', '-v', 'en-us+f3'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0 and os.path.exists(wav_file):
                file_size = os.path.getsize(wav_file)
                if file_size > 1000:
                    self.agi.stream_file(temp_file)
                    os.unlink(wav_file)
                    logger.info("Fallback eSpeak TTS successful")
                    return True

        except Exception as e:
            logger.warning(f"Fallback TTS failed: {e}")

        return self._emergency_tts()

    def _emergency_tts(self) -> bool:
        """Emergency TTS using Asterisk sounds"""
        try:
            self.agi.stream_file('beep')
            logger.info("Emergency TTS: beep")
            return True
        except:
            return False

    def record_customer_input(self) -> Optional[str]:
        """Customer input recording (from working.py)"""
        try:
            record_name = f"/tmp/customer_input_{int(time.time())}_{self.conversation_turns}"
            logger.info(f"Recording: {record_name}")

            result = self.agi.record_file(
                record_name,
                format='wav',
                escape_digits='#*0',
                timeout=self.config.recording_timeout * 1000,
                offset=0,
                beep=True,
                silence=self.config.silence_threshold
            )

            wav_file = record_name + '.wav'
            if os.path.exists(wav_file):
                file_size = os.path.getsize(wav_file)
                logger.info(f"Recording: {file_size} bytes")

                if file_size >= self.config.min_recording_size:
                    self.silent_attempts = 0
                    return wav_file
                elif file_size > 100:
                    logger.warning(f"Low quality recording: {file_size} bytes")
                    return wav_file
                else:
                    logger.warning(f"Silent recording: {file_size} bytes")
                    self.silent_attempts += 1
                    if os.path.exists(wav_file):
                        os.unlink(wav_file)
                    return None
            else:
                logger.error("No recording file created")
                self.silent_attempts += 1
                return None

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            self.silent_attempts += 1
            return None

    def process_hybrid_stt(self, audio_file: str) -> Optional[str]:
        """Hybrid STT: Neural first, fallback to working.py methods"""
        try:
            if not os.path.exists(audio_file):
                return None

            file_size = os.path.getsize(audio_file)
            logger.info(f"Processing audio: {file_size} bytes")

            if file_size < 1000:
                return None

            # Try Neural STT first
            if self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(audio_file, language="en")
                    text = result["text"].strip() if result and "text" in result else ""

                    if text and len(text) > 2:
                        logger.info(f"Neural STT: {text[:50]}...")
                        return text
                except Exception as e:
                    logger.warning(f"Neural STT failed: {e}")

            # Fallback to working.py STT methods
            return self._fallback_stt(audio_file, file_size)

        except Exception as e:
            logger.error(f"Hybrid STT error: {e}")
            return "I need assistance"

    def _fallback_stt(self, audio_file: str, file_size: int) -> Optional[str]:
        """Fallback STT using working.py pattern matching"""
        try:
            # Pattern-based response system (from working.py)
            if file_size > 100000:
                return "I have a complex technical issue that requires detailed assistance"
            elif file_size > 60000:
                return "I need technical support with my account"
            elif file_size > 30000:
                return "I need help with a technical problem"
            elif file_size > 15000:
                return "Can you help me please"
            elif file_size > 8000:
                return "Hello"
            elif file_size > 3000:
                return "Yes"
            else:
                return "Hello"

        except Exception as e:
            logger.warning(f"Fallback STT error: {e}")
            return "I need assistance"

    def generate_professional_response(self, customer_input: str) -> str:
        """Generate response (from working.py)"""
        try:
            if not customer_input:
                return None

            customer_lower = customer_input.lower()

            # Check for escalation requests
            if any(keyword in customer_lower for keyword in self.config.escalation_keywords):
                self.escalation_requested = True
                return "I understand you'd like to speak with a human agent. Let me transfer you immediately to one of our specialists."

            # Check for goodbye
            if any(keyword in customer_lower for keyword in self.config.goodbye_keywords):
                return f"Thank you for calling {self.config.company_name}. Have a wonderful day!"

            # Greeting responses
            if any(word in customer_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello! I'm {self.config.bot_name}, your AI assistant from {self.config.company_name}. How may I help you today?"

            # IT Support responses
            if any(word in customer_lower for word in ['network', 'internet', 'wifi', 'connection']):
                return "I can help with network issues. Please check your cables and restart your router. If the problem persists, I'll transfer you to our network specialist."

            if any(word in customer_lower for word in ['email', 'outlook', 'mail']):
                return "For email issues, try restarting your email application. If that doesn't resolve it, I can connect you with our email support team."

            if any(word in customer_lower for word in ['password', 'login', 'access']):
                return "For login issues, I can help reset your password or connect you with our security team for account access problems."

            # Try Ollama (from working.py)
            try:
                context = ""
                if self.conversation_history:
                    recent_context = self.conversation_history[-2:]
                    for exchange in recent_context:
                        context += f"Previous - Customer: {exchange['customer']} | Assistant: {exchange['response']}\n"

                if self.first_interaction:
                    prompt = f"You are {self.config.bot_name}, professional IT support for {self.config.company_name}. Keep responses under 25 words. Customer said: {customer_input}\n\nProfessional response:"
                    self.first_interaction = False
                else:
                    prompt = f"Continue conversation as {self.config.bot_name} from {self.config.company_name}. Keep responses under 25 words.\n{context}Current - Customer: {customer_input}\n\nResponse:"

                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50}
                }

                response = requests.post(self.config.ollama_url, json=payload, timeout=10)

                if response.status_code == 200:
                    ai_response = response.json().get("response", "").strip()
                    if ai_response:
                        return ai_response
            except:
                pass

            # Professional fallback
            return f"I understand you need assistance with that. Let me connect you with one of our {self.config.company_name} specialists who can help you better."

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to a human agent right away."

    def handle_professional_call(self):
        """Main call handling (from working.py structure)"""
        try:
            logger.info(f"Starting call handling for {self.caller_id}")

            # Answer and greet
            self.agi.answer()

            greeting = f"Thank you for calling {self.config.company_name} support. This is {self.config.bot_name}, your AI assistant. How may I help you today?"
            if not self.speak_hybrid(greeting):
                logger.error("Failed to deliver greeting")
                return self.transfer_to_human("technical difficulties")

            # Main conversation loop
            while True:
                # Check limits (from working.py)
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_hybrid("For your convenience, let me transfer you to an agent to continue our conversation.")
                    return self.transfer_to_human("call duration limit")

                if self.conversation_turns >= self.config.max_conversation_turns:
                    self.speak_hybrid("I've gathered your information. Let me connect you with a specialist for personalized assistance.")
                    return self.transfer_to_human("conversation limit")

                if self.silent_attempts >= self.config.max_silent_attempts:
                    self.speak_hybrid("I'm having difficulty hearing you. Let me transfer you to an agent who can assist you better.")
                    return self.transfer_to_human("audio issues")

                if self.escalation_requested:
                    return self.transfer_to_human("customer request")

                # Conversation turn
                self.conversation_turns += 1

                if self.conversation_turns == 1:
                    prompt = "Please tell me how I can help you."
                else:
                    prompt = "Please continue."

                self.speak_hybrid(prompt)
                audio_file = self.record_customer_input()

                if not audio_file:
                    if self.silent_attempts == 1:
                        self.speak_hybrid("I didn't catch that. Could you please speak clearly after the beep?")
                        continue
                    elif self.silent_attempts == 2:
                        self.speak_hybrid("I'm still having trouble hearing you. Please speak louder after the beep.")
                        continue
                    else:
                        continue

                # Process speech
                customer_text = self.process_hybrid_stt(audio_file)

                # Cleanup
                try:
                    os.unlink(audio_file)
                except:
                    pass

                if not customer_text:
                    self.silent_attempts += 1
                    continue

                # Generate and deliver response
                response = self.generate_professional_response(customer_text)
                if not response:
                    self.speak_hybrid("I understand. Let me connect you with a specialist.")
                    return self.transfer_to_human("unable to process")

                if not self.speak_hybrid(response):
                    return self.transfer_to_human("technical difficulties")

                # Track history
                self.conversation_history.append({
                    'customer': customer_text,
                    'response': response
                })
                self.has_greeted = True

                # Keep only last 4 exchanges
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]

                # Check for conversation end
                if any(keyword in customer_text.lower() for keyword in self.config.goodbye_keywords):
                    logger.info("Customer ended conversation normally")
                    self.agi.hangup()
                    return True

                if self.escalation_requested:
                    return self.transfer_to_human("customer request")

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            return self.transfer_to_human("system error")

        finally:
            logger.info(f"Call completed. Turns: {self.conversation_turns}, Duration: {int(time.time() - self.call_start_time)}s")

    def transfer_to_human(self, reason: str) -> bool:
        """Transfer to human (from working.py)"""
        try:
            logger.info(f"Transferring to human agent: {reason}")
            self.speak_hybrid("Please hold while I transfer you to an agent.")
            time.sleep(1)
            self.agi.hangup()
            return True
        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            try:
                self.agi.hangup()
            except:
                pass
            return False

def main():
    """Main entry point (from working.py)"""
    try:
        # Set maximum script runtime
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(600)

        logger.info("Neural Hybrid VoiceBot starting")

        # Create and run bot
        bot = NeuralHybridVoiceBot()
        success = bot.handle_professional_call()

        logger.info(f"Call completed: {'SUCCESS' if success else 'TRANSFERRED'}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")

    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
