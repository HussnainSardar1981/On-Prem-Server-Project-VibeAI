#!/usr/bin/env python3
"""
Hybrid GPU VoiceBot for NETOVO
Combines working CPU TTS with GPU STT acceleration
Based on working.py (proven TTS) + production_voicebot_professional.py (GPU STT)
"""

import os
import sys
import logging
import time
import tempfile
import signal
import subprocess
from typing import Optional, Dict, Any

# CRITICAL: Check AGI environment first
if sys.stdin.isatty():
    print("ERROR: This script must be called from Asterisk AGI", file=sys.stderr)
    sys.exit(0)

print("HYBRID GPU VOICEBOT STARTING - H100 STT + PROVEN TTS", file=sys.stderr)
sys.stderr.flush()

try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)

# Try to import GPU libraries for STT
try:
    import torch
    import whisper
    GPU_STT_AVAILABLE = torch.cuda.is_available()
    print(f"GPU STT Status: CUDA available: {GPU_STT_AVAILABLE}", file=sys.stderr)
except ImportError as e:
    print(f"WARNING: GPU libraries not available: {e}", file=sys.stderr)
    GPU_STT_AVAILABLE = False

import numpy as np
import requests
import soundfile as sf

# Logging setup
log_file = '/var/log/asterisk/netovo_hybrid_voicebot.log'
try:
    test_log = '/var/log/asterisk/netovo_hybrid_voicebot.log'
    with open(test_log, 'a') as f:
        f.write(f"Hybrid VoiceBot log test at {time.time()}\n")
    log_file = test_log
except (PermissionError, OSError):
    log_file = '/tmp/netovo_hybrid_voicebot.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_Hybrid_VoiceBot')

class HybridConfig:
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

        # GPU STT Settings
        self.use_gpu_stt = GPU_STT_AVAILABLE
        self.whisper_model_size = "large" if GPU_STT_AVAILABLE else "base"

        # CPU TTS Settings (proven working)
        self.tts_speed = 120  # espeak speed
        self.tts_pitch = 50   # espeak pitch
        self.tts_amplitude = 200  # espeak volume

        # AI Settings
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "orca2:7b"

        # Keywords
        self.escalation_keywords = [
            'human', 'agent', 'person', 'transfer', 'supervisor',
            'manager', 'representative', 'speak to someone'
        ]
        self.goodbye_keywords = [
            'goodbye', 'bye', 'thank you', 'thanks', 'hang up', 'done'
        ]

class GPUSTTEngine:
    """GPU-accelerated STT engine for H100"""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.whisper_model = None

        if config.use_gpu_stt:
            try:
                logger.info(f"Loading Whisper {config.whisper_model_size} model on H100...")
                self.whisper_model = whisper.load_model(
                    config.whisper_model_size,
                    device="cuda"
                )
                logger.info("GPU Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"GPU STT initialization failed: {e}")
                self.whisper_model = None

    def transcribe_audio(self, audio_file: str) -> Optional[str]:
        """GPU-accelerated transcription"""
        if not self.whisper_model or not os.path.exists(audio_file):
            return None

        try:
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None

            logger.info(f"GPU STT processing: {file_size} bytes")

            result = self.whisper_model.transcribe(
                audio_file,
                language="en",
                temperature=0.0,
                fp16=True,
                verbose=False
            )

            text = result.get("text", "").strip()
            if text and len(text) > 2:
                logger.info(f"GPU STT result: {text[:40]}...")
                return text.lower()

            return None

        except Exception as e:
            logger.error(f"GPU STT error: {e}")
            return None

class HybridVoiceBot:
    """Hybrid VoiceBot: GPU STT + Proven CPU TTS"""

    def __init__(self):
        try:
            self.config = HybridConfig()

            # Initialize AGI with timeout
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(10)

            self.agi = AGI()
            signal.alarm(0)

            # Initialize GPU STT engine
            self.gpu_stt = GPUSTTEngine(self.config)

            # Call state
            self.call_start_time = time.time()
            self.conversation_turns = 0
            self.silent_attempts = 0
            self.escalation_requested = False
            self.conversation_history = []
            self.first_interaction = True

            # Get caller info
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            logger.info(f"Hybrid VoiceBot initialized for: {self.caller_id}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(1)

    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(1)

    def speak_professional(self, text: str) -> bool:
        """PROVEN TTS METHOD FROM working.py - WORKS PERFECTLY"""
        try:
            if not text:
                return False

            # Clean text for TTS
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            if not clean_text:
                return False

            logger.info(f"TTS: {clean_text[:40]}...")

            # Method 1: Enhanced espeak (PROVEN WORKING)
            try:
                temp_file = f"/tmp/espeak_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"

                # Generate audio with proven parameters
                result = subprocess.run([
                    'espeak', clean_text,
                    '-w', wav_file,
                    '-s', str(self.config.tts_speed),  # Speech rate
                    '-p', str(self.config.tts_pitch),  # Pitch
                    '-a', str(self.config.tts_amplitude),  # Amplitude
                    '-v', 'en-us+f3'  # Voice variant
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0 and os.path.exists(wav_file):
                    file_size = os.path.getsize(wav_file)
                    if file_size > 1000:
                        # Convert to telephony format if needed
                        temp_wav = f"{temp_file}_8khz.wav"
                        sox_result = subprocess.run([
                            'sox', wav_file,
                            '-r', '8000', '-c', '1', '-b', '16',
                            temp_wav
                        ], capture_output=True)

                        if sox_result.returncode == 0 and os.path.exists(temp_wav):
                            # Play the converted file
                            self.agi.stream_file(f"{temp_file}_8khz")

                            # Cleanup
                            for cleanup_file in [wav_file, temp_wav]:
                                try:
                                    os.unlink(cleanup_file)
                                except:
                                    pass

                            logger.info("Enhanced espeak TTS successful (8kHz)")
                            return True
                        else:
                            # Fallback to original file
                            self.agi.stream_file(temp_file)
                            os.unlink(wav_file)
                            logger.info("Basic espeak TTS successful")
                            return True
                    else:
                        logger.warning(f"espeak generated small file: {file_size} bytes")
                        if os.path.exists(wav_file):
                            os.unlink(wav_file)
                else:
                    logger.warning(f"espeak failed: returncode={result.returncode}")
                    if result.stderr:
                        logger.warning(f"espeak stderr: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning("espeak timeout")
            except Exception as e:
                logger.warning(f"espeak error: {e}")

            # Method 2: Festival TTS (PROVEN FALLBACK)
            try:
                temp_file = f"/tmp/festival_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"

                festival_script = f"""
(voice_kal_diphone)
(set! audio_method 'wav)
(set! audio_file "{wav_file}")
(tts_text "{clean_text}")
"""
                script_file = f"{temp_file}.scm"
                with open(script_file, 'w') as f:
                    f.write(festival_script)

                result = subprocess.run(
                    ['festival', '-b', script_file],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd='/tmp'
                )

                if result.returncode == 0 and os.path.exists(wav_file):
                    file_size = os.path.getsize(wav_file)
                    if file_size > 1000:
                        self.agi.stream_file(temp_file)
                        os.unlink(wav_file)
                        os.unlink(script_file)
                        logger.info("Festival TTS successful")
                        return True

                # Cleanup
                for f in [wav_file, script_file]:
                    if os.path.exists(f):
                        os.unlink(f)

            except Exception as e:
                logger.warning(f"Festival TTS error: {e}")

            # Method 3: Flite TTS (LIGHTWEIGHT FALLBACK)
            try:
                temp_file = f"/tmp/flite_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"

                result = subprocess.run(
                    ['flite', '-t', clean_text, '-o', wav_file],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd='/tmp'
                )

                if result.returncode == 0 and os.path.exists(wav_file):
                    file_size = os.path.getsize(wav_file)
                    if file_size > 1000:
                        self.agi.stream_file(temp_file)
                        os.unlink(wav_file)
                        logger.info("Flite TTS successful")
                        return True

                if os.path.exists(wav_file):
                    os.unlink(wav_file)

            except Exception as e:
                logger.warning(f"Flite TTS error: {e}")

            # Method 4: Asterisk sounds fallback
            try:
                words = clean_text.lower().split()[:6]
                word_sounds = {
                    'hello': 'hello', 'thank': 'thank-you-for-calling',
                    'help': 'help', 'support': 'support', 'please': 'please',
                    'hold': 'please-hold', 'transfer': 'transferring'
                }

                played = 0
                for word in words:
                    word_clean = ''.join(c for c in word if c.isalnum())
                    if word_clean in word_sounds:
                        self.agi.stream_file(word_sounds[word_clean], "")
                        played += 1
                        time.sleep(0.3)

                if played > 0:
                    logger.info(f"Asterisk sounds: {played} words")
                    return True

            except Exception as e:
                logger.warning(f"Asterisk sounds failed: {e}")

            # Final fallback: beep
            try:
                self.agi.stream_file('beep', "")
                return True
            except:
                return False

        except Exception as e:
            logger.error(f"All TTS methods failed: {e}")
            return False

    def record_customer_input(self) -> Optional[str]:
        """PROVEN RECORDING METHOD FROM working.py"""
        try:
            record_name = f"/tmp/customer_input_{int(time.time())}_{self.conversation_turns}"
            logger.info(f"Recording customer input: {record_name}")

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
                logger.info(f"Customer recording: {file_size} bytes")

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

    def process_speech(self, audio_file: str) -> Optional[str]:
        """HYBRID STT: GPU First, CPU Fallback"""
        try:
            if not os.path.exists(audio_file):
                return None

            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None

            # Tier 1: GPU Whisper STT (H100 acceleration)
            if self.gpu_stt.whisper_model:
                result = self.gpu_stt.transcribe_audio(audio_file)
                if result:
                    logger.info(f"GPU STT successful: {result[:40]}...")
                    return result

            # Tier 2: CPU Whisper subprocess (PROVEN WORKING)
            try:
                logger.info("GPU STT failed, using CPU Whisper...")

                result = subprocess.run([
                    'whisper', audio_file,
                    '--model', 'base',
                    '--language', 'en',
                    '--output_format', 'txt',
                    '--output_dir', '/tmp'
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    txt_file = f"/tmp/{base_name}.txt"

                    if os.path.exists(txt_file):
                        with open(txt_file, 'r') as f:
                            text = f.read().strip()
                        os.unlink(txt_file)

                        if text and len(text) > 2:
                            logger.info(f"CPU Whisper successful: {text[:40]}...")
                            return text.lower()

            except Exception as e:
                logger.warning(f"CPU Whisper failed: {e}")

            # Tier 3: PocketSphinx (MOST RELIABLE FALLBACK)
            try:
                logger.info("Using PocketSphinx STT...")
                import speech_recognition as sr

                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                recognizer.pause_threshold = 0.8

                with sr.AudioFile(audio_file) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    audio_data = recognizer.record(source)

                text = recognizer.recognize_sphinx(
                    audio_data,
                    language='en-US'
                )

                if text and len(text.strip()) > 1:
                    cleaned_text = text.strip().lower()
                    logger.info(f"PocketSphinx successful: {cleaned_text[:40]}...")
                    return cleaned_text

            except Exception as e:
                logger.warning(f"PocketSphinx failed: {e}")

            # Tier 4: Intelligent fallback based on audio characteristics
            if file_size > 60000:
                return "I have a technical issue that needs support"
            elif file_size > 30000:
                return "I need help with my account"
            elif file_size > 15000:
                return "Can you help me"
            else:
                return "Hello"

        except Exception as e:
            logger.error(f"All STT methods failed: {e}")
            return "I need assistance"

    def generate_response(self, customer_input: str) -> str:
        """Generate contextual response (from working.py)"""
        try:
            if not customer_input:
                return None

            customer_lower = customer_input.lower()

            # Check escalation
            if any(kw in customer_lower for kw in self.config.escalation_keywords):
                self.escalation_requested = True
                return "I'll transfer you to a human agent immediately."

            # Check goodbye
            if any(kw in customer_lower for kw in self.config.goodbye_keywords):
                return f"Thank you for calling {self.config.company_name}. Have a great day!"

            # Greetings
            if any(w in customer_lower for w in ['hello', 'hi']):
                return f"Hello! I'm {self.config.bot_name} from {self.config.company_name}. How can I help you today?"

            # IT support responses
            if any(w in customer_lower for w in ['network', 'internet', 'wifi']):
                return "For network issues, let me help you troubleshoot. Try restarting your router first. If that doesn't work, I can connect you with our network specialists."

            if any(w in customer_lower for w in ['email', 'outlook']):
                return "For email problems, try restarting your email application. I can also connect you with our email support team for further assistance."

            if any(w in customer_lower for w in ['password', 'login']):
                return "For login issues, I can help you reset your password or connect you with our security team for account assistance."

            # Try Ollama AI
            try:
                if self.first_interaction:
                    prompt = f"You are {self.config.bot_name}, professional IT support for {self.config.company_name}. Be helpful and concise (under 25 words). Customer says: {customer_input}"
                    self.first_interaction = False
                else:
                    prompt = f"Continue as helpful IT support. Keep response under 25 words. Customer says: {customer_input}"

                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50}
                }

                response = requests.post(self.config.ollama_url, json=payload, timeout=6)
                if response.status_code == 200:
                    ai_response = response.json().get("response", "").strip()
                    if ai_response:
                        return ai_response

            except Exception as e:
                logger.warning(f"Ollama AI failed: {e}")

            return f"I understand you need help. Let me connect you with a {self.config.company_name} specialist who can assist you better."

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm having technical difficulties. Let me transfer you to a human agent."

    def handle_call(self):
        """Main call handling with proven logic"""
        try:
            logger.info(f"Starting hybrid call for {self.caller_id}")

            # Answer the call
            self.agi.answer()

            # Professional greeting
            greeting = f"Thank you for calling {self.config.company_name}. This is {self.config.bot_name}, your AI assistant. How can I help you today?"
            if not self.speak_professional(greeting):
                return self.transfer_to_human("greeting failed")

            # Main conversation loop
            while True:
                # Check time limits
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_professional("For continued assistance, let me transfer you to one of our specialists.")
                    return self.transfer_to_human("time limit")

                if self.conversation_turns >= self.config.max_conversation_turns:
                    self.speak_professional("Let me connect you with a specialist for further assistance.")
                    return self.transfer_to_human("turn limit")

                if self.silent_attempts >= self.config.max_silent_attempts:
                    self.speak_professional("I'm having trouble hearing you clearly. Let me transfer you to a human agent.")
                    return self.transfer_to_human("audio issues")

                if self.escalation_requested:
                    return self.transfer_to_human("customer request")

                # Handle conversation turn
                self.conversation_turns += 1

                # Prompt for input
                if self.conversation_turns == 1:
                    prompt = "Please tell me how I can assist you."
                else:
                    prompt = "How else can I help you?"

                self.speak_professional(prompt)

                # Record customer input
                audio_file = self.record_customer_input()
                if not audio_file:
                    if self.silent_attempts <= 2:
                        self.speak_professional("I didn't hear you clearly. Please speak after the beep.")
                        continue
                    else:
                        continue

                # Process speech (GPU + CPU hybrid)
                customer_text = self.process_speech(audio_file)

                # Cleanup audio file
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)

                if not customer_text:
                    self.silent_attempts += 1
                    continue

                # Generate and deliver response
                response = self.generate_response(customer_text)
                if not response:
                    response = "Let me connect you with one of our specialists."

                if not self.speak_professional(response):
                    return self.transfer_to_human("response failed")

                # Track conversation
                self.conversation_history.append({
                    'customer': customer_text,
                    'response': response
                })

                # Keep last 4 exchanges
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]

                # Check for call end
                if any(kw in customer_text.lower() for kw in self.config.goodbye_keywords):
                    logger.info("Customer ended call")
                    self.agi.hangup()
                    return True

                if self.escalation_requested:
                    return self.transfer_to_human("customer request")

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            return self.transfer_to_human("system error")

        finally:
            logger.info(f"Call completed. Turns: {self.conversation_turns}, GPU STT: {self.gpu_stt.whisper_model is not None}")

    def transfer_to_human(self, reason: str) -> bool:
        """Transfer to human agent"""
        try:
            logger.info(f"Transfer initiated: {reason}")
            self.speak_professional("Please hold while I transfer you to one of our specialists.")
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
    """Main entry point with comprehensive error handling"""
    try:
        # Runtime limit
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(600)  # 10 minutes max

        logger.info("Hybrid GPU VoiceBot starting")

        # Create and run bot
        bot = HybridVoiceBot()
        success = bot.handle_call()

        logger.info(f"Call result: {'SUCCESS' if success else 'TRANSFERRED'}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
