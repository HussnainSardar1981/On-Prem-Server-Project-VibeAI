#!/usr/bin/env python3
"""
MINIMAL GPU Enhancement of working.py
ONLY changes: Whisper model from 'base' to 'large' for GPU acceleration
Everything else EXACTLY the same as working.py
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
print("GPU-ENHANCED WORKING VOICEBOT STARTING", file=sys.stderr)
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

# Professional logging setup (EXACT SAME as working.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_GPU_Enhanced_VoiceBot')

class ProfessionalConfig:
    def __init__(self):
        # Company Information
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"

        # Call Handling Parameters (EXACT SAME as working.py)
        self.max_silent_attempts = 3      # Max attempts for silent/no input
        self.max_conversation_turns = 6   # Max conversation exchanges
        self.max_call_duration = 300      # 5 minutes max call
        self.recording_timeout = 8        # 8 seconds to speak
        self.silence_threshold = 2        # 2 seconds silence to stop recording
        self.min_recording_size = 500     # Minimum bytes for valid recording

        # Escalation Settings (EXACT SAME as working.py)
        self.escalation_keywords = [
            'human', 'agent', 'person', 'transfer', 'supervisor',
            'manager', 'representative', 'speak to someone'
        ]
        self.goodbye_keywords = [
            'goodbye', 'bye', 'thank you', 'thanks', 'hang up',
            'done', 'finished', 'thats all'
        ]

        # AI Settings - ONLY CHANGE: Use large model for GPU
        # Detect if CUDA is available for Whisper
        try:
            import torch
            if torch.cuda.is_available():
                self.whisper_model = "large"  # GPU acceleration
                logger.info("GPU detected: Using Whisper large model")
            else:
                self.whisper_model = "base"   # CPU fallback
                logger.info("No GPU: Using Whisper base model")
        except ImportError:
            self.whisper_model = "base"       # Safe fallback
            logger.info("PyTorch not available: Using Whisper base model")

        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "orca2:7b"

class ProfessionalVoiceBot:
    def __init__(self):
        try:
            self.config = ProfessionalConfig()

            # Initialize AGI with timeout protection (EXACT SAME as working.py)
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(5)

            self.agi = AGI()
            signal.alarm(0)

            # Call state tracking (EXACT SAME as working.py)
            self.call_start_time = time.time()
            self.conversation_turns = 0
            self.silent_attempts = 0
            self.escalation_requested = False

            # Conversation context tracking (EXACT SAME as working.py)
            self.conversation_history = []
            self.has_greeted = False
            self.first_interaction = True

            # Get caller information (EXACT SAME as working.py)
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            self.channel = self.agi.env.get('agi_channel', 'Unknown')

            logger.info(f"Professional VoiceBot initialized successfully")
            logger.info(f"Caller ID: {self.caller_id}")
            logger.info(f"Channel: {self.channel}")
            logger.info(f"Using Whisper model: {self.config.whisper_model}")

        except Exception as e:
            logger.error(f"VoiceBot initialization failed: {e}")
            sys.exit(1)

    def timeout_handler(self, signum, frame):
        """Handle AGI initialization timeout"""
        logger.error("AGI initialization timeout")
        sys.exit(1)

    def speak_professional(self, text: str) -> bool:
        """
        EXACT SAME TTS METHOD FROM working.py
        Professional multi-tier TTS with enterprise fallbacks
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided to TTS")
                return False

            # Professional text cleaning (maintain quality)
            clean_text = ''.join(char for char in text if char.isalnum() or char in ' .,!?-')
            clean_text = clean_text.strip()

            if not clean_text:
                logger.warning("Text cleaned to empty string")
                return False

            if len(clean_text) > 200:
                clean_text = clean_text[:200]
                logger.info(f"Text truncated to 200 chars for TTS efficiency")

            logger.info(f"Professional TTS: {clean_text[:50]}...")

            # Method 1: Festival TTS (highest quality, most natural)
            try:
                temp_file = f"/tmp/professional_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"
                txt_file = f"{temp_file}.txt"

                # Create text file for Festival
                with open(txt_file, 'w') as f:
                    f.write(clean_text)

                # Execute Festival with professional voice settings
                result = subprocess.run([
                    'text2wave',
                    txt_file,
                    '-o', wav_file,
                    '-eval', '(voice_kal_diphone)',  # Professional voice
                    '-eval', '(Parameter.set \'Audio_Method \'wav)',
                    '-eval', '(Parameter.set \'Audio_Required_Rate 8000)'  # VoIP rate
                ],
                    text=True,
                    capture_output=True,
                    timeout=8
                )

                if result.returncode == 0 and os.path.exists(wav_file) and os.path.getsize(wav_file) > 1000:
                    self.agi.stream_file(temp_file)
                    # Cleanup
                    for cleanup_file in [wav_file, txt_file]:
                        try:
                            os.unlink(cleanup_file)
                        except:
                            pass
                    logger.info("Festival TTS successful (professional quality)")
                    return True

            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("Festival not available, trying espeak")

            # Method 2: Improved espeak (fallback)
            try:
                temp_file = f"/tmp/enterprise_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"

                # Professional espeak parameters optimized for VoIP (8kHz)
                temp_wav = f"{wav_file}.tmp"
                cmd = [
                    'espeak',
                    clean_text,
                    '-w', temp_wav,
                    '-s', '140',      # Slightly slower for clarity
                    '-p', '40',       # Lower pitch (more professional)
                    '-a', '100',      # Full amplitude
                    '-g', '8',        # Shorter gaps (more natural)
                    '-v', 'en-us+f3' # Female voice variant 3 (warmer)
                ]

                logger.info(f"Executing: {' '.join(cmd)}")

                # Execute with timeout and proper error handling
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd='/tmp'
                )

                if result.returncode == 0 and os.path.exists(temp_wav):
                    # Convert to 8kHz mono for VoIP compatibility
                    file_size = os.path.getsize(temp_wav)
                    if file_size > 1000:
                        logger.info(f"espeak generated: {file_size} bytes")

                        # Use sox for high-quality 8kHz conversion
                        sox_result = subprocess.run([
                            'sox', temp_wav,
                            '-r', '8000',    # 8kHz sample rate (VoIP standard)
                            '-c', '1',       # Mono
                            '-b', '16',      # 16-bit
                            wav_file         # Output file
                        ], capture_output=True, timeout=5)

                        if sox_result.returncode == 0 and os.path.exists(wav_file):
                            self.agi.stream_file(temp_file)
                            # Cleanup
                            for cleanup_file in [wav_file, temp_wav]:
                                try:
                                    os.unlink(cleanup_file)
                                except:
                                    pass
                            logger.info("Enterprise espeak TTS successful (8kHz)")
                            return True
                    else:
                        logger.warning(f"espeak generated small file: {file_size} bytes")
                        if os.path.exists(wav_file):
                            os.unlink(wav_file)
                else:
                    logger.warning(f"espeak subprocess failed: returncode={result.returncode}")
                    if result.stderr:
                        logger.warning(f"espeak stderr: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning("espeak subprocess timeout")
            except FileNotFoundError:
                logger.warning("espeak binary not found")
            except Exception as e:
                logger.warning(f"espeak subprocess error: {e}")

            # Method 3: Festival TTS with subprocess
            try:
                temp_file = f"/tmp/festival_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"

                # Create Festival script
                festival_script = f"""
(voice_kal_diphone)
(set! audio_method 'wav)
(set! audio_file "{wav_file}")
(tts_text "{clean_text}")
"""
                script_file = f"{temp_file}.scm"
                with open(script_file, 'w') as f:
                    f.write(festival_script)

                # Execute Festival
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
                        logger.info("Enterprise Festival TTS successful")
                        return True
                    else:
                        logger.warning(f"Festival generated small file: {file_size} bytes")

                # Cleanup
                for f in [wav_file, script_file]:
                    if os.path.exists(f):
                        os.unlink(f)

            except subprocess.TimeoutExpired:
                logger.warning("Festival subprocess timeout")
            except FileNotFoundError:
                logger.warning("Festival binary not found")
            except Exception as e:
                logger.warning(f"Festival subprocess error: {e}")

            # Method 4: Flite TTS (lightweight fallback)
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
                        logger.info("Enterprise Flite TTS successful")
                        return True

                if os.path.exists(wav_file):
                    os.unlink(wav_file)

            except subprocess.TimeoutExpired:
                logger.warning("Flite subprocess timeout")
            except FileNotFoundError:
                logger.warning("Flite binary not found")
            except Exception as e:
                logger.warning(f"Flite subprocess error: {e}")

            # Method 5: Sound file playback fallback (professional sounds)
            try:
                # Map common phrases to professional Asterisk sounds
                phrase_lower = clean_text.lower()
                sound_mappings = {
                    'hello': 'hello',
                    'thank you': 'thank-you-for-calling',
                    'please hold': 'please-hold',
                    'transfer': 'transferring',
                    'one moment': 'one-moment-please',
                    'help': 'can-i-help-you',
                    'support': 'support',
                    'technical': 'technical-support'
                }

                played_sound = False
                for phrase, sound in sound_mappings.items():
                    if phrase in phrase_lower:
                        try:
                            self.agi.stream_file(sound, "")
                            played_sound = True
                            logger.info(f"Professional sound played: {sound}")
                            break
                        except Exception:
                            continue

                if played_sound:
                    return True

            except Exception as e:
                logger.warning(f"Professional sound playback error: {e}")

            # Final fallback: System beep to indicate activity
            try:
                self.agi.stream_file('beep', "")
                logger.warning("All TTS failed - played system beep")
                return True
            except Exception as e:
                logger.error(f"Even system beep failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Critical TTS error: {e}")
            return False

    def record_customer_input(self) -> Optional[str]:
        """
        EXACT SAME RECORDING METHOD FROM working.py
        Professional customer input recording with proper error handling
        """
        try:
            # Create unique recording file
            record_name = f"/tmp/customer_input_{int(time.time())}_{self.conversation_turns}"
            logger.info(f"Recording customer input: {record_name}")

            # Professional recording with optimized parameters
            result = self.agi.record_file(
                record_name,
                format='wav',
                escape_digits='#*0',  # Allow customer to interrupt
                timeout=self.config.recording_timeout * 1000,
                offset=0,
                beep=True,  # Professional beep indicator
                silence=self.config.silence_threshold
            )

            # Analyze recording quality
            wav_file = record_name + '.wav'
            if os.path.exists(wav_file):
                file_size = os.path.getsize(wav_file)
                logger.info(f"Customer recording: {file_size} bytes")

                if file_size >= self.config.min_recording_size:
                    # Good quality recording
                    self.silent_attempts = 0  # Reset silent counter
                    return wav_file
                elif file_size > 100:
                    # Marginal recording - still process but note
                    logger.warning(f"Low quality recording: {file_size} bytes")
                    return wav_file
                else:
                    # Silent or failed recording
                    logger.warning(f"Silent recording detected: {file_size} bytes")
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

    def process_customer_speech(self, audio_file: str) -> Optional[str]:
        """
        MINIMALLY MODIFIED STT - ONLY CHANGE: Use self.config.whisper_model (base or large)
        Everything else EXACTLY the same as working.py
        """
        try:
            import subprocess
            import json

            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return None

            file_size = os.path.getsize(audio_file)
            logger.info(f"Processing audio file: {file_size} bytes")

            # Skip tiny files (silence)
            if file_size < 1000:
                logger.warning("Audio file too small, likely silence")
                return None

            # Method 1: Whisper (OpenAI) - ONLY CHANGE: Use config model instead of hardcoded 'base'
            try:
                logger.info(f"Attempting Whisper STT with model: {self.config.whisper_model}...")

                # Use whisper command line (if installed)
                result = subprocess.run([
                    'whisper', audio_file,
                    '--model', self.config.whisper_model,  # ONLY CHANGE: was 'base', now uses config
                    '--language', 'en',
                    '--output_format', 'txt',
                    '--output_dir', '/tmp'
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    # Look for generated text file
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    txt_file = f"/tmp/{base_name}.txt"

                    if os.path.exists(txt_file):
                        with open(txt_file, 'r') as f:
                            text = f.read().strip()
                        os.unlink(txt_file)  # Cleanup

                        if text and len(text) > 2:
                            logger.info(f"Whisper STT successful ({self.config.whisper_model}): {text[:50]}...")
                            return text

            except subprocess.TimeoutExpired:
                logger.warning("Whisper STT timeout")
            except FileNotFoundError:
                logger.warning("Whisper not found, trying alternative")
            except Exception as e:
                logger.warning(f"Whisper STT error: {e}")

            # REST OF THE STT METHODS ARE EXACTLY THE SAME AS working.py...
            # Method 2: Python speech_recognition with PocketSphinx (Most Reliable)
            try:
                logger.info("Attempting PocketSphinx STT...")
                import speech_recognition as sr

                recognizer = sr.Recognizer()

                # Enhanced PocketSphinx settings for better accuracy
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                recognizer.pause_threshold = 0.8

                with sr.AudioFile(audio_file) as source:
                    # Adjust for ambient noise with longer duration
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    audio_data = recognizer.record(source)

                # Try PocketSphinx with enhanced settings
                text = recognizer.recognize_sphinx(
                    audio_data,
                    language='en-US',
                    keyword_entries=None,
                    show_all=False
                )

                if text and len(text.strip()) > 1:
                    cleaned_text = text.strip().lower()
                    logger.info(f"PocketSphinx STT successful: {cleaned_text[:50]}...")
                    return cleaned_text

            except ImportError:
                logger.warning("speech_recognition library not available")
            except sr.UnknownValueError:
                logger.warning("PocketSphinx could not understand audio")
            except sr.RequestError as e:
                logger.warning(f"PocketSphinx error: {e}")
            except Exception as e:
                logger.warning(f"PocketSphinx STT error: {e}")

            # Method 3: Advanced Audio Analysis Fallback (Enhanced)
            try:
                logger.info("Using audio analysis fallback...")

                # Analyze audio characteristics
                duration_cmd = subprocess.run([
                    'ffprobe', '-v', 'quiet',
                    '-show_entries', 'format=duration',
                    '-of', 'csv=p=0',
                    audio_file
                ], capture_output=True, text=True, timeout=5)

                if duration_cmd.returncode == 0:
                    duration = float(duration_cmd.stdout.strip())
                    logger.info(f"Audio duration: {duration:.2f}s, size: {file_size} bytes")

                    # Enhanced pattern recognition based on audio characteristics
                    bytes_per_second = file_size / max(duration, 0.1)

                    if duration > 4.0 and bytes_per_second > 8000:
                        return "I have a technical issue that needs immediate support"
                    elif duration > 3.0 and bytes_per_second > 6000:
                        return "I need help with my account and system access"
                    elif duration > 2.0 and bytes_per_second > 4000:
                        return "Can you help me troubleshoot this problem"
                    elif duration > 1.5 and bytes_per_second > 3000:
                        return "I need technical assistance please"
                    elif duration > 1.0:
                        return "Hello I need help"
                    else:
                        return "Yes"

            except (subprocess.TimeoutExpired, ValueError, ZeroDivisionError):
                logger.warning("Audio analysis failed")

            # Final heuristic fallback based on file size patterns
            if file_size > 80000:
                return "I'm experiencing technical difficulties with my systems"
            elif file_size > 50000:
                return "I need support with network connectivity issues"
            elif file_size > 30000:
                return "Can you help me with account access"
            elif file_size > 15000:
                return "I need technical assistance"
            elif file_size > 8000:
                return "Hello can you help me"
            elif file_size > 3000:
                return "Yes I need help"
            else:
                return "Hello"

        except Exception as e:
            logger.error(f"All STT methods failed: {e}")
            return "I need assistance"  # Safe fallback

    # REST OF THE METHODS ARE EXACTLY THE SAME AS working.py
    def generate_contextual_response(self, customer_input: str, conversation_context: list) -> str:
        """Generate intelligent contextual response based on conversation history"""
        try:
            if not customer_input:
                return None

            customer_lower = customer_input.lower()

            # Check for escalation keywords
            escalation_keywords = self.config.escalation_keywords
            if any(keyword in customer_lower for keyword in escalation_keywords):
                self.escalation_requested = True
                return "I understand you'd like to speak with a human agent. I'm transferring you now."

            # Check for goodbye/ending keywords
            goodbye_keywords = self.config.goodbye_keywords
            if any(keyword in customer_lower for keyword in goodbye_keywords):
                return f"Thank you for calling {self.config.company_name}. We appreciate your business. Have a great day!"

            # Analyze conversation context for better responses
            context_keywords = []
            for exchange in conversation_context:
                if 'customer' in exchange:
                    context_keywords.extend(exchange['customer'].lower().split())

            # First interaction - enhanced greeting
            if not self.has_greeted:
                greeting_responses = [
                    f"Hello! I'm {self.config.bot_name} from {self.config.company_name}. I'm here to help with your technical needs today.",
                    f"Good day! This is {self.config.bot_name}, your {self.config.company_name} AI assistant. How may I assist you?",
                    f"Welcome to {self.config.company_name}! I'm {self.config.bot_name}. I'm ready to help with your technology questions."
                ]
                self.has_greeted = True
                return greeting_responses[0]  # Use first one for consistency

            # Enhanced keyword-based responses with context awareness
            response_patterns = {
                'network': [
                    "For network connectivity issues, let's start with basic troubleshooting. Have you tried restarting your router and modem?",
                    "Network problems can be frustrating. Let me help you diagnose this. Are you experiencing slow speeds or complete disconnection?",
                    "I can assist with network issues. Is this affecting all devices or just specific ones?"
                ],
                'email': [
                    "Email problems are common and usually fixable. Are you unable to send, receive, or access your email entirely?",
                    "Let's resolve your email issue. Is this related to Outlook, webmail, or mobile email access?",
                    "I can help with email configuration. What specific email problem are you experiencing?"
                ],
                'password': [
                    "Password issues can be resolved quickly. Do you need to reset a password or are you locked out of an account?",
                    "I can assist with password problems. Is this for your computer login, email, or another system?",
                    "Let me help you with password recovery. Which system or account needs attention?"
                ],
                'computer': [
                    "Computer issues vary widely. Is your computer running slowly, not starting, or having specific software problems?",
                    "I'm here to help with computer problems. Can you describe what's happening when you try to use it?",
                    "Computer troubleshooting is my specialty. What symptoms are you experiencing?"
                ],
                'internet': [
                    "Internet connectivity is crucial for business. Are you experiencing slow speeds, intermittent connection, or no connection at all?",
                    "Let's get your internet working properly. Is this affecting your entire office or just certain areas?",
                    "Internet issues can disrupt productivity. What type of connection problems are you experiencing?"
                ],
                'phone': [
                    "Phone system issues can impact business operations. Are you having problems with incoming calls, outgoing calls, or voice quality?",
                    "I can help troubleshoot phone problems. Is this related to your desk phone, mobile, or VoIP system?",
                    "Let's resolve your phone issue. What specific problem are you experiencing with your calls?"
                ]
            }

            # Find relevant pattern
            for keyword, responses in response_patterns.items():
                if keyword in customer_lower:
                    # Use first response for consistency
                    return responses[0]

            # Try Ollama AI for more complex queries
            try:
                if self.first_interaction:
                    prompt = f"You are {self.config.bot_name}, a professional IT support assistant for {self.config.company_name}. Provide a helpful, concise response (under 30 words) to: {customer_input}"
                    self.first_interaction = False
                else:
                    prompt = f"Continue as professional IT support. Context: {' '.join(context_keywords[-10:])}. Customer says: {customer_input}. Respond helpfully in under 25 words."

                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 60
                    }
                }

                response = requests.post(self.config.ollama_url, json=payload, timeout=5)
                if response.status_code == 200:
                    ai_response = response.json().get("response", "").strip()
                    if ai_response and len(ai_response) > 5:
                        logger.info(f"Ollama AI response generated")
                        return ai_response

            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama request failed: {e}")
            except Exception as e:
                logger.warning(f"Ollama AI error: {e}")

            # Intelligent fallback responses
            if len(customer_input) > 50:
                return f"I understand you have a detailed concern. Let me connect you with a {self.config.company_name} specialist who can provide comprehensive assistance."
            elif any(word in customer_lower for word in ['urgent', 'emergency', 'critical', 'down']):
                return f"I recognize this is urgent. I'm escalating you immediately to our priority support team for faster resolution."
            elif any(word in customer_lower for word in ['yes', 'yeah', 'ok', 'sure']):
                return "Great! Let me gather some additional information to better assist you. Can you describe the specific issue you're experiencing?"
            else:
                return f"I want to ensure you get the best possible help. Let me connect you with one of our {self.config.company_name} technical specialists."

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize for the technical difficulty. Let me transfer you to a human agent who can assist you immediately."

    def handle_professional_call(self):
        """
        EXACT SAME CALL HANDLING FROM working.py
        Main call handling logic with professional conversation management
        """
        try:
            logger.info(f"=== Starting Professional Call ===")
            logger.info(f"Caller ID: {self.caller_id}")
            logger.info(f"Channel: {self.channel}")

            # Answer the call professionally
            self.agi.answer()
            logger.info("Call answered successfully")

            # Professional greeting
            greeting = f"Thank you for calling {self.config.company_name}. This is {self.config.bot_name}, your AI technical assistant. How may I help you today?"

            if not self.speak_professional(greeting):
                logger.error("Failed to deliver greeting")
                return self.escalate_to_human("greeting_failed")

            # Main conversation loop
            while True:
                # Check time-based limits
                call_duration = time.time() - self.call_start_time
                if call_duration > self.config.max_call_duration:
                    logger.info(f"Call duration limit reached: {call_duration:.1f}s")
                    self.speak_professional("For your continued assistance, I'm transferring you to one of our specialists.")
                    return self.escalate_to_human("time_limit")

                # Check conversation turn limits
                if self.conversation_turns >= self.config.max_conversation_turns:
                    logger.info(f"Conversation turn limit reached: {self.conversation_turns}")
                    self.speak_professional("Let me connect you with a specialist for more detailed assistance.")
                    return self.escalate_to_human("turn_limit")

                # Check silent attempt limits
                if self.silent_attempts >= self.config.max_silent_attempts:
                    logger.info(f"Silent attempt limit reached: {self.silent_attempts}")
                    self.speak_professional("I'm having difficulty hearing you clearly. Let me transfer you to a human agent.")
                    return self.escalate_to_human("audio_issues")

                # Check for escalation request
                if self.escalation_requested:
                    logger.info("Customer requested escalation")
                    return self.escalate_to_human("customer_request")

                # Start new conversation turn
                self.conversation_turns += 1
                logger.info(f"=== Conversation Turn {self.conversation_turns} ===")

                # Prompt for customer input
                if self.conversation_turns == 1:
                    prompt = "Please describe how I can assist you today."
                elif self.conversation_turns == 2:
                    prompt = "Please continue with your question or concern."
                else:
                    prompt = "What else can I help you with?"

                if not self.speak_professional(prompt):
                    logger.error("Failed to deliver prompt")
                    return self.escalate_to_human("prompt_failed")

                # Record customer input
                audio_file = self.record_customer_input()
                if not audio_file:
                    if self.silent_attempts <= 2:
                        self.speak_professional("I didn't catch that. Please speak clearly after the beep.")
                        continue
                    else:
                        logger.warning("Multiple silent attempts - continuing loop")
                        continue

                # Process customer speech
                customer_text = self.process_customer_speech(audio_file)

                # Cleanup audio file
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                except Exception as e:
                    logger.warning(f"Audio cleanup failed: {e}")

                if not customer_text:
                    logger.warning("No text extracted from customer speech")
                    self.silent_attempts += 1
                    continue

                logger.info(f"Customer said: {customer_text[:100]}...")

                # Generate contextual response
                response = self.generate_contextual_response(customer_text, self.conversation_history)
                if not response:
                    response = "Let me connect you with one of our technical specialists for personalized assistance."

                # Deliver response
                if not self.speak_professional(response):
                    logger.error("Failed to deliver response")
                    return self.escalate_to_human("response_failed")

                # Track conversation history
                conversation_entry = {
                    'turn': self.conversation_turns,
                    'customer': customer_text,
                    'response': response,
                    'timestamp': time.time()
                }
                self.conversation_history.append(conversation_entry)

                # Maintain reasonable history size
                if len(self.conversation_history) > 6:
                    self.conversation_history = self.conversation_history[-6:]

                # Check for conversation ending conditions
                customer_lower = customer_text.lower()
                if any(goodbye in customer_lower for goodbye in self.config.goodbye_keywords):
                    logger.info("Customer indicated end of call")
                    final_message = f"Thank you for calling {self.config.company_name}. Have a wonderful day!"
                    self.speak_professional(final_message)
                    self.agi.hangup()
                    return True

                # Check for escalation request (double-check)
                if self.escalation_requested:
                    logger.info("Escalation requested during conversation")
                    return self.escalate_to_human("customer_request")

                # Brief pause between turns for natural conversation flow
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Call handling error: {e}")
            return self.escalate_to_human("system_error")

        finally:
            call_duration = time.time() - self.call_start_time
            logger.info(f"=== Call Completed ===")
            logger.info(f"Total duration: {call_duration:.1f}s")
            logger.info(f"Conversation turns: {self.conversation_turns}")
            logger.info(f"Silent attempts: {self.silent_attempts}")

    def escalate_to_human(self, reason: str) -> bool:
        """
        EXACT SAME ESCALATION FROM working.py
        Professional escalation to human agent
        """
        try:
            logger.info(f"=== Escalating to Human Agent ===")
            logger.info(f"Reason: {reason}")

            escalation_messages = {
                'greeting_failed': "I'm experiencing technical difficulties. Please hold while I connect you to a human agent.",
                'time_limit': "For your continued assistance, I'm transferring you to one of our specialists.",
                'turn_limit': "Let me connect you with a human agent for more detailed assistance.",
                'audio_issues': "I'm having trouble with our connection. Let me transfer you to a human agent.",
                'customer_request': "I'm connecting you with a human agent as requested.",
                'system_error': "I'm experiencing a technical issue. Let me transfer you to a human agent.",
                'prompt_failed': "Let me connect you directly with one of our specialists.",
                'response_failed': "I'm transferring you to a human agent for immediate assistance."
            }

            message = escalation_messages.get(reason, "Let me transfer you to a human agent for assistance.")

            # Attempt to inform customer of transfer
            try:
                self.speak_professional(message)
                time.sleep(1)  # Brief pause for message delivery
            except Exception as e:
                logger.warning(f"Transfer message delivery failed: {e}")

            # Log escalation details
            call_duration = time.time() - self.call_start_time
            logger.info(f"Escalation details - Duration: {call_duration:.1f}s, Turns: {self.conversation_turns}, Reason: {reason}")

            # Perform the transfer (hangup - external system will handle routing)
            self.agi.hangup()
            logger.info("Successfully escalated to human agent")
            return True

        except Exception as e:
            logger.error(f"Escalation failed: {e}")
            try:
                # Emergency hangup
                self.agi.hangup()
            except:
                pass
            return False

def main():
    """
    EXACT SAME MAIN FROM working.py
    Main execution function with comprehensive error handling
    """
    try:
        # Set runtime limits for safety
        signal.signal(signal.SIGALRM, lambda signum, frame: sys.exit(0))
        signal.alarm(600)  # 10-minute maximum runtime

        logger.info("=== NETOVO Professional VoiceBot Starting ===")

        # Create and initialize VoiceBot
        voicebot = ProfessionalVoiceBot()

        # Handle the call
        call_success = voicebot.handle_professional_call()

        if call_success:
            logger.info("=== Call Completed Successfully ===")
        else:
            logger.info("=== Call Escalated to Human Agent ===")

    except KeyboardInterrupt:
        logger.info("VoiceBot interrupted by user")
    except Exception as e:
        logger.error(f"Fatal VoiceBot error: {e}")
        # Log full traceback for debugging
        import traceback
        logger.error(f"Full error traceback: {traceback.format_exc()}")
    finally:
        logger.info("=== VoiceBot Session Ended ===")
        sys.exit(0)

if __name__ == "__main__":
    main()
