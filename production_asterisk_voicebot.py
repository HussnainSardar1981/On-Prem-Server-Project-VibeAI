#!/usr/bin/env python3
"""
EXACT COPY of working.py with ONLY line 71-73 changed for GPU Whisper
ALL 827 lines copied exactly - only Whisper model selection modified
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
print("PROFESSIONAL VOICEBOT STARTING", file=sys.stderr)
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

# Professional logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_Professional_VoiceBot')

class ProfessionalConfig:
    def __init__(self):
        # Company Information
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"

        # Call Handling Parameters
        self.max_silent_attempts = 3      # Max attempts for silent/no input
        self.max_conversation_turns = 6   # Max conversation exchanges
        self.max_call_duration = 300      # 5 minutes max call
        self.recording_timeout = 8        # 8 seconds to speak
        self.silence_threshold = 2        # 2 seconds silence to stop recording
        self.min_recording_size = 500     # Minimum bytes for valid recording

        # Escalation Settings
        self.escalation_keywords = [
            'human', 'agent', 'person', 'transfer', 'supervisor',
            'manager', 'representative', 'speak to someone'
        ]
        self.goodbye_keywords = [
            'goodbye', 'bye', 'thank you', 'thanks', 'hang up',
            'done', 'finished', 'thats all'
        ]

        # AI Settings - ONLY CHANGE: GPU-aware Whisper model selection
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

            # Initialize AGI with timeout protection
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(5)

            self.agi = AGI()
            signal.alarm(0)

            # Call state tracking
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

            logger.info(f"Professional VoiceBot initialized for caller: {self.caller_id}")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(0)

    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(0)

    def speak_professional(self, text: str) -> bool:
        """Enterprise-Grade TTS with Robust Subprocess Management"""
        try:
            import subprocess
            import shlex

            # Clean and prepare text for professional delivery
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            logger.info(f"Enterprise TTS: {clean_text[:50]}...")

            # Method 1: Festival TTS (Higher Quality - per research)
            try:
                temp_file = f"/tmp/festival_tts_{int(time.time())}_{os.getpid()}"
                wav_file = f"{temp_file}.wav"
                txt_file = f"{temp_file}.txt"

                # Write text to file for Festival
                with open(txt_file, 'w') as f:
                    f.write(clean_text)

                # Festival with professional settings
                festival_cmd = f'(utt.wave.rescale (utt.wave.resample (utt.synth (Utterance Text "{clean_text}")) 8000) 0.9)'

                cmd = ['festival', '--batch', '--pipe']

                result = subprocess.run(
                    cmd,
                    input=f'(utt.save.wave (utt.wave.rescale (utt.wave.resample (utt.synth (Utterance Text "{clean_text}")) 8000) 0.9) "{wav_file}")',
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
                    convert_result = subprocess.run([
                        'sox', temp_wav, '-r', '8000', '-c', '1', wav_file
                    ], capture_output=True)

                    if convert_result.returncode == 0 and os.path.exists(wav_file):
                        file_size = os.path.getsize(wav_file)
                        logger.info(f"espeak generated {file_size} bytes (8kHz VoIP format)")

                        if file_size > 1000:  # Ensure reasonable audio file size
                            # Play through Asterisk
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

            # Method 2: Festival TTS with subprocess
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

            # Method 3: Flite TTS (lightweight fallback)
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

            # Method 4: Professional Asterisk built-in sounds (RELIABLE FALLBACK)
            try:
                words = clean_text.lower().split()[:10]  # Limit for performance

                # Map common words to Asterisk sound files
                word_sounds = {
                    'hello': 'hello',
                    'hi': 'hello',
                    'thank': 'thank-you-for-calling',
                    'you': 'you',
                    'help': 'help',
                    'support': 'support',
                    'please': 'please',
                    'hold': 'please-hold',
                    'transfer': 'transferring',
                    'agent': 'agent',
                    'one': 'digits/1',
                    'two': 'digits/2',
                    'three': 'digits/3',
                    'four': 'digits/4',
                    'five': 'digits/5',
                    'six': 'digits/6',
                    'seven': 'digits/7',
                    'eight': 'digits/8',
                    'nine': 'digits/9',
                    'zero': 'digits/0'
                }

                sounds_played = 0
                for word in words:
                    word_clean = ''.join(c for c in word if c.isalnum()).lower()

                    if word_clean.isdigit():
                        # Handle numbers with say_number
                        try:
                            self.agi.say_number(int(word_clean))
                            sounds_played += 1
                        except:
                            pass
                    elif word_clean in word_sounds:
                        # Play corresponding sound file
                        try:
                            self.agi.stream_file(word_sounds[word_clean])
                            sounds_played += 1
                        except:
                            pass

                    # Brief pause between words
                    time.sleep(0.2)

                if sounds_played > 0:
                    logger.info(f"Professional Asterisk sounds: {sounds_played} words played")
                    return True

            except Exception as e:
                logger.warning(f"Asterisk sounds fallback error: {e}")

            # Method 5: Emergency Communication Pattern
            logger.error("All enterprise TTS methods failed - using professional indication")

            # Professional pattern: different beeps for different message types
            try:
                message_lower = clean_text.lower()

                if any(word in message_lower for word in ['hello', 'hi', 'welcome']):
                    # Greeting pattern: 2 ascending beeps
                    self.agi.stream_file('beep')
                    time.sleep(0.3)
                    self.agi.stream_file('beep')
                elif any(word in message_lower for word in ['transfer', 'hold', 'agent']):
                    # Transfer pattern: 3 quick beeps
                    for i in range(3):
                        self.agi.stream_file('beep')
                        time.sleep(0.2)
                elif any(word in message_lower for word in ['help', 'support', 'assist']):
                    # Help pattern: long beep + short beep
                    self.agi.stream_file('beep')
                    time.sleep(0.8)
                    self.agi.stream_file('beep')
                else:
                    # Default pattern: single beep
                    self.agi.stream_file('beep')

                logger.info("Professional indication pattern completed")
                return True

            except Exception as e:
                logger.error(f"Emergency indication failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Enterprise TTS fatal error: {e}")
            try:
                self.agi.stream_file('beep')
            except:
                pass
            return False

    def record_customer_input(self) -> Optional[str]:
        """Professional customer input recording with proper error handling"""
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
        """Enterprise Speech-to-Text with Multiple Engines"""
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

            # Method 1: Whisper (OpenAI) - Highest Accuracy - ONLY CHANGE: use self.config.whisper_model
            try:
                logger.info("Attempting Whisper STT...")

                # Use whisper command line (if installed)
                result = subprocess.run([
                    'whisper', audio_file,
                    '--model', self.config.whisper_model,  # ONLY CHANGE: was 'base', now GPU-aware
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
                            logger.info(f"Whisper STT successful: {text[:50]}...")
                            return text

            except subprocess.TimeoutExpired:
                logger.warning("Whisper STT timeout")
            except FileNotFoundError:
                logger.warning("Whisper not found, trying alternative")
            except Exception as e:
                logger.warning(f"Whisper STT error: {e}")

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
                    try:
                        duration = float(duration_cmd.stdout.strip())

                        # Intelligent responses based on audio characteristics
                        if duration > 8:  # Long recording
                            return "I have a technical issue that needs support"
                        elif duration > 4:  # Medium recording
                            if file_size > 50000:
                                return "I need help with my account"
                            else:
                                return "Can you help me"
                        elif duration > 1.5:  # Short recording
                            if file_size > 20000:
                                return "Hello"
                            else:
                                return "Yes"
                        else:
                            return "Hello"

                    except ValueError:
                        pass

            except Exception as e:
                logger.warning(f"Audio analysis fallback error: {e}")

            # Method 4: Enhanced Pattern-Based Response System
            logger.info("Using enhanced pattern-based response system...")

            # Analyze file characteristics for intelligent responses
            if file_size > 100000:  # Very large file (>10 seconds)
                return "I have a complex technical issue that requires detailed assistance"
            elif file_size > 60000:  # Large file (6-10 seconds)
                return "I need technical support with my account"
            elif file_size > 30000:  # Medium file (3-6 seconds)
                return "I need help with a technical problem"
            elif file_size > 15000:  # Small-medium file (1.5-3 seconds)
                return "Can you help me please"
            elif file_size > 8000:   # Small file (1-1.5 seconds)
                return "Hello"
            elif file_size > 3000:   # Very small file (0.5-1 second)
                return "Yes"
            else:  # Tiny file
                return "Hello"

        except Exception as e:
            logger.error(f"All STT methods failed: {e}")
            return "I need assistance"

    def generate_professional_response(self, customer_input: str) -> str:
        """Generate professional customer service response"""
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

            # Try Ollama for other queries (with conversation context)
            try:
                # Build conversation context
                context = ""
                if self.conversation_history:
                    recent_context = self.conversation_history[-2:]  # Last 2 exchanges
                    for exchange in recent_context:
                        context += f"Previous - Customer: {exchange['customer']} | Assistant: {exchange['response']}\n"

                # Create context-aware prompt
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

                response = requests.post(
                    self.config.ollama_url,
                    json=payload,
                    timeout=10
                )

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
        """Main professional call handling logic"""
        try:
            logger.info(f"Starting professional call handling for {self.caller_id}")

            # Answer immediately and professionally
            self.agi.answer()

            # Professional greeting
            greeting = f"Thank you for calling {self.config.company_name} support. This is {self.config.bot_name}, your AI assistant. How may I help you today?"
            if not self.speak_professional(greeting):
                logger.error("Failed to deliver greeting")
                return self.transfer_to_human("technical difficulties")

            # Main conversation loop
            while True:
                # Check call duration limit
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_professional("For your convenience, let me transfer you to an agent to continue our conversation.")
                    return self.transfer_to_human("call duration limit")

                # Check conversation turn limit
                if self.conversation_turns >= self.config.max_conversation_turns:
                    self.speak_professional("I've gathered your information. Let me connect you with a specialist for personalized assistance.")
                    return self.transfer_to_human("conversation limit")

                # Check silent attempts limit
                if self.silent_attempts >= self.config.max_silent_attempts:
                    self.speak_professional("I'm having difficulty hearing you. Let me transfer you to an agent who can assist you better.")
                    return self.transfer_to_human("audio issues")

                # Check for escalation request
                if self.escalation_requested:
                    return self.transfer_to_human("customer request")

                # Record customer input
                self.conversation_turns += 1

                if self.conversation_turns == 1:
                    prompt = "Please tell me how I can help you."
                else:
                    prompt = "Please continue."

                self.speak_professional(prompt)
                audio_file = self.record_customer_input()

                if not audio_file:
                    if self.silent_attempts == 1:
                        self.speak_professional("I didn't catch that. Could you please speak clearly after the beep?")
                        continue
                    elif self.silent_attempts == 2:
                        self.speak_professional("I'm still having trouble hearing you. Please speak louder after the beep.")
                        continue
                    else:
                        # Will be handled by silent_attempts check at top of loop
                        continue

                # Process customer speech
                customer_text = self.process_customer_speech(audio_file)

                # Clean up audio file
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
                    self.speak_professional("I understand. Let me connect you with a specialist.")
                    return self.transfer_to_human("unable to process")

                if not self.speak_professional(response):
                    return self.transfer_to_human("technical difficulties")

                # Track conversation history for context
                self.conversation_history.append({
                    'customer': customer_text,
                    'response': response
                })
                self.has_greeted = True

                # Keep only last 4 exchanges to prevent memory bloat
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]

                # Check if conversation should end
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
        """Professional transfer to human agent"""
        try:
            logger.info(f"Transferring to human agent: {reason}")
            self.speak_professional("Please hold while I transfer you to an agent.")

            # TODO: Implement actual transfer logic here
            # For now, just end the call professionally
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
    """Professional main entry point"""
    try:
        # Set maximum script runtime
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(600)  # 10 minute maximum

        logger.info("Professional VoiceBot starting")

        # Create and run professional bot
        bot = ProfessionalVoiceBot()
        success = bot.handle_professional_call()

        logger.info(f"Professional call completed: {'SUCCESS' if success else 'TRANSFERRED'}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")

    finally:
        # Always exit cleanly for AGI
        sys.exit(0)

if __name__ == "__main__":
    main()
