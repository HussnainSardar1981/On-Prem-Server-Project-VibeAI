#!/usr/bin/env python3
"""
Professional Production VoiceBot for NETOVO
Handles all customer scenarios with proper error recovery
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

        # AI Settings
        self.whisper_model = "tiny"
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
        """Professional text-to-speech - FIXED VERSION"""
        try:
            # Clean text for professional delivery
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            logger.info(f"Speaking: {clean_text[:50]}...")

            # QUICK FIX: Use espeak to generate proper speech
            try:
                temp_file = f"/tmp/speak_{int(time.time())}"
                wav_file = f"{temp_file}.wav"

                # Generate speech with espeak (pronounces words correctly)
                cmd = f'espeak "{clean_text}" -w {wav_file} -s 150 -p 50'
                result = os.system(cmd)

                if result == 0 and os.path.exists(wav_file):
                    # Play the generated speech file
                    self.agi.stream_file(temp_file)
                    # Cleanup
                    os.unlink(wav_file)
                    logger.info("espeak TTS successful")
                    return True
                else:
                    logger.warning(f"espeak failed with code: {result}")

            except Exception as e:
                logger.warning(f"espeak TTS failed: {e}")

            # FALLBACK 1: Try Festival if available
            try:
                temp_file = f"/tmp/festival_{int(time.time())}.wav"
                cmd = f'echo "{clean_text}" | festival --tts --otype wav --stdout > {temp_file}'
                result = os.system(cmd)

                if result == 0 and os.path.exists(temp_file):
                    self.agi.stream_file(temp_file.replace('.wav', ''))
                    os.unlink(temp_file)
                    logger.info("Festival TTS successful")
                    return True

            except Exception as e:
                logger.warning(f"Festival TTS failed: {e}")

            # FALLBACK 2: Use gTTS if available
            try:
                import gtts
                temp_file = f"/tmp/gtts_{int(time.time())}.mp3"

                tts = gtts.gTTS(text=clean_text, lang='en', slow=False)
                tts.save(temp_file)

                if os.path.exists(temp_file):
                    # Convert mp3 to wav for Asterisk
                    wav_file = temp_file.replace('.mp3', '.wav')
                    os.system(f'ffmpeg -i {temp_file} {wav_file} -y 2>/dev/null')

                    if os.path.exists(wav_file):
                        self.agi.stream_file(wav_file.replace('.wav', ''))
                        os.unlink(temp_file)
                        os.unlink(wav_file)
                        logger.info("gTTS successful")
                        return True

            except ImportError:
                logger.warning("gTTS not available")
            except Exception as e:
                logger.warning(f"gTTS failed: {e}")

            # FALLBACK 3: Simple word-by-word for short messages
            words = clean_text.split()
            if len(words) <= 8:  # Only for short messages
                try:
                    for word in words:
                        word_lower = word.lower()

                        # Handle numbers properly
                        if word_lower.isdigit():
                            self.agi.say_number(int(word_lower))
                        # Handle common words
                        elif word_lower in ['hello', 'hi', 'thank', 'you', 'please', 'help', 'support']:
                            # Try to find pre-recorded sound files
                            try:
                                self.agi.stream_file(f'custom/{word_lower}')
                            except:
                                # If no custom file, play beep for the word
                                self.agi.stream_file('beep')
                        else:
                            # For other words, just play a tone
                            self.agi.stream_file('beep')

                        time.sleep(0.3)  # Pause between words

                    logger.info("Word-by-word fallback completed")
                    return True

                except Exception as e:
                    logger.warning(f"Word-by-word failed: {e}")

            # LAST RESORT: Play beeps to indicate we're trying to communicate
            logger.error("All TTS methods failed - using indication beeps")
            for i in range(3):
                self.agi.stream_file('beep')
                time.sleep(0.5)
            return False

        except Exception as e:
            logger.error(f"Fatal speech error: {e}")
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
        """Process customer speech with STT (placeholder for now)"""
        try:
            # TODO: Implement Whisper STT here
            # For now, simulate based on file size for testing
            if os.path.exists(audio_file):
                file_size = os.path.getsize(audio_file)
                if file_size > 5000:
                    return "I need help with IT support"
                elif file_size > 1000:
                    return "hello"
                else:
                    return None
            return None
        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            return None

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

            # Try Ollama for other queries (with timeout)
            try:
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": f"You are {self.config.bot_name}, professional IT support for {self.config.company_name}. Keep responses under 25 words. Customer said: {customer_input}\n\nProfessional response:",
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
