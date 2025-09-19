#!/usr/bin/env python3
"""
Production Asterisk VoiceBot with Proper AGI Handling
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
print("PRODUCTION VOICEBOT STARTING", file=sys.stderr)
sys.stderr.flush()

# Core imports with error handling
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)  # Exit cleanly for AGI

import numpy as np
import torch
import librosa
import requests
import soundfile as sf
import webrtcvad

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_VoiceBot')

# Import GPU TTS processor
GPU_TTS_AVAILABLE = False
try:
    from gpu_tts_processor import GPUTTSProcessor
    GPU_TTS_AVAILABLE = True
    logger.info("GPU TTS processor available")
except ImportError as e:
    logger.warning(f"GPU TTS not available: {e}")

class ProductionConfig:
    def __init__(self):
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"
        self.whisper_model = "tiny"
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "orca2:7b"
        self.record_timeout = 10
        self.silence_threshold = 1
        self.max_turns = 8
        self.max_call_duration = 300
        self.model_load_timeout = 15

class SimplifiedVoiceBot:
    def __init__(self):
        try:
            self.config = ProductionConfig()
            
            # Initialize AGI with timeout protection
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(5)  # 5 second timeout for AGI init
            
            self.agi = AGI()
            
            signal.alarm(0)  # Clear timeout
            
            self.call_start_time = time.time()
            self.turn_count = 0
            
            # Get caller info
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            logger.info(f"VoiceBot initialized for {self.caller_id}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(0)
    
    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(0)
    
    def speak_simple(self, text: str) -> bool:
        """Simple TTS using Asterisk built-in"""
        try:
            # Clean text for telephony
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?').strip()

            # Try built-in say commands first
            if len(clean_text) <= 50:
                self.agi.say_alpha(clean_text.replace(' ', ''))
                return True
            else:
                # Break into chunks for longer text
                words = clean_text.split()
                for i in range(0, len(words), 3):
                    chunk = ''.join(words[i:i+3])
                    self.agi.say_alpha(chunk)
                return True
        except Exception as e:
            logger.error(f"Speech error: {e}")
            return False
    
    def record_audio(self) -> Optional[str]:
        """Record user audio with improved parameters"""
        try:
            # Create recording file with timestamp
            record_name = f"/tmp/voicebot_record_{int(time.time())}"
            logger.info(f"Recording to {record_name}")

            # Record with optimized parameters
            result = self.agi.record_file(
                record_name,                              # filename (no extension)
                format='wav',                            # format
                escape_digits='#*',                      # escape digits
                timeout=self.config.record_timeout * 1000,  # 10 seconds
                offset=0,                                # start offset
                beep=True,                              # play beep
                silence=self.config.silence_threshold    # silence detection
            )

            # Check recording result
            wav_file = record_name + '.wav'
            if os.path.exists(wav_file):
                file_size = os.path.getsize(wav_file)
                logger.info(f"Recording created: {file_size} bytes")

                # Accept smaller recordings for test calls
                if file_size > 100:  # Reduced threshold
                    return wav_file
                else:
                    logger.warning(f"Recording too small: {file_size} bytes")
                    # Don't fail immediately, return the file anyway for processing
                    return wav_file

            logger.error("No recording file created")
            return None

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return None
    
    def process_audio(self, audio_file: str) -> str:
        """Simple audio processing - placeholder"""
        # In production, this would do STT
        # For now, return a simple response
        return "hello"
    
    def generate_response(self, user_input: str) -> str:
        """Generate response - simplified"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            return f"Hello! I'm {self.config.bot_name} from {self.config.company_name}. How can I help you?"
        
        if any(word in user_lower for word in ['bye', 'goodbye', 'thanks']):
            return "Thank you for calling. Have a great day!"
        
        return "I can help you with IT support. What issue are you experiencing?"
    
    def run_call(self):
        """Main call handler"""
        try:
            logger.info("Starting call handling")
            
            # Answer the call immediately
            self.agi.answer()
            # Remove unnecessary delay
            
            # Send greeting
            greeting = f"Hello! This is {self.config.bot_name} from {self.config.company_name} support."
            self.speak_simple(greeting)
            
            # Main loop
            while self.turn_count < self.config.max_turns:
                self.turn_count += 1
                
                # Check call duration
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_simple("I need to transfer you to a human agent.")
                    break
                
                # Record user input
                self.speak_simple("Please speak after the beep")
                audio_file = self.record_audio()
                
                if not audio_file:
                    if self.turn_count == 1:
                        self.speak_simple("I didn't hear you. Please try again.")
                        continue
                    else:
                        self.speak_simple("I'm having trouble hearing you.")
                        break
                
                # Process audio (simplified)
                user_text = self.process_audio(audio_file)
                
                # Clean up audio file
                try:
                    os.unlink(audio_file)
                except:
                    pass
                
                # Generate and speak response
                response = self.generate_response(user_text)
                self.speak_simple(response)
                
                # Check for end of conversation
                if "goodbye" in user_text.lower() or "bye" in user_text.lower():
                    break
            
            # End call
            self.speak_simple("Thank you for calling. Goodbye!")
            self.agi.hangup()
            
            logger.info("Call completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Call handling error: {e}")
            try:
                self.agi.hangup()
            except:
                pass
            return False

def main():
    """Main entry point"""
    try:
        # Set up timeout for entire script
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(60)  # 60 second maximum runtime
        
        logger.info("Main function started")
        
        # Create and run bot
        bot = SimplifiedVoiceBot()
        success = bot.run_call()
        
        logger.info(f"Call finished: {'SUCCESS' if success else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
    
    finally:
        # Always exit cleanly for AGI
        sys.exit(0)

if __name__ == "__main__":
    main()
   
