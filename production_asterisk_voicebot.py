import os
import sys
import logging
import time
import tempfile
import signal
import threading
import queue
import subprocess
from typing import Optional, Dict, Any

# CRITICAL: Check AGI environment first
if sys.stdin.isatty():
    print("ERROR: This script must be called from Asterisk AGI", file=sys.stderr)
    sys.exit(0)

print("GPU VOICEBOT STARTING - H100 ACCELERATION", file=sys.stderr)
sys.stderr.flush()

try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(0)

# Try to import GPU libraries
try:
    import torch
    import whisper
    from TTS.api import TTS
    import soundfile as sf
    NEURAL_AVAILABLE = True
    print(f"Neural libraries loaded. CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
except ImportError as e:
    print(f"WARNING: Neural libraries not available: {e}", file=sys.stderr)
    NEURAL_AVAILABLE = False

import requests
import numpy as np

# Fixed logging setup - write to temp if /var/log fails
log_file = '/tmp/netovo_gpu_voicebot.log'
try:
    # Try to write to /var/log/asterisk if permissions allow
    test_log = '/var/log/asterisk/netovo_gpu_voicebot.log'
    with open(test_log, 'a') as f:
        f.write(f"Log test at {time.time()}\n")
    log_file = test_log
except (PermissionError, OSError):
    # Fall back to /tmp which is always writable
    log_file = '/tmp/netovo_gpu_voicebot.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_GPU_VoiceBot')
logger.info(f"Logging to: {log_file}")

class GPUConfig:
    def __init__(self):
        # Company Information
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"
        
        # Call Handling Parameters
        self.max_silent_attempts = 3
        self.max_conversation_turns = 6
        self.max_call_duration = 300
        self.recording_timeout = 8
        self.silence_threshold = 2
        self.min_recording_size = 500
        
        # GPU Settings
        self.device = "cuda" if torch.cuda.is_available() and NEURAL_AVAILABLE else "cpu"
        self.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
        self.whisper_model = "large" if self.device == "cuda" else "base"
        
        # AGI Audio Settings (CRITICAL for compatibility)
        self.agi_sample_rate = 8000  # 8kHz for telephony
        self.agi_channels = 1        # Mono
        
        # Ollama Settings  
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

class SafeGPUEngine:
    """GPU engine with comprehensive error handling"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = config.device
        self.tts_engine = None
        self.whisper_model = None
        self.tts_lock = threading.Lock()
        
        logger.info(f"Initializing engine on device: {self.device}")
        if NEURAL_AVAILABLE:
            self._safe_init_models()
    
    def _safe_init_models(self):
        """Safely initialize models with extensive error handling"""
        try:
            if self.device == "cuda":
                logger.info("Loading TTS model on H100...")
                self.tts_engine = TTS(
                    model_name=self.config.tts_model,
                    progress_bar=False,
                    gpu=True
                )
                logger.info("TTS model loaded")
                
                logger.info("Loading Whisper model on H100...")  
                self.whisper_model = whisper.load_model(
                    self.config.whisper_model,
                    device=self.device
                )
                logger.info("Whisper model loaded")
                
                # Quick warmup
                self._safe_warmup()
            else:
                logger.info("CUDA not available, using CPU fallbacks")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.tts_engine = None
            self.whisper_model = None
    
    def _safe_warmup(self):
        """Safe model warmup"""
        try:
            if self.tts_engine:
                warmup_file = f"/tmp/warmup_{int(time.time())}.wav"
                self.tts_engine.tts_to_file(text="test", file_path=warmup_file)
                if os.path.exists(warmup_file):
                    os.unlink(warmup_file)
                logger.info("TTS warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def generate_neural_tts(self, text: str) -> Optional[str]:
        """Generate neural TTS with extensive error handling"""
        if not self.tts_engine or not text:
            return None
            
        try:
            with self.tts_lock:
                clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
                if not clean_text:
                    return None
                
                timestamp = int(time.time())
                pid = os.getpid()
                temp_base = f"/tmp/neural_tts_{timestamp}_{pid}"
                temp_wav = f"{temp_base}.wav"
                agi_base = f"/tmp/agi_tts_{timestamp}_{pid}"
                agi_wav = f"{agi_base}.wav"
                
                logger.info(f"Neural TTS: {clean_text[:30]}...")
                
                # Generate with timeout
                self.tts_engine.tts_to_file(text=clean_text, file_path=temp_wav)
                
                if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
                    return None
                
                # Convert to telephony format
                sox_result = subprocess.run([
                    'sox', temp_wav,
                    '-r', '8000', '-c', '1', '-b', '16',
                    agi_wav
                ], capture_output=True, timeout=10)
                
                if sox_result.returncode == 0 and os.path.exists(agi_wav):
                    # Cleanup temp file
                    if os.path.exists(temp_wav):
                        os.unlink(temp_wav)
                    
                    logger.info("Neural TTS generated successfully")
                    return agi_base  # Return base name without extension
                
                # Cleanup on failure
                for f in [temp_wav, agi_wav]:
                    if os.path.exists(f):
                        os.unlink(f)
                
                return None
                
        except Exception as e:
            logger.error(f"Neural TTS error: {e}")
            return None
    
    def process_neural_stt(self, audio_file: str) -> Optional[str]:
        """Neural STT with error handling"""
        if not self.whisper_model or not os.path.exists(audio_file):
            return None
        
        try:
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None
                
            logger.info(f"Neural STT processing: {file_size} bytes")
            
            result = self.whisper_model.transcribe(
                audio_file,
                language="en", 
                temperature=0.0,
                fp16=True,
                verbose=False
            )
            
            text = result.get("text", "").strip()
            if text and len(text) > 2:
                logger.info(f"Neural STT: {text[:40]}...")
                return text.lower()
            
            return None
            
        except Exception as e:
            logger.error(f"Neural STT error: {e}")
            return None

class GPUVoiceBot:
    """Main GPU VoiceBot with comprehensive error handling"""
    
    def __init__(self):
        try:
            self.config = GPUConfig()
            
            # Initialize AGI with timeout
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(10)
            
            self.agi = AGI()
            signal.alarm(0)
            
            # Initialize GPU engine
            self.gpu_engine = SafeGPUEngine(self.config)
            
            # Call state
            self.call_start_time = time.time()
            self.conversation_turns = 0
            self.silent_attempts = 0
            self.escalation_requested = False
            self.conversation_history = []
            self.first_interaction = True
            
            # Get caller info
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            logger.info(f"GPU VoiceBot initialized for: {self.caller_id}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(1)
    
    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(1)
    
    def speak_professional(self, text: str) -> bool:
        """Multi-tier TTS with GPU priority"""
        try:
            if not text:
                return False
                
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            logger.info(f"TTS: {clean_text[:40]}...")
            
            # Tier 1: Neural TTS (GPU)
            if self.gpu_engine.tts_engine:
                agi_file = self.gpu_engine.generate_neural_tts(clean_text)
                if agi_file:
                    try:
                        self.agi.stream_file(agi_file, "")
                        # Cleanup
                        wav_file = agi_file + ".wav"
                        if os.path.exists(wav_file):
                            os.unlink(wav_file)
                        logger.info("Neural TTS successful")
                        return True
                    except Exception as e:
                        logger.warning(f"Neural TTS playback failed: {e}")
                        # Cleanup on failure
                        wav_file = agi_file + ".wav"
                        if os.path.exists(wav_file):
                            os.unlink(wav_file)
            
            # Tier 2: Enhanced espeak
            try:
                temp_base = f"/tmp/espeak_{int(time.time())}_{os.getpid()}"
                temp_wav = f"{temp_base}.wav"
                
                espeak_result = subprocess.run([
                    'espeak', clean_text,
                    '-w', temp_wav,
                    '-s', '140', '-p', '40', '-a', '100',
                    '-v', 'en-us+f3'
                ], capture_output=True, timeout=8)
                
                if espeak_result.returncode == 0 and os.path.exists(temp_wav):
                    agi_wav = f"{temp_base}_agi.wav"
                    
                    sox_result = subprocess.run([
                        'sox', temp_wav, '-r', '8000', '-c', '1', agi_wav
                    ], capture_output=True)
                    
                    if sox_result.returncode == 0:
                        self.agi.stream_file(temp_base + "_agi", "")
                        
                        # Cleanup
                        for f in [temp_wav, agi_wav]:
                            if os.path.exists(f):
                                os.unlink(f)
                        
                        logger.info("Enhanced espeak successful")
                        return True
                
                # Cleanup on failure
                for f in [temp_wav, temp_base + "_agi.wav"]:
                    if os.path.exists(f):
                        os.unlink(f)
                        
            except Exception as e:
                logger.warning(f"Enhanced espeak failed: {e}")
            
            # Tier 3: Asterisk sounds fallback
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
            
            # Tier 4: Emergency beep
            try:
                self.agi.stream_file('beep', "")
                return True
            except:
                return False
                
        except Exception as e:
            logger.error(f"All TTS failed: {e}")
            return False
    
    def record_customer_input(self) -> Optional[str]:
        """Record customer with proper error handling"""
        try:
            record_name = f"/tmp/customer_{int(time.time())}_{self.conversation_turns}"
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
                    return wav_file
                else:
                    self.silent_attempts += 1
                    if os.path.exists(wav_file):
                        os.unlink(wav_file)
                    return None
            else:
                self.silent_attempts += 1
                return None
                
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            self.silent_attempts += 1
            return None
    
    def process_speech(self, audio_file: str) -> Optional[str]:
        """Multi-tier STT processing"""
        try:
            if not audio_file or not os.path.exists(audio_file):
                return None
            
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None
            
            # Tier 1: Neural STT (GPU)
            if self.gpu_engine.whisper_model:
                result = self.gpu_engine.process_neural_stt(audio_file)
                if result:
                    return result
            
            # Tier 2: Subprocess Whisper
            try:
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
                            logger.info(f"Whisper subprocess: {text[:40]}...")
                            return text.lower()
                            
            except Exception as e:
                logger.warning(f"Whisper subprocess failed: {e}")
            
            # Tier 3: Intelligent fallback based on audio size
            if file_size > 60000:
                return "I have a technical issue that needs support"
            elif file_size > 30000:
                return "I need help with my account"  
            elif file_size > 15000:
                return "Can you help me"
            else:
                return "Hello"
                
        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            return "I need assistance"
    
    def generate_response(self, customer_input: str) -> str:
        """Generate contextual response"""
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
                return f"Hello! I'm {self.config.bot_name} from {self.config.company_name}. How can I help?"
            
            # IT support
            if any(w in customer_lower for w in ['network', 'internet', 'wifi']):
                return "For network issues, try restarting your router. If that doesn't help, I'll connect you with our network team."
            
            if any(w in customer_lower for w in ['email', 'outlook']):
                return "For email problems, try restarting your email app. I can also connect you with email support."
            
            if any(w in customer_lower for w in ['password', 'login']):
                return "For login issues, I can help reset passwords or connect you with our security team."
            
            # Try Ollama
            try:
                if self.first_interaction:
                    prompt = f"You are {self.config.bot_name}, IT support for {self.config.company_name}. Keep under 25 words. Customer: {customer_input}"
                    self.first_interaction = False
                else:
                    prompt = f"Continue as {self.config.bot_name}. Keep under 25 words. Customer: {customer_input}"
                
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
                        
            except:
                pass
            
            return f"Let me connect you with a {self.config.company_name} specialist."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm having technical difficulties. Let me transfer you."
    
    def handle_call(self):
        """Main call handling with comprehensive error recovery"""
        try:
            logger.info(f"Starting call for {self.caller_id}")
            
            # Answer
            self.agi.answer()
            
            # Greeting
            greeting = f"Thank you for calling {self.config.company_name}. This is {self.config.bot_name}. How can I help?"
            if not self.speak_professional(greeting):
                return self.transfer_to_human("greeting failed")
            
            # Main loop
            while True:
                # Check limits
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_professional("Let me transfer you for continued assistance.")
                    return self.transfer_to_human("time limit")
                
                if self.conversation_turns >= self.config.max_conversation_turns:
                    self.speak_professional("Let me connect you with a specialist.")
                    return self.transfer_to_human("turn limit")
                
                if self.silent_attempts >= self.config.max_silent_attempts:
                    self.speak_professional("I'm having trouble hearing. Let me transfer you.")
                    return self.transfer_to_human("audio issues")
                
                if self.escalation_requested:
                    return self.transfer_to_human("customer request")
                
                # Handle turn
                self.conversation_turns += 1
                
                prompt = "How can I help?" if self.conversation_turns == 1 else "Please continue."
                self.speak_professional(prompt)
                
                # Record
                audio_file = self.record_customer_input()
                if not audio_file:
                    if self.silent_attempts <= 2:
                        self.speak_professional("Please speak after the beep.")
                        continue
                    else:
                        continue
                
                # Process
                customer_text = self.process_speech(audio_file)
                
                # Cleanup
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)
                
                if not customer_text:
                    self.silent_attempts += 1
                    continue
                
                # Generate response
                response = self.generate_response(customer_text)
                if not response:
                    response = "Let me connect you with a specialist."
                
                if not self.speak_professional(response):
                    return self.transfer_to_human("response failed")
                
                # Track conversation
                self.conversation_history.append({
                    'customer': customer_text,
                    'response': response
                })
                
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]
                
                # Check end conditions
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
            logger.info(f"Call completed. Turns: {self.conversation_turns}")
    
    def transfer_to_human(self, reason: str) -> bool:
        """Transfer to human"""
        try:
            logger.info(f"Transfer: {reason}")
            self.speak_professional("Please hold while I transfer you.")
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
    """Main entry with comprehensive error handling"""
    try:
        # Runtime limit
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(600)  # 10 minutes max
        
        logger.info("GPU VoiceBot starting")
        
        # Create and run
        bot = GPUVoiceBot()
        success = bot.handle_call()
        
        logger.info(f"Call result: {'SUCCESS' if success else 'TRANSFERRED'}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Always log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
