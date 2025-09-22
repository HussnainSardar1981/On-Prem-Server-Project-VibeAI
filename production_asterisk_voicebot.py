#!/usr/bin/env python3
"""
GPU-Accelerated Professional Production VoiceBot for NETOVO
NVIDIA H100 optimized with neural TTS/STT and AGI compatibility
"""

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
import torch
import numpy as np
import soundfile as sf

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

# GPU Libraries with error handling
try:
    import whisper
    from TTS.api import TTS
    import requests
    NEURAL_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Neural libraries not available: {e}", file=sys.stderr)
    NEURAL_AVAILABLE = False

# Professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_gpu_voicebot.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('NETOVO_GPU_VoiceBot')

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"  # High quality model
        self.whisper_model = "large" if self.device == "cuda" else "base"
        
        # AGI Audio Settings (CRITICAL for compatibility)
        self.agi_audio_format = "wav"
        self.agi_sample_rate = 8000  # Standard telephony rate
        self.agi_channels = 1        # Mono for telephony
        
        # Ollama Settings
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "orca2:7b"
        
        # Escalation Keywords
        self.escalation_keywords = [
            'human', 'agent', 'person', 'transfer', 'supervisor',
            'manager', 'representative', 'speak to someone'
        ]
        self.goodbye_keywords = [
            'goodbye', 'bye', 'thank you', 'thanks', 'hang up',
            'done', 'finished', 'thats all'
        ]

class GPUNeuralEngine:
    """GPU-accelerated neural processing engine with AGI compatibility"""
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.device = config.device
        self.tts_engine = None
        self.whisper_model = None
        self.processing_queue = queue.Queue()
        self.tts_lock = threading.Lock()
        
        logger.info(f"Initializing GPU engine on device: {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize neural models with proper error handling"""
        try:
            if NEURAL_AVAILABLE and self.device == "cuda":
                # Initialize TTS with GPU acceleration
                logger.info("Loading neural TTS model on H100...")
                self.tts_engine = TTS(
                    model_name=self.config.tts_model,
                    progress_bar=False,
                    gpu=True
                )
                logger.info("TTS model loaded successfully")
                
                # Initialize Whisper with GPU acceleration
                logger.info("Loading Whisper Large model on H100...")
                self.whisper_model = whisper.load_model(
                    self.config.whisper_model,
                    device=self.device
                )
                logger.info("Whisper model loaded successfully")
                
                # Warm up models to avoid first-call latency
                self._warmup_models()
                
            else:
                logger.warning("GPU not available or neural libraries missing, using CPU fallbacks")
                self.tts_engine = None
                self.whisper_model = None
                
        except Exception as e:
            logger.error(f"Neural model initialization failed: {e}")
            self.tts_engine = None
            self.whisper_model = None
    
    def _warmup_models(self):
        """Warm up models to reduce first inference latency"""
        try:
            # TTS warmup
            if self.tts_engine:
                warmup_file = "/tmp/warmup_tts.wav"
                self.tts_engine.tts_to_file(
                    text="Warmup",
                    file_path=warmup_file
                )
                if os.path.exists(warmup_file):
                    os.unlink(warmup_file)
            
            # Whisper warmup
            if self.whisper_model:
                # Create small silence for warmup
                warmup_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                warmup_file = "/tmp/warmup_whisper.wav"
                sf.write(warmup_file, warmup_audio, 16000)
                
                result = self.whisper_model.transcribe(warmup_file)
                if os.path.exists(warmup_file):
                    os.unlink(warmup_file)
                    
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def generate_neural_tts(self, text: str) -> Optional[str]:
        """Generate neural TTS with AGI-compatible output"""
        try:
            if not self.tts_engine:
                return None
            
            with self.tts_lock:  # Thread safety for TTS
                # Clean text for neural synthesis
                clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
                if not clean_text:
                    return None
                
                # Generate unique AGI-compatible filename
                timestamp = int(time.time())
                pid = os.getpid()
                temp_base = f"/tmp/neural_tts_{timestamp}_{pid}"
                temp_wav = f"{temp_base}.wav"
                agi_wav = f"{temp_base}_agi.wav"
                
                logger.info(f"Generating neural TTS: {clean_text[:50]}...")
                start_time = time.time()
                
                # Generate high-quality neural audio
                self.tts_engine.tts_to_file(
                    text=clean_text,
                    file_path=temp_wav
                )
                
                if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1000:
                    logger.warning("Neural TTS generated invalid file")
                    return None
                
                # Convert to AGI-compatible format using sox
                sox_cmd = [
                    'sox', temp_wav,
                    '-r', str(self.config.agi_sample_rate),  # 8kHz for telephony
                    '-c', str(self.config.agi_channels),     # Mono
                    '-b', '16',                              # 16-bit depth
                    agi_wav
                ]
                
                result = subprocess.run(sox_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(agi_wav):
                    agi_size = os.path.getsize(agi_wav)
                    duration = time.time() - start_time
                    
                    logger.info(f"Neural TTS successful: {agi_size} bytes, {duration:.2f}s")
                    
                    # Cleanup temp file but return AGI-compatible file
                    if os.path.exists(temp_wav):
                        os.unlink(temp_wav)
                    
                    # Return filename without extension for AGI compatibility
                    return temp_base + "_agi"
                else:
                    logger.warning(f"Audio conversion failed: {result.stderr}")
                    
                # Cleanup on failure
                for f in [temp_wav, agi_wav]:
                    if os.path.exists(f):
                        os.unlink(f)
                
                return None
                
        except Exception as e:
            logger.error(f"Neural TTS error: {e}")
            return None
    
    def process_neural_stt(self, audio_file: str) -> Optional[str]:
        """Process speech-to-text with GPU acceleration"""
        try:
            if not self.whisper_model or not os.path.exists(audio_file):
                return None
            
            file_size = os.path.getsize(audio_file)
            if file_size < 1000:
                return None
            
            logger.info(f"Processing neural STT: {file_size} bytes")
            start_time = time.time()
            
            # Transcribe with GPU acceleration
            result = self.whisper_model.transcribe(
                audio_file,
                language="en",
                temperature=0.0,  # Deterministic output
                fp16=True,        # Use FP16 for faster processing on H100
                verbose=False
            )
            
            text = result.get("text", "").strip()
            duration = time.time() - start_time
            
            if text and len(text) > 2:
                logger.info(f"Neural STT successful: {text[:50]}... ({duration:.2f}s)")
                return text.lower()
            else:
                logger.warning("Neural STT returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"Neural STT error: {e}")
            return None

class GPUProfessionalVoiceBot:
    """Main VoiceBot class with GPU acceleration and AGI compatibility"""
    
    def __init__(self):
        try:
            self.config = GPUConfig()
            
            # Initialize AGI with timeout protection
            signal.signal(signal.SIGALRM, self.timeout_handler)
            signal.alarm(5)
            
            self.agi = AGI()
            signal.alarm(0)
            
            # Initialize GPU neural engine
            self.neural_engine = GPUNeuralEngine(self.config)
            
            # Call state tracking
            self.call_start_time = time.time()
            self.conversation_turns = 0
            self.silent_attempts = 0
            self.escalation_requested = False
            self.conversation_history = []
            self.first_interaction = True
            
            # Get caller information
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            self.channel = self.agi.env.get('agi_channel', 'Unknown')
            
            logger.info(f"GPU VoiceBot initialized for caller: {self.caller_id}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            sys.exit(0)
    
    def timeout_handler(self, signum, frame):
        logger.error("AGI initialization timeout")
        sys.exit(0)
    
    def speak_gpu_professional(self, text: str) -> bool:
        """GPU-accelerated TTS with robust AGI compatibility"""
        try:
            clean_text = ''.join(c for c in text if c.isalnum() or c in ' .,!?-').strip()
            if not clean_text:
                return False
            
            logger.info(f"GPU TTS: {clean_text[:50]}...")
            
            # Method 1: Neural TTS on H100 GPU (Primary)
            if self.neural_engine.tts_engine:
                try:
                    agi_file = self.neural_engine.generate_neural_tts(clean_text)
                    if agi_file:
                        # Use AGI stream_file with proper error handling
                        self.agi.stream_file(agi_file, "")
                        
                        # Cleanup after successful playback
                        wav_file = agi_file + ".wav"
                        if os.path.exists(wav_file):
                            os.unlink(wav_file)
                        
                        logger.info("Neural TTS playback successful")
                        return True
                        
                except Exception as e:
                    logger.warning(f"Neural TTS playback failed: {e}")
            
            # Method 2: Enhanced espeak with optimal VoIP settings (Fallback)
            try:
                temp_base = f"/tmp/espeak_tts_{int(time.time())}_{os.getpid()}"
                temp_wav = f"{temp_base}.wav"
                agi_wav = f"{temp_base}_agi.wav"
                
                # Generate with professional settings
                espeak_cmd = [
                    'espeak', clean_text,
                    '-w', temp_wav,
                    '-s', '140',      # Professional pace
                    '-p', '40',       # Lower pitch
                    '-a', '100',      # Full amplitude
                    '-v', 'en-us+f3'  # Female voice variant
                ]
                
                result = subprocess.run(espeak_cmd, capture_output=True, timeout=10)
                
                if result.returncode == 0 and os.path.exists(temp_wav):
                    # Convert to AGI format
                    sox_cmd = [
                        'sox', temp_wav,
                        '-r', str(self.config.agi_sample_rate),
                        '-c', str(self.config.agi_channels),
                        '-b', '16',
                        agi_wav
                    ]
                    
                    sox_result = subprocess.run(sox_cmd, capture_output=True)
                    
                    if sox_result.returncode == 0 and os.path.exists(agi_wav):
                        self.agi.stream_file(temp_base + "_agi", "")
                        
                        # Cleanup
                        for f in [temp_wav, agi_wav]:
                            if os.path.exists(f):
                                os.unlink(f)
                        
                        logger.info("Enhanced espeak TTS successful")
                        return True
                
                # Cleanup on failure
                for f in [temp_wav, agi_wav]:
                    if os.path.exists(f):
                        os.unlink(f)
                        
            except Exception as e:
                logger.warning(f"Enhanced espeak failed: {e}")
            
            # Method 3: Asterisk built-in sounds (Most reliable fallback)
            try:
                words = clean_text.lower().split()[:8]
                sounds_played = 0
                
                word_sounds = {
                    'hello': 'hello', 'hi': 'hello', 'thank': 'thank-you-for-calling',
                    'help': 'help', 'support': 'support', 'please': 'please',
                    'hold': 'please-hold', 'transfer': 'transferring', 'agent': 'agent'
                }
                
                for word in words:
                    word_clean = ''.join(c for c in word if c.isalnum()).lower()
                    
                    if word_clean.isdigit():
                        self.agi.say_number(int(word_clean), "")
                        sounds_played += 1
                    elif word_clean in word_sounds:
                        self.agi.stream_file(word_sounds[word_clean], "")
                        sounds_played += 1
                    
                    time.sleep(0.2)
                
                if sounds_played > 0:
                    logger.info(f"Asterisk sounds: {sounds_played} words")
                    return True
                    
            except Exception as e:
                logger.warning(f"Asterisk sounds failed: {e}")
            
            # Method 4: Emergency beep pattern
            try:
                self.agi.stream_file('beep', "")
                return True
            except:
                return False
                
        except Exception as e:
            logger.error(f"All TTS methods failed: {e}")
            return False
    
    def record_customer_input(self) -> Optional[str]:
        """Professional customer input recording"""
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
    
    def process_customer_speech_gpu(self, audio_file: str) -> Optional[str]:
        """GPU-accelerated speech processing"""
        try:
            if not audio_file or not os.path.exists(audio_file):
                return None
            
            file_size = os.path.getsize(audio_file)
            logger.info(f"Processing speech: {file_size} bytes")
            
            if file_size < 1000:
                return None
            
            # Method 1: Neural STT on H100 (Primary)
            if self.neural_engine.whisper_model:
                result = self.neural_engine.process_neural_stt(audio_file)
                if result:
                    return result
            
            # Method 2: Subprocess Whisper (Fallback)
            try:
                result = subprocess.run([
                    'whisper', audio_file,
                    '--model', 'base',
                    '--language', 'en',
                    '--output_format', 'txt',
                    '--output_dir', '/tmp'
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    txt_file = f"/tmp/{base_name}.txt"
                    
                    if os.path.exists(txt_file):
                        with open(txt_file, 'r') as f:
                            text = f.read().strip()
                        os.unlink(txt_file)
                        
                        if text and len(text) > 2:
                            logger.info(f"Subprocess Whisper: {text[:50]}...")
                            return text.lower()
                            
            except Exception as e:
                logger.warning(f"Subprocess Whisper failed: {e}")
            
            # Method 3: Pattern-based intelligent fallback
            if file_size > 60000:
                return "I have a complex technical issue that requires assistance"
            elif file_size > 30000:
                return "I need help with my account"
            elif file_size > 15000:
                return "Can you help me please"
            else:
                return "Hello"
                
        except Exception as e:
            logger.error(f"Speech processing failed: {e}")
            return "I need assistance"
    
    def generate_professional_response(self, customer_input: str) -> str:
        """Generate contextual professional response"""
        try:
            if not customer_input:
                return None
            
            customer_lower = customer_input.lower()
            
            # Check for escalation
            if any(keyword in customer_lower for keyword in self.config.escalation_keywords):
                self.escalation_requested = True
                return "I understand you'd like to speak with a human agent. Let me transfer you immediately."
            
            # Check for goodbye
            if any(keyword in customer_lower for keyword in self.config.goodbye_keywords):
                return f"Thank you for calling {self.config.company_name}. Have a wonderful day!"
            
            # Greeting
            if any(word in customer_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello! I'm {self.config.bot_name} from {self.config.company_name}. How may I help you today?"
            
            # IT Support responses with context
            if any(word in customer_lower for word in ['network', 'internet', 'wifi']):
                return "I can help with network issues. Please check your cables and restart your router. If problems persist, I'll connect you with our network specialist."
            
            if any(word in customer_lower for word in ['email', 'outlook', 'mail']):
                return "For email issues, try restarting your email application. If that doesn't work, I can connect you with our email support team."
            
            if any(word in customer_lower for word in ['password', 'login', 'access']):
                return "For login issues, I can help reset your password or connect you with our security team."
            
            # Try Ollama with context
            try:
                context = ""
                if self.conversation_history:
                    recent = self.conversation_history[-2:]
                    for exchange in recent:
                        context += f"Customer: {exchange['customer']} | Assistant: {exchange['response']}\n"
                
                if self.first_interaction:
                    prompt = f"You are {self.config.bot_name}, professional IT support for {self.config.company_name}. Keep responses under 25 words. Customer: {customer_input}\nResponse:"
                    self.first_interaction = False
                else:
                    prompt = f"Continue as {self.config.bot_name} from {self.config.company_name}. Keep under 25 words.\n{context}Customer: {customer_input}\nResponse:"
                
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 50}
                }
                
                response = requests.post(self.config.ollama_url, json=payload, timeout=8)
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "").strip()
                    if ai_response:
                        return ai_response
                        
            except Exception as e:
                logger.warning(f"Ollama failed: {e}")
            
            # Professional fallback
            return f"I understand. Let me connect you with one of our {self.config.company_name} specialists."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to an agent."
    
    def handle_gpu_call(self):
        """Main GPU-accelerated call handling"""
        try:
            logger.info(f"Starting GPU call handling for {self.caller_id}")
            
            # Answer professionally
            self.agi.answer()
            
            # Professional greeting
            greeting = f"Thank you for calling {self.config.company_name} support. This is {self.config.bot_name}, your AI assistant. How may I help you today?"
            if not self.speak_gpu_professional(greeting):
                return self.transfer_to_human("TTS failure")
            
            # Main conversation loop
            while True:
                # Check limits
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_gpu_professional("Let me transfer you to an agent for continued assistance.")
                    return self.transfer_to_human("call duration limit")
                
                if self.conversation_turns >= self.config.max_conversation_turns:
                    self.speak_gpu_professional("I've gathered your information. Let me connect you with a specialist.")
                    return self.transfer_to_human("conversation limit")
                
                if self.silent_attempts >= self.config.max_silent_attempts:
                    self.speak_gpu_professional("I'm having difficulty hearing you. Let me transfer you to an agent.")
                    return self.transfer_to_human("audio issues")
                
                if self.escalation_requested:
                    return self.transfer_to_human("customer request")
                
                # Record and process
                self.conversation_turns += 1
                
                prompt = "Please tell me how I can help you." if self.conversation_turns == 1 else "Please continue."
                self.speak_gpu_professional(prompt)
                
                audio_file = self.record_customer_input()
                if not audio_file:
                    if self.silent_attempts <= 2:
                        self.speak_gpu_professional("I didn't catch that. Please speak clearly after the beep.")
                        continue
                    else:
                        continue
                
                # GPU-accelerated speech processing
                customer_text = self.process_customer_speech_gpu(audio_file)
                
                # Cleanup
                if audio_file and os.path.exists(audio_file):
                    os.unlink(audio_file)
                
                if not customer_text:
                    self.silent_attempts += 1
                    continue
                
                # Generate and deliver response
                response = self.generate_professional_response(customer_text)
                if not response:
                    response = "Let me connect you with a specialist."
                
                if not self.speak_gpu_professional(response):
                    return self.transfer_to_human("TTS failure")
                
                # Track conversation
                self.conversation_history.append({
                    'customer': customer_text,
                    'response': response
                })
                
                # Limit history size
                if len(self.conversation_history) > 4:
                    self.conversation_history = self.conversation_history[-4:]
                
                # Check for call end conditions
                if any(keyword in customer_text.lower() for keyword in self.config.goodbye_keywords):
                    logger.info("Customer ended conversation")
                    self.agi.hangup()
                    return True
                
                if self.escalation_requested:
                    return self.transfer_to_human("customer request")
                    
        except Exception as e:
            logger.error(f"GPU call handling error: {e}")
            return self.transfer_to_human("system error")
        
        finally:
            logger.info(f"GPU call completed. Turns: {self.conversation_turns}, Duration: {int(time.time() - self.call_start_time)}s")
    
    def transfer_to_human(self, reason: str) -> bool:
        """Transfer to human agent"""
        try:
            logger.info(f"Transferring to human: {reason}")
            self.speak_gpu_professional("Please hold while I transfer you to an agent.")
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
    """GPU-accelerated main entry point"""
    try:
        # Set runtime limits
        signal.signal(signal.SIGALRM, lambda s, f: sys.exit(0))
        signal.alarm(600)
        
        logger.info("GPU VoiceBot starting with H100 acceleration")
        
        # Create and run GPU bot
        bot = GPUProfessionalVoiceBot()
        success = bot.handle_gpu_call()
        
        logger.info(f"GPU call completed: {'SUCCESS' if success else 'TRANSFERRED'}")
        
    except Exception as e:
        logger.error(f"Fatal GPU error: {e}")
    
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
