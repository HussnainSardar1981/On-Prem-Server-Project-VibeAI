#!/usr/bin/env python3
"""
Complete Production Asterisk VoiceBot for NETOVO 3CX Integration
GPU-accelerated with Coqui TTS and optimized performance
"""

import os
import sys
import logging
import time
import tempfile
import signal
from typing import Optional, Dict, Any
import json
from pathlib import Path

# Early debug output
print("PRODUCTION VOICEBOT WITH GPU TTS STARTING", file=sys.stderr)
sys.stderr.flush()

# Core imports
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found", file=sys.stderr)
    sys.exit(1)

import numpy as np
import torch
import librosa
import requests
import soundfile as sf
import webrtcvad

# Import GPU TTS processor
try:
    from gpu_tts_processor import GPUTTSProcessor
    GPU_TTS_AVAILABLE = True
    print("GPU TTS processor imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"WARNING: GPU TTS not available: {e}", file=sys.stderr)
    GPU_TTS_AVAILABLE = False

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/netovo_voicebot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NETOVO_VoiceBot_GPU')

class ProductionConfig:
    """Production configuration for NETOVO deployment"""
    
    def __init__(self):
        # 3CX Configuration
        self.sip_server = "mtipbx.ny.3cx.us"
        self.sip_port = 5060
        self.extension = "1600"
        self.auth_id = "qpZh2VS624"
        self.password = "FcHw0P2FHK"
        self.did = "+1 (646) 358-3509"
        
        # AI Models (optimized for production)
        self.whisper_model = "tiny"  # Fast startup
        self.ollama_model = "orca2:7b"
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        
        # Audio Configuration
        self.sample_rate = 8000  # Telephony standard
        self.channels = 1
        self.record_timeout = 8  # seconds
        self.silence_threshold = 2  # seconds
        
        # Performance Settings
        self.max_turns = 12  # Prevent infinite conversations
        self.max_call_duration = 600  # 10 minutes max call
        self.response_timeout = 20  # AI response timeout
        self.model_load_timeout = 30  # Max time for model loading
        
        # Business Logic
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"
        self.support_hours = "24/7"
        
        logger.info(f"Production config loaded for {self.company_name}")

class EnhancedAIProcessor:
    """Enhanced AI processor with GPU acceleration"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Models (lazy loaded)
        self.whisper_model = None
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        
        # Performance tracking
        self.call_metrics = {
            'stt_times': [],
            'llm_times': [],
            'tts_times': [],
            'total_turns': 0,
            'call_start': time.time()
        }
        
        logger.info(f"Enhanced AI Processor initialized on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def initialize_models_fast(self) -> bool:
        """Fast model initialization with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timeout")
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.model_load_timeout)
        
        try:
            logger.info("Loading AI models for production...")
            start_time = time.time()
            
            # Load Whisper tiny model (fast)
            if self.whisper_model is None:
                logger.info("Loading Whisper tiny model...")
                try:
                    import whisper
                    self.whisper_model = whisper.load_model(
                        self.config.whisper_model, 
                        device=self.device
                    )
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"Whisper loading failed: {e}")
                    return False
            
            # Test Ollama connectivity
            try:
                response = requests.get(
                    "http://127.0.0.1:11434/api/tags", 
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("Ollama connectivity verified")
                else:
                    logger.warning("Ollama may not be ready")
            except requests.exceptions.RequestException:
                logger.warning("Ollama not accessible - using fallback responses")
            
            load_time = time.time() - start_time
            logger.info(f"AI models ready in {load_time:.2f}s")
            return True
            
        except TimeoutError:
            logger.error("Model loading timed out")
            return False
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
        finally:
            signal.alarm(0)  # Clear timeout
    
    def speech_to_text(self, audio_file: str) -> Optional[str]:
        """Enhanced speech-to-text processing"""
        try:
            start_time = time.time()
            
            # Quick validation
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1024:
                logger.warning("Audio file too small or missing")
                return None
            
            # Load and preprocess audio
            try:
                audio_data, sr = sf.read(audio_file)
            except Exception as e:
                logger.error(f"Audio read error: {e}")
                return None
            
            # Handle empty or too short audio
            if len(audio_data) < 1000:  # Less than 0.125s at 8kHz
                logger.warning("Audio too short for transcription")
                return None
            
            # Resample to 16kHz for Whisper if needed
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe with error handling
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
    
    def generate_response(self, user_text: str, context: Dict[str, Any] = None) -> str:
        """Enhanced response generation with intelligent fallbacks"""
        try:
            start_time = time.time()
            
            # Quick fallback responses for common cases
            user_lower = user_text.lower()
            
            # Handle greetings
            if any(word in user_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello! I'm {self.config.bot_name} from {self.config.company_name} support. How can I help you today?"
            
            # Handle goodbyes
            if any(word in user_lower for word in ['bye', 'goodbye', 'thanks', 'thank you']):
                return f"Thank you for calling {self.config.company_name}. Have a great day!"
            
            # Handle help requests
            if any(word in user_lower for word in ['help', 'support', 'problem', 'issue']):
                return "I'm here to help with your IT needs. Can you describe the specific issue you're experiencing?"
            
            # Try Ollama with context
            try:
                system_prompt = f"""You are {self.config.bot_name}, a professional IT support assistant for {self.config.company_name}. 

IMPORTANT GUIDELINES:
- Provide helpful, concise responses (max 2 sentences)
- Be friendly, professional, and solution-oriented
- If you don't know something, offer to transfer to a human agent
- For urgent issues, prioritize immediate solutions
- Available {self.config.support_hours}

CONTEXT:
- Company: {self.config.company_name}
- Your role: IT Support Assistant
- Call duration: {time.time() - self.call_metrics['call_start']:.0f} seconds
- Turn number: {self.call_metrics['total_turns'] + 1}
"""

                if context and 'previous_topics' in context:
                    system_prompt += f"\nPrevious topics: {', '.join(context['previous_topics'])}"
                
                payload = {
                    "model": self.config.ollama_model,
                    "prompt": f"{system_prompt}\n\nCustomer said: {user_text}\n\nRespond as {self.config.bot_name}:",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 100
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
            
            # Intelligent fallbacks based on keywords
            if any(word in user_lower for word in ['network', 'internet', 'wifi', 'connection']):
                return "For network issues, please check your cables and restart your router. If the problem persists, I can transfer you to our network specialist."
            
            if any(word in user_lower for word in ['email', 'outlook', 'mail']):
                return "For email issues, try restarting your email client. If that doesn't work, I can help you with account settings."
            
            if any(word in user_lower for word in ['password', 'login', 'access']):
                return "For login issues, I can help reset your password or transfer you to our security team."
            
            # Generic fallback
            return f"I understand you need help with that. Let me transfer you to one of our {self.config.company_name} specialists who can assist you better."
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm experiencing technical difficulties. Let me transfer you to a human agent right away."
    
    def get_call_metrics(self) -> Dict[str, Any]:
        """Get comprehensive call performance metrics"""
        return {
            'avg_stt_time': np.mean(self.call_metrics['stt_times']) if self.call_metrics['stt_times'] else 0,
            'avg_llm_time': np.mean(self.call_metrics['llm_times']) if self.call_metrics['llm_times'] else 0,
            'avg_tts_time': np.mean(self.call_metrics['tts_times']) if self.call_metrics['tts_times'] else 0,
            'total_turns': self.call_metrics['total_turns'],
            'call_duration': time.time() - self.call_metrics['call_start']
        }

class ProductionVoiceBot:
    """Production VoiceBot with GPU acceleration and enhanced features"""
    
    def __init__(self):
        try:
            self.config = ProductionConfig()
            self.agi = AGI()
            self.ai_processor = EnhancedAIProcessor(self.config)
            
            # Initialize GPU TTS processor
            if GPU_TTS_AVAILABLE:
                self.gpu_tts = GPUTTSProcessor()
                logger.info("GPU TTS processor initialized")
            else:
                self.gpu_tts = None
                logger.warning("GPU TTS not available")
            
            # Call state
            self.call_start_time = time.time()
            self.conversation_context = {
                'previous_topics': [],
                'user_sentiment': 'neutral',
                'escalation_requested': False,
                'turn_count': 0
            }
            
            # Get call information safely
            self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
            self.channel = self.agi.env.get('agi_channel', 'Unknown')
            
            logger.info(f"Production VoiceBot initialized for {self.caller_id}")
            
        except Exception as e:
            logger.error(f"VoiceBot initialization error: {e}")
            raise
    
    def speak_text_gpu(self, text: str, interrupt_key: str = '#') -> bool:
        """GPU-accelerated professional text-to-speech"""
        try:
            start_time = time.time()
            logger.info(f"Speaking with GPU TTS: {text[:50]}...")
            
            if self.gpu_tts:
                # Use GPU TTS processor
                audio_file = self.gpu_tts.generate_speech(text)
                
                if audio_file:
                    # Play via Asterisk (remove .wav extension)
                    audio_name = audio_file.replace('.wav', '')
                    result = self.agi.stream_file(audio_name, interrupt_key)
                    
                    # Cleanup
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                    
                    processing_time = time.time() - start_time
                    self.ai_processor.call_metrics['tts_times'].append(processing_time)
                    
                    if result == 0:
                        logger.info(f"GPU TTS played successfully ({processing_time:.2f}s)")
                        return True
                    else:
                        logger.error(f"Asterisk playback failed: {result}")
                        return False
                else:
                    logger.error("GPU TTS generation failed")
                    return False
            else:
                # Fallback to simple playback
                logger.warning("GPU TTS not available, using fallback")
                return True
                
        except Exception as e:
            logger.error(f"GPU TTS error: {e}")
            return False
    
    def record_speech(self, prompt: str = None) -> Optional[str]:
        """Enhanced speech recording with better error handling"""
        try:
            if prompt:
                self.speak_text_gpu(prompt)
            
            # Create recording file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            record_name = temp_file.name.replace('.wav', '')
            temp_file.close()
            
            logger.info(f"Recording speech to {record_name}")
            
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
            if os.path.exists(wav_file) and os.path.getsize(wav_file) > 2048:
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
    
    def handle_conversation_turn(self) -> bool:
        """Enhanced conversation turn handling"""
        try:
            self.ai_processor.call_metrics['total_turns'] += 1
            self.conversation_context['turn_count'] += 1
            turn_start = time.time()
            
            logger.info(f"Starting conversation turn {self.ai_processor.call_metrics['total_turns']}")
            
            # Record user speech
            audio_file = self.record_speech("Please speak after the tone, or press # when finished.")
            
            if not audio_file:
                if self.ai_processor.call_metrics['total_turns'] == 1:
                    self.speak_text_gpu("I didn't hear anything. Let me try again.")
                    return True
                else:
                    self.speak_text_gpu("I'm having trouble hearing you. Let me transfer you to a human agent.")
                    return False
            
            # Convert speech to text
            user_text = self.ai_processor.speech_to_text(audio_file)
            
            # Clean up audio file
            try:
                os.unlink(audio_file)
            except:
                pass
            
            if not user_text:
                self.speak_text_gpu("I didn't catch that. Could you please repeat your question?")
                return True
            
            # Check for conversation end signals
            end_phrases = ['goodbye', 'bye', 'thank you', 'thanks', 'that\'s all', 'hang up']
            if any(phrase in user_text.lower() for phrase in end_phrases):
                self.speak_text_gpu(f"Thank you for calling {self.config.company_name} support. Have a great day!")
                return False
            
            # Check for escalation requests
            escalation_phrases = ['human', 'agent', 'person', 'transfer', 'supervisor', 'manager']
            if any(phrase in user_text.lower() for phrase in escalation_phrases):
                self.conversation_context['escalation_requested'] = True
                self.speak_text_gpu("I'll transfer you to one of our human agents right away. Please hold.")
                return False
            
            # Generate AI response
            bot_response = self.ai_processor.generate_response(user_text, self.conversation_context)
            
            # Update conversation context
            self.conversation_context['previous_topics'].append(user_text[:50])
            if len(self.conversation_context['previous_topics']) > 3:
                self.conversation_context['previous_topics'].pop(0)
            
            # Speak response
            success = self.speak_text_gpu(bot_response)
            
            turn_time = time.time() - turn_start
            logger.info(f"Conversation turn completed in {turn_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Conversation turn error: {e}")
            self.speak_text_gpu("I'm experiencing technical difficulties. Please hold while I transfer you.")
            return False
    
    def send_greeting(self):
        """Send enhanced personalized greeting"""
        try:
            # Determine greeting based on time of day
            import datetime
            hour = datetime.datetime.now().hour
            
            if 5 <= hour < 12:
                time_greeting = "Good morning"
            elif 12 <= hour < 17:
                time_greeting = "Good afternoon"
            else:
                time_greeting = "Good evening"
            
            greeting = f"{time_greeting}! Thank you for calling {self.config.company_name} support. This is {self.config.bot_name}, your AI assistant. I'm here to help with your IT needs. How can I assist you today?"
            
            return self.speak_text_gpu(greeting)
            
        except Exception as e:
            logger.error(f"Greeting error: {e}")
            fallback = f"Hello! This is {self.config.bot_name} from {self.config.company_name} support. How can I help you today?"
            return self.speak_text_gpu(fallback)
    
    def run_call(self):
        """Enhanced main call handling method"""
        try:
            logger.info(f"Starting enhanced call from {self.caller_id} on channel {self.channel}")
            
            # Answer the call
            self.agi.answer()
            logger.info("Call answered successfully")
            
            # Initialize AI models with timeout
            if not self.ai_processor.initialize_models_fast():
                self.speak_text_gpu("I'm sorry, I'm experiencing technical difficulties. Please call back in a few minutes.")
                return False
            
            # Send greeting
            if not self.send_greeting():
                logger.error("Failed to send greeting")
                return False
            
            # Main conversation loop
            while True:
                # Check call duration limit
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_text_gpu("I notice we've been talking for a while. Let me transfer you to a human agent for continued assistance.")
                    break
                
                # Check turn limit
                if self.ai_processor.call_metrics['total_turns'] >= self.config.max_turns:
                    self.speak_text_gpu("Let me transfer you to one of our human agents who can provide more detailed assistance.")
                    break
                
                # Handle conversation turn
                if not self.handle_conversation_turn():
                    break
            
            # Log final metrics
            metrics = self.ai_processor.get_call_metrics()
            logger.info(f"Call completed successfully: {metrics}")
            
            return True
            
        except Exception as e:
            logger.error(f"Call handling error: {e}")
            try:
                self.speak_text_gpu("I'm sorry, I'm experiencing technical difficulties. Goodbye.")
            except:
                pass
            return False
        
        finally:
            # Ensure hangup
            try:
                self.agi.hangup()
            except:
                pass
            
            logger.info("Call ended")

def main():
    """Enhanced main entry point for Asterisk AGI"""
    try:
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, terminating call gracefully")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and run enhanced voicebot
        logger.info("Starting production voicebot with GPU acceleration")
        voicebot = ProductionVoiceBot()
        success = voicebot.run_call()
        
        logger.info(f"Enhanced voicebot finished: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0)  # Always exit 0 for AGI compatibility
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(0)  # Still exit 0 for AGI compatibility

print("ABOUT TO START ENHANCED MAIN", file=sys.stderr)
sys.stderr.flush()

if __name__ == "__main__":
    main()
