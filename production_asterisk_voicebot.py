#!/usr/bin/env python3
"""
Production Asterisk VoiceBot for NETOVO 3CX Integration
Complete pipeline: SIP → STT → LLM → TTS → SIP
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

# Asterisk AGI
try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 module not found.")
    print("Install with: pip install pyst2")
    sys.exit(1)

# AI Pipeline Components (from your working test_pipeline.py)
import numpy as np
import torch
import librosa
# Import whisper inside functions to avoid conflicts
from TTS.api import TTS
import requests
import soundfile as sf
import webrtcvad

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
logger = logging.getLogger('NETOVO_VoiceBot')

class ProductionConfig:
    """Production configuration for NETOVO deployment"""
    
    def __init__(self):
        # 3CX Configuration (from NETOVO)
        self.sip_server = "mtipbx.ny.3cx.us"
        self.sip_port = 5060
        self.extension = "1600"
        self.auth_id = "qpZh2VS624"
        self.password = "FcHw0P2FHK"
        self.did = "+1 (646) 358-3509"
        
        # AI Models
        self.whisper_model = "base"  # Fast and accurate for production
        self.tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
        self.ollama_model = "orca2:7b"
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        
        # Audio Configuration
        self.sample_rate = 8000  # Telephony standard
        self.channels = 1
        self.record_timeout = 10  # seconds
        self.silence_threshold = 2  # seconds
        
        # Performance Settings
        self.max_turns = 15  # Prevent infinite conversations
        self.max_call_duration = 600  # 10 minutes max call
        self.response_timeout = 30  # AI response timeout
        
        # Business Logic
        self.company_name = "NETOVO"
        self.bot_name = "Alexis"
        self.support_hours = "24/7"
        
        logger.info(f"Production config loaded for {self.company_name}")

class AIProcessor:
    """Production AI pipeline processor"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Models (lazy loaded)
        self.whisper_model = None
        self.tts_model = None
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        
        # Performance tracking
        self.call_metrics = {
            'stt_times': [],
            'llm_times': [],
            'tts_times': [],
            'total_turns': 0,
            'call_start': time.time()
        }
        
        logger.info(f"AI Processor initialized on {self.device}")
    
    def initialize_models(self):
        """Initialize AI models (called once per call)"""
        try:
            logger.info("Loading AI models for new call...")
            start_time = time.time()
            
            # Load Whisper - FIXED IMPORT HANDLING
            if self.whisper_model is None:
                logger.info("Loading Whisper model...")
                whisper_module = None
                
                # Try different import methods for whisper
                import_methods = [
                    ('whisper', lambda: __import__('whisper')),
                    ('openai_whisper', lambda: __import__('openai_whisper')),
                    ('openai-whisper as whisper', lambda: __import__('openai_whisper'))
                ]
                
                for method_name, import_func in import_methods:
                    try:
                        logger.info(f"Trying import method: {method_name}")
                        whisper_module = import_func()
                        logger.info(f"Successfully imported whisper using: {method_name}")
                        break
                    except ImportError as e:
                        logger.warning(f"Import method {method_name} failed: {e}")
                        continue
                
                if whisper_module is None:
                    logger.error("All whisper import methods failed")
                    return False
                
                # Load the actual model
                try:
                    self.whisper_model = whisper_module.load_model(
                        self.config.whisper_model, 
                        device=self.device
                    )
                    logger.info("Whisper model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load whisper model: {e}")
                    return False
            
            # Load TTS
            if self.tts_model is None:
                logger.info("Loading TTS model...")
                self.tts_model = TTS(
                    model_name=self.config.tts_model,
                    gpu=(self.device == 'cuda'),
                    progress_bar=False
                )
                logger.info("TTS loaded successfully")
            
            load_time = time.time() - start_time
            logger.info(f"AI models ready in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"AI model initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def speech_to_text(self, audio_file: str) -> Optional[str]:
        """Convert speech to text with error handling"""
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            audio_data, sr = sf.read(audio_file)
            
            # Handle empty or too short audio
            if len(audio_data) < 1000:  # Less than 0.125s at 8kHz
                logger.warning("Audio too short for transcription")
                return None
            
            # Resample to 16kHz for Whisper
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_data, 
                language='en',
                task='transcribe',
                fp16=False  # More stable on some systems
            )
            
            text = result['text'].strip()
            processing_time = time.time() - start_time
            self.call_metrics['stt_times'].append(processing_time)
            
            logger.info(f"STT ({processing_time:.2f}s): '{text}'")
            return text if text else None
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None
    
    def generate_response(self, user_text: str, context: Dict[str, Any] = None) -> str:
        """Generate contextual AI response"""
        try:
            start_time = time.time()
            
            # Build context-aware prompt
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

            # Add conversation context if available
            if context and 'previous_topics' in context:
                system_prompt += f"\nPrevious topics discussed: {', '.join(context['previous_topics'])}"
            
            payload = {
                "model": self.config.ollama_model,
                "prompt": f"{system_prompt}\n\nCustomer said: {user_text}\n\nRespond as {self.config.bot_name}:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 150
                }
            }
            
            response = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.config.response_timeout
            )
            response.raise_for_status()
            
            bot_response = response.json().get("response", "").strip()
            processing_time = time.time() - start_time
            self.call_metrics['llm_times'].append(processing_time)
            
            logger.info(f"LLM ({processing_time:.2f}s): '{bot_response}'")
            
            # Fallback responses
            if not bot_response:
                return f"I'm here to help with your {self.config.company_name} IT needs. Could you please rephrase your question?"
            
            return bot_response
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timeout")
            return "I'm processing your request. Could you please wait a moment and try again?"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm experiencing some technical difficulties. Let me transfer you to a human agent."
    
    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech and return audio file path"""
        try:
            start_time = time.time()
            
            if not text.strip():
                return None
            
            # Generate TTS
            wav = self.tts_model.tts(text=text)
            
            # Convert to telephony format (8kHz, mono)
            wav_8k = librosa.resample(wav, orig_sr=22050, target_sr=self.config.sample_rate)
            
            # Normalize and clip to prevent distortion
            wav_8k = np.clip(wav_8k, -0.95, 0.95)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, wav_8k, self.config.sample_rate)
            
            processing_time = time.time() - start_time
            self.call_metrics['tts_times'].append(processing_time)
            
            logger.info(f"TTS ({processing_time:.2f}s): {len(wav_8k)} samples → {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def get_call_metrics(self) -> Dict[str, Any]:
        """Get call performance metrics"""
        return {
            'avg_stt_time': np.mean(self.call_metrics['stt_times']) if self.call_metrics['stt_times'] else 0,
            'avg_llm_time': np.mean(self.call_metrics['llm_times']) if self.call_metrics['llm_times'] else 0,
            'avg_tts_time': np.mean(self.call_metrics['tts_times']) if self.call_metrics['tts_times'] else 0,
            'total_turns': self.call_metrics['total_turns'],
            'call_duration': time.time() - self.call_metrics['call_start']
        }

class ProductionVoiceBot:
    """Production VoiceBot using Asterisk AGI"""
    
    def __init__(self):
        self.config = ProductionConfig()
        self.agi = AGI()
        self.ai_processor = AIProcessor(self.config)
        
        # Call state
        self.call_start_time = time.time()
        self.conversation_context = {
            'previous_topics': [],
            'user_sentiment': 'neutral',
            'escalation_requested': False
        }
        
        # Get call information
        self.caller_id = self.agi.env.get('agi_callerid', 'Unknown')
        self.channel = self.agi.env.get('agi_channel', 'Unknown')
        
        logger.info(f"Production VoiceBot initialized for call from {self.caller_id}")
    
    def speak_text(self, text: str, interrupt_key: str = '#') -> bool:
        """Speak text with interruption support"""
        try:
            audio_file = self.ai_processor.text_to_speech(text)
            if not audio_file:
                logger.error("TTS failed, using fallback")
                return False
            
            # Play audio file (remove .wav extension for Asterisk)
            audio_name = audio_file.replace('.wav', '')
            result = self.agi.stream_file(audio_name, interrupt_key)
            
            # Clean up temporary file
            try:
                os.unlink(audio_file)
            except:
                pass
            
            return result == 0
            
        except Exception as e:
            logger.error(f"Speech playback error: {e}")
            return False
    
    def record_speech(self, prompt: str = None) -> Optional[str]:
        """Record user speech with optional prompt"""
        try:
            if prompt:
                self.speak_text(prompt)
            
            # Create recording file
            record_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            record_name = record_file.name.replace('.wav', '')
            
            logger.info(f"Recording speech to {record_name}")
            
            # Record with silence detection
            self.agi.record_file(
                record_name,
                format='wav',
                escape_digits='#*',
                timeout=self.config.record_timeout * 1000,
                offset=0,
                beep=True,
                silence=self.config.silence_threshold
            )
            
            # Check if recording exists and has content
            if os.path.exists(record_file.name) and os.path.getsize(record_file.name) > 2048:
                logger.info(f"Recording successful: {os.path.getsize(record_file.name)} bytes")
                return record_file.name
            else:
                logger.warning("Recording failed or too short")
                try:
                    os.unlink(record_file.name)
                except:
                    pass
                return None
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None
    
    def handle_conversation_turn(self) -> bool:
        """Handle one conversation turn"""
        try:
            self.ai_processor.call_metrics['total_turns'] += 1
            turn_start = time.time()
            
            logger.info(f"Starting conversation turn {self.ai_processor.call_metrics['total_turns']}")
            
            # Record user speech
            audio_file = self.record_speech("Please speak after the tone, or press # when finished.")
            
            if not audio_file:
                if self.ai_processor.call_metrics['total_turns'] == 1:
                    self.speak_text("I didn't hear anything. Let me try again.")
                    return True
                else:
                    self.speak_text("I'm having trouble hearing you. Let me transfer you to a human agent.")
                    return False
            
            # Convert speech to text
            user_text = self.ai_processor.speech_to_text(audio_file)
            
            # Clean up audio file
            try:
                os.unlink(audio_file)
            except:
                pass
            
            if not user_text:
                self.speak_text("I didn't catch that. Could you please repeat your question?")
                return True
            
            # Check for conversation end signals
            end_phrases = ['goodbye', 'bye', 'thank you', 'thanks', 'that\'s all', 'hang up']
            if any(phrase in user_text.lower() for phrase in end_phrases):
                self.speak_text(f"Thank you for calling {self.config.company_name} support. Have a great day!")
                return False
            
            # Check for escalation requests
            escalation_phrases = ['human', 'agent', 'person', 'transfer', 'supervisor', 'manager']
            if any(phrase in user_text.lower() for phrase in escalation_phrases):
                self.conversation_context['escalation_requested'] = True
                self.speak_text("I'll transfer you to one of our human agents right away. Please hold.")
                return False
            
            # Generate AI response
            bot_response = self.ai_processor.generate_response(user_text, self.conversation_context)
            
            # Update conversation context
            self.conversation_context['previous_topics'].append(user_text[:50])
            if len(self.conversation_context['previous_topics']) > 3:
                self.conversation_context['previous_topics'].pop(0)
            
            # Speak response
            success = self.speak_text(bot_response)
            
            turn_time = time.time() - turn_start
            logger.info(f"Conversation turn completed in {turn_time:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Conversation turn error: {e}")
            self.speak_text("I'm experiencing technical difficulties. Please hold while I transfer you.")
            return False
    
    def send_greeting(self):
        """Send personalized greeting"""
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
            
            return self.speak_text(greeting)
            
        except Exception as e:
            logger.error(f"Greeting error: {e}")
            fallback = f"Hello! This is {self.config.bot_name} from {self.config.company_name} support. How can I help you today?"
            return self.speak_text(fallback)
    
    def run_call(self):
        """Main call handling method"""
        try:
            logger.info(f"Starting call from {self.caller_id} on channel {self.channel}")
            
            # Answer the call
            self.agi.answer()
            logger.info("Call answered successfully")
            
            # Initialize AI models
            if not self.ai_processor.initialize_models():
                self.speak_text("I'm sorry, I'm experiencing technical difficulties. Please call back in a few minutes.")
                return False
            
            # Send greeting
            if not self.send_greeting():
                logger.error("Failed to send greeting")
                return False
            
            # Main conversation loop
            while True:
                # Check call duration limit
                if time.time() - self.call_start_time > self.config.max_call_duration:
                    self.speak_text("I notice we've been talking for a while. Let me transfer you to a human agent for continued assistance.")
                    break
                
                # Check turn limit
                if self.ai_processor.call_metrics['total_turns'] >= self.config.max_turns:
                    self.speak_text("Let me transfer you to one of our human agents who can provide more detailed assistance.")
                    break
                
                # Handle conversation turn
                if not self.handle_conversation_turn():
                    break
            
            # Log final metrics
            metrics = self.ai_processor.get_call_metrics()
            logger.info(f"Call completed: {metrics}")
            
            return True
            
        except Exception as e:
            logger.error(f"Call handling error: {e}")
            try:
                self.speak_text("I'm sorry, I'm experiencing technical difficulties. Goodbye.")
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
    """Main entry point for Asterisk AGI"""
    try:
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, terminating call")
            sys.exit(1)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create and run voicebot
        voicebot = ProductionVoiceBot()
        success = voicebot.run_call()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
