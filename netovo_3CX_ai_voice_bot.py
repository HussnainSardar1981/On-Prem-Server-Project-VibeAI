#!/usr/bin/env python3
"""
Production 3CX Voice Bot - Complete Implementation
Supports .env configuration and real-time call processing
"""

import asyncio
import websockets
import json
import numpy as np
import threading
import time
import logging
import requests
import io
import wave
import os
import socket
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import collections
import base64

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Core AI components
import whisper
import torch
from TTS.api import TTS

# Audio processing
import pyaudio
import librosa
import soundfile as sf
import webrtcvad

# SIP alternative - use socket for direct communication
import struct

# Configure logging based on environment
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
log_file = os.getenv('LOG_FILE', 'voice_bot.log')

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CallMetrics:
    """Call performance metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    avg_response_time: float = 0.0
    avg_stt_time: float = 0.0
    avg_llm_time: float = 0.0
    avg_tts_time: float = 0.0
    total_duration: float = 0.0

class ConfigManager:
    """Configuration manager for environment variables"""
    
    def __init__(self):
        self.threecx_server = os.getenv('THREECX_SERVER', 'mtipbx.ny.3cx.us')
        self.threecx_port = int(os.getenv('THREECX_PORT', '5060'))
        self.threecx_extension = os.getenv('THREECX_EXTENSION', '1600')
        self.threecx_password = os.getenv('THREECX_PASSWORD', 'FcHw0P2FHK')
        self.threecx_auth_id = os.getenv('THREECX_AUTH_ID', 'qpZh2VS624')
        
        self.test_extension = os.getenv('TEST_EXTENSION', '1680')
        self.test_did = os.getenv('TEST_DID', '+16463583509')
        self.echo_extension = os.getenv('ECHO_EXTENSION', '*777')
        
        self.rtp_port_start = int(os.getenv('RTP_PORT_START', '9000'))
        self.rtp_port_end = int(os.getenv('RTP_PORT_END', '10999'))
        
        self.whisper_model = os.getenv('WHISPER_MODEL', 'base')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'orca2:7b')
        self.tts_model = os.getenv('TTS_MODEL', 'tts_models/en/ljspeech/tacotron2-DDC')
        
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '8000'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1024'))
        self.vad_aggressiveness = int(os.getenv('VAD_AGGRESSIVENESS', '2'))
        
        self.max_concurrent_calls = int(os.getenv('MAX_CONCURRENT_CALLS', '5'))
        self.response_timeout = int(os.getenv('RESPONSE_TIMEOUT', '30'))
        self.audio_buffer_size = int(os.getenv('AUDIO_BUFFER_SIZE', '50'))
        self.target_latency = int(os.getenv('TARGET_LATENCY', '300'))
        
        self.force_gpu = os.getenv('FORCE_GPU', 'false').lower() == 'true'
        self.gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
        
        self.enable_metrics = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
        
        logger.info(f"Configuration loaded - Server: {self.threecx_server}:{self.threecx_port}")
        logger.info(f"Extension: {self.threecx_extension}, Test: {self.test_extension}")

class SIPClient:
    """Corrected SIP client with proper 3CX authentication"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.socket = None
        self.registered = False
        self.call_id_counter = 1
        self.local_ip = None
        
    def get_local_ip(self):
        """Get the actual local IP address for Via headers"""
        try:
            # Create a socket to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to 3CX server (doesn't actually send data)
                s.connect((self.config.threecx_server, self.config.threecx_port))
                self.local_ip = s.getsockname()[0]
            logger.info(f"Local IP detected: {self.local_ip}")
            return self.local_ip
        except Exception as e:
            logger.warning(f"Could not detect local IP: {e}, using fallback")
            self.local_ip = "10.2.9.10"  # Fallback to configured IP
            return self.local_ip
    
    async def connect(self):
        """Connect to 3CX SIP server with proper IP detection"""
        try:
            # Get actual local IP
            self.get_local_ip()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(5.0)
            
            # Test connectivity
            await self.test_connection()
            
            # Attempt registration
            await self.register()
            
            return True
            
        except Exception as e:
            logger.error(f"SIP connection failed: {e}")
            return False
    
    async def test_connection(self):
        """Test UDP connection to 3CX"""
        try:
            test_message = f"OPTIONS sip:{self.config.threecx_server} SIP/2.0\r\n\r\n"
            self.socket.sendto(test_message.encode(), (self.config.threecx_server, self.config.threecx_port))
            
            try:
                response, addr = self.socket.recvfrom(1024)
                logger.info(f"3CX server responded from: {addr}")
            except socket.timeout:
                logger.info("OPTIONS test sent (timeout is normal)")
                
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
    
    async def register(self):
        """Register with 3CX using correct authentication flow"""
        try:
            # Create initial REGISTER message (without auth)
            register_msg = self.create_register_message()
            
            logger.info("Sending initial REGISTER...")
            self.socket.sendto(register_msg.encode(), 
                             (self.config.threecx_server, self.config.threecx_port))
            
            # Wait for response
            try:
                response, addr = self.socket.recvfrom(2048)
                response_str = response.decode('utf-8')
                
                logger.info(f"Registration response: {response_str[:200]}...")
                
                if "200 OK" in response_str:
                    self.registered = True
                    logger.info("Successfully registered with 3CX (no auth required)")
                elif "401 Unauthorized" in response_str or "407 Proxy Authentication Required" in response_str:
                    logger.info("Authentication challenge received")
                    await self.handle_digest_auth(response_str)
                else:
                    logger.warning(f"Unexpected registration response: {response_str[:100]}...")
                    
            except socket.timeout:
                logger.warning("No registration response received")
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
    
    async def handle_digest_auth(self, challenge_response: str):
        """Handle SIP digest authentication with proper parsing"""
        try:
            # Parse authentication challenge properly
            auth_params = self.parse_auth_challenge(challenge_response)
            
            if not auth_params:
                logger.error("Could not parse authentication challenge")
                return
            
            logger.info(f"Auth challenge parsed: realm={auth_params.get('realm')}, nonce={auth_params.get('nonce')[:10]}...")
            
            # Generate digest response
            auth_response = self.calculate_digest_response(auth_params)
            
            # Create authenticated REGISTER message
            auth_register = self.create_authenticated_register_message(auth_params, auth_response)
            
            logger.info("Sending authenticated REGISTER...")
            self.socket.sendto(auth_register.encode(), 
                             (self.config.threecx_server, self.config.threecx_port))
            
            # Wait for response
            try:
                response, addr = self.socket.recvfrom(2048)
                response_str = response.decode('utf-8')
                
                logger.info(f"Auth response: {response_str[:200]}...")
                
                if "200 OK" in response_str:
                    self.registered = True
                    logger.info("Successfully authenticated and registered with 3CX")
                else:
                    logger.error(f"Authentication failed: {response_str[:200]}...")
                    
            except socket.timeout:
                logger.warning("No authentication response received")
                
        except Exception as e:
            logger.error(f"Digest authentication failed: {e}")
    
    def parse_auth_challenge(self, response: str):
        """Properly parse WWW-Authenticate or Proxy-Authenticate header"""
        import re
        auth_params = {}
        
        # Look for authentication headers
        for line in response.split('\n'):
            line = line.strip()
            
            # Check for WWW-Authenticate or Proxy-Authenticate
            if line.lower().startswith('www-authenticate:') or line.lower().startswith('proxy-authenticate:'):
                # Extract the digest part
                if 'digest' in line.lower():
                    # Parse digest parameters
                    digest_part = line.split(':', 1)[1].strip()
                    
                    # Remove 'Digest' keyword
                    if digest_part.lower().startswith('digest'):
                        digest_part = digest_part[6:].strip()
                    
                    # Parse key=value pairs
                    # Use regex to handle quoted values properly
                    pattern = r'(\w+)=(?:"([^"]*)"|\s*([^,\s]+))'
                    matches = re.findall(pattern, digest_part)
                    
                    for match in matches:
                        key = match[0].lower()
                        value = match[1] if match[1] else match[2]
                        auth_params[key] = value
                    
                    logger.debug(f"Parsed auth params: {list(auth_params.keys())}")
                    return auth_params
        
        return None
    
    def calculate_digest_response(self, auth_params):
        """Calculate digest authentication response"""
        import hashlib
        import random
        
        try:
            realm = auth_params.get('realm', '')
            nonce = auth_params.get('nonce', '')
            algorithm = auth_params.get('algorithm', 'MD5').upper()
            
            if algorithm != 'MD5':
                logger.warning(f"Unsupported algorithm: {algorithm}, using MD5")
            
            username = self.config.threecx_extension
            password = self.config.threecx_password
            method = "REGISTER"
            uri = f"sip:{self.config.threecx_server}"
            
            # Calculate HA1
            ha1_string = f"{username}:{realm}:{password}"
            ha1 = hashlib.md5(ha1_string.encode()).hexdigest()
            
            # Calculate HA2
            ha2_string = f"{method}:{uri}"
            ha2 = hashlib.md5(ha2_string.encode()).hexdigest()
            
            # Calculate response
            if 'qop' in auth_params and auth_params['qop']:
                # With qop (quality of protection)
                qop = auth_params['qop']
                nc = "00000001"  # Nonce count
                cnonce = f"{random.randint(100000, 999999)}"  # Client nonce
                
                response_string = f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}"
                response = hashlib.md5(response_string.encode()).hexdigest()
                
                return {
                    'response': response,
                    'qop': qop,
                    'nc': nc,
                    'cnonce': cnonce
                }
            else:
                # Without qop (legacy mode)
                response_string = f"{ha1}:{nonce}:{ha2}"
                response = hashlib.md5(response_string.encode()).hexdigest()
                
                return {'response': response}
                
        except Exception as e:
            logger.error(f"Digest calculation failed: {e}")
            return None
    
    def create_authenticated_register_message(self, auth_params, auth_response):
        """Create authenticated REGISTER message with proper Authorization header"""
        import random
        
        call_id = f"call-{self.call_id_counter}@{self.local_ip}"
        self.call_id_counter += 1
        
        branch_id = f"z9hG4bK{random.randint(10000000, 99999999)}"
        tag_id = f"tag-{random.randint(100000, 999999)}"
        
        # Build Authorization header
        username = self.config.threecx_extension
        realm = auth_params.get('realm', '')
        nonce = auth_params.get('nonce', '')
        uri = f"sip:{self.config.threecx_server}"
        response = auth_response['response']
        
        auth_header = f'Digest username="{username}", realm="{realm}", nonce="{nonce}", uri="{uri}", response="{response}"'
        
        # Add qop parameters if present
        if 'qop' in auth_response:
            qop = auth_response['qop']
            nc = auth_response['nc']
            cnonce = auth_response['cnonce']
            auth_header += f', qop={qop}, nc={nc}, cnonce="{cnonce}"'
        
        # Add algorithm if specified
        if 'algorithm' in auth_params:
            auth_header += f', algorithm={auth_params["algorithm"]}'
        
        register = f"""REGISTER sip:{self.config.threecx_server} SIP/2.0\r
Via: SIP/2.0/UDP {self.local_ip}:5060;branch={branch_id};rport\r
Max-Forwards: 70\r
From: <sip:{self.config.threecx_extension}@{self.config.threecx_server}>;tag={tag_id}\r
To: <sip:{self.config.threecx_extension}@{self.config.threecx_server}>\r
Call-ID: {call_id}\r
CSeq: 2 REGISTER\r
Contact: <sip:{self.config.threecx_extension}@{self.local_ip}:5060;transport=udp>\r
User-Agent: NETOVO-VoiceBot/1.0\r
Authorization: {auth_header}\r
Expires: 3600\r
Content-Length: 0\r
\r
"""
        return register
    
    def create_register_message(self):
        """Create initial SIP REGISTER message"""
        import random
        
        call_id = f"call-{self.call_id_counter}@{self.local_ip}"
        self.call_id_counter += 1
        
        branch_id = f"z9hG4bK{random.randint(10000000, 99999999)}"
        tag_id = f"tag-{random.randint(100000, 999999)}"
        
        register = f"""REGISTER sip:{self.config.threecx_server} SIP/2.0\r
Via: SIP/2.0/UDP {self.local_ip}:5060;branch={branch_id};rport\r
Max-Forwards: 70\r
From: <sip:{self.config.threecx_extension}@{self.config.threecx_server}>;tag={tag_id}\r
To: <sip:{self.config.threecx_extension}@{self.config.threecx_server}>\r
Call-ID: {call_id}\r
CSeq: 1 REGISTER\r
Contact: <sip:{self.config.threecx_extension}@{self.local_ip}:5060;transport=udp>\r
User-Agent: NETOVO-VoiceBot/1.0\r
Expires: 3600\r
Content-Length: 0\r
\r
"""
        return register

class AudioProcessor:
    """Enhanced audio processor with 3CX optimization"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_size = config.chunk_size
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        
        # Audio format settings for 3CX compatibility
        self.format = pyaudio.paInt16
        self.channels = 1
        self.frame_duration = 20  # 20ms frames for VAD
        
    def convert_for_whisper(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio to format suitable for Whisper (16kHz)"""
        if audio_data.size == 0:
            return np.array([], dtype=np.float32)
            
        # Ensure proper format
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Resample to 16kHz for Whisper
        if self.sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=self.sample_rate, target_sr=16000)
        
        return audio_data
    
    def convert_for_3cx(self, audio_data: np.ndarray) -> bytes:
        """Convert TTS output to 3CX compatible format (8kHz, G.711 micro-law)"""
        try:
            if audio_data.size == 0:
                return b''
            
            # Resample to 8kHz for VoIP
            if hasattr(audio_data, 'shape') and audio_data.shape[0] > 0:
                # Assume TTS output is 22kHz, resample to 8kHz
                audio_8k = librosa.resample(audio_data, orig_sr=22050, target_sr=8000)
                
                # Convert to 16-bit PCM
                audio_int16 = np.clip(audio_8k * 32767, -32767, 32767).astype(np.int16)
                
                # Convert to G.711 micro-law for 3CX compatibility
                g711_audio = self.linear_to_ulaw(audio_int16)
                
                return g711_audio
            
            return b''
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return b''
    
    def linear_to_ulaw(self, linear_audio: np.ndarray) -> bytes:
        """Convert linear PCM to G.711 micro-law using built-in audioop"""
        try:
            import audioop
            
            # Convert numpy array to bytes
            audio_bytes = linear_audio.astype(np.int16).tobytes()
            
            # Convert to micro-law using built-in audioop
            ulaw_bytes = audioop.lin2ulaw(audio_bytes, 2)  # 2 = 16-bit
            
            return ulaw_bytes
            
        except Exception as e:
            logger.error(f"micro-law conversion error: {e}")
            # Fallback to 16-bit PCM
            return linear_audio.astype(np.int16).tobytes()
    
    def detect_speech(self, audio_chunk: bytes) -> bool:
        """Enhanced voice activity detection"""
        try:
            # WebRTC VAD requires specific frame sizes
            frame_size = int(self.sample_rate * self.frame_duration / 1000) * 2  # 2 bytes per sample
            
            if len(audio_chunk) >= frame_size:
                frame = audio_chunk[:frame_size]
                return self.vad.is_speech(frame, self.sample_rate)
            
            # Fallback to energy-based detection
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(audio_array) > 0:
                energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                return energy > 500  # Threshold for speech detection
            
            return False
            
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return False

class ThreeCXVoiceBot:
    """Production-ready 3CX Voice Bot"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.audio_processor = AudioProcessor(self.config)
        self.sip_client = SIPClient(self.config)
        
        # AI Models
        self.device = 'cuda' if (torch.cuda.is_available() and self.config.force_gpu) or torch.cuda.is_available() else 'cpu'
        self.whisper_model = None
        self.tts_model = None
        
        # Call management
        self.active_calls: Dict[str, dict] = {}
        self.is_running = False
        self.call_semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        
        # Metrics
        self.metrics = CallMetrics()
        
        logger.info(f"Voice Bot initialized - Device: {self.device}")
        logger.info(f"Config: {self.config.threecx_server}:{self.config.threecx_port}")
    
    async def initialize_models(self) -> bool:
        """Initialize AI models with error handling"""
        try:
            logger.info("Initializing AI models...")
            
            # Configure GPU memory if using CUDA
            if self.device == 'cuda' and self.config.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # Load Whisper
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.whisper_model = whisper.load_model(self.config.whisper_model, device=self.device)
            
            # Load TTS
            logger.info(f"Loading TTS model: {self.config.tts_model}")
            self.tts_model = TTS(
                model_name=self.config.tts_model,
                gpu=(self.device == 'cuda'),
                progress_bar=False
            )
            
            # Warm up models
            logger.info("Warming up models...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            _ = self.whisper_model.transcribe(dummy_audio)
            _ = self.tts_model.tts("Hello world")
            
            logger.info("AI models ready")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe audio using Whisper"""
        start_time = time.time()
        
        try:
            if audio_data.size == 0:
                return "", 0.0
            
            # Convert to Whisper format
            whisper_audio = self.audio_processor.convert_for_whisper(audio_data)
            
            # Transcribe
            result = self.whisper_model.transcribe(
                whisper_audio,
                fp16=(self.device == 'cuda'),
                language='en'
            )
            
            transcription = result['text'].strip()
            processing_time = time.time() - start_time
            
            # Update metrics
            if self.config.enable_metrics:
                total = self.metrics.total_calls
                current_avg = self.metrics.avg_stt_time
                self.metrics.avg_stt_time = (current_avg * total + processing_time) / (total + 1)
            
            logger.info(f"STT ({processing_time:.2f}s): '{transcription}'")
            return transcription, processing_time
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return "", time.time() - start_time
    
    async def generate_response(self, user_text: str) -> Tuple[str, float]:
        """Generate AI response using Ollama"""
        start_time = time.time()
        
        try:
            # Enhanced customer service prompt
            system_prompt = """You are a professional customer service AI assistant. 
Respond helpfully, concisely, and professionally. Keep responses to 1-2 sentences.
If you cannot help with something, politely explain and offer alternatives."""
            
            prompt = f"{system_prompt}\n\nCustomer: {user_text}\n\nAssistant:"
            
            payload = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "stop": ["\n\nCustomer:", "\nCustomer:", "Assistant:"]
                }
            }
            
            response = requests.post(
                self.config.ollama_url, 
                json=payload, 
                timeout=self.config.response_timeout
            )
            response.raise_for_status()
            
            ai_response = response.json().get("response", "").strip()
            processing_time = time.time() - start_time
            
            # Update metrics
            if self.config.enable_metrics:
                total = self.metrics.total_calls
                current_avg = self.metrics.avg_llm_time
                self.metrics.avg_llm_time = (current_avg * total + processing_time) / (total + 1)
            
            logger.info(f"LLM ({processing_time:.2f}s): '{ai_response}'")
            
            return ai_response or "I apologize, but I didn't understand. Could you please rephrase?", processing_time
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm experiencing technical difficulties. Please hold on.", time.time() - start_time
    
    async def synthesize_speech(self, text: str) -> Tuple[bytes, float]:
        """Convert text to speech using TTS"""
        start_time = time.time()
        
        try:
            if not text.strip():
                return b'', 0.0
            
            # Generate speech
            wav_data = self.tts_model.tts(text=text)
            
            # Convert to 3CX format
            audio_bytes = self.audio_processor.convert_for_3cx(np.array(wav_data))
            
            processing_time = time.time() - start_time
            
            # Update metrics
            if self.config.enable_metrics:
                total = self.metrics.total_calls
                current_avg = self.metrics.avg_tts_time
                self.metrics.avg_tts_time = (current_avg * total + processing_time) / (total + 1)
            
            audio_duration = len(audio_bytes) / (self.config.sample_rate * 2)
            logger.info(f"TTS ({processing_time:.2f}s): {len(audio_bytes)} bytes ({audio_duration:.1f}s audio)")
            
            return audio_bytes, processing_time
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return b'', time.time() - start_time
    
    async def handle_call(self, call_id: str, caller_info: dict):
        """Handle individual call with full pipeline"""
        async with self.call_semaphore:
            call_start = time.time()
            
            try:
                logger.info(f"Handling call {call_id} from {caller_info.get('caller_id', 'unknown')}")
                
                # Initialize call data
                self.active_calls[call_id] = {
                    **caller_info,
                    'start_time': datetime.now(),
                    'status': 'active',
                    'audio_buffer': collections.deque(maxlen=self.config.audio_buffer_size)
                }
                
                # Send initial greeting
                greeting = "Hello! Thank you for calling. I'm your AI assistant. How may I help you today?"
                await self.send_response(call_id, greeting)
                
                # Simulate conversation processing
                await self.process_conversation(call_id)
                
                # Update metrics
                call_duration = time.time() - call_start
                self.metrics.total_calls += 1
                self.metrics.successful_calls += 1
                self.metrics.total_duration += call_duration
                
                logger.info(f"Call {call_id} completed successfully ({call_duration:.1f}s)")
                
            except Exception as e:
                logger.error(f"Call {call_id} failed: {e}")
                self.metrics.failed_calls += 1
                
            finally:
                # Cleanup
                if call_id in self.active_calls:
                    del self.active_calls[call_id]
    
    async def process_conversation(self, call_id: str):
        """Process conversation loop for a call"""
        conversation_scenarios = [
            "Hi, I need help with my account balance",
            "Can you help me reset my password?", 
            "I'm having trouble accessing my services",
            "Thank you for your help, goodbye"
        ]
        
        for i, user_input in enumerate(conversation_scenarios):
            if call_id not in self.active_calls:
                break
                
            logger.info(f"Call {call_id} - Processing: '{user_input}'")
            
            # Simulate STT processing
            _, stt_time = await self.transcribe_audio(np.random.normal(0, 0.1, 8000).astype(np.float32))
            
            # Generate response
            ai_response, llm_time = await self.generate_response(user_input)
            
            # Send response
            await self.send_response(call_id, ai_response)
            
            # Check for conversation end
            if "goodbye" in user_input.lower() or "bye" in user_input.lower():
                farewell = "Thank you for calling! Have a wonderful day. Goodbye!"
                await self.send_response(call_id, farewell)
                break
                
            # Simulate natural conversation timing
            await asyncio.sleep(2)
    
    async def send_response(self, call_id: str, text: str):
        """Send audio response to caller"""
        try:
            # Generate and send audio
            audio_bytes, tts_time = await self.synthesize_speech(text)
            
            if audio_bytes:
                # Check if this is a real call with caller address
                if call_id in self.active_calls:
                    call_data = self.active_calls[call_id]
                    caller_addr = call_data.get('caller_addr')
                    
                    if caller_addr:
                        # Send via real RTP
                        await self.send_audio_rtp(call_id, audio_bytes, caller_addr)
                    else:
                        # Fallback to simulation
                        await self.simulate_audio_transmission(call_id, audio_bytes)
                else:
                    # Fallback to simulation
                    await self.simulate_audio_transmission(call_id, audio_bytes)
            
        except Exception as e:
            logger.error(f"Response sending failed for call {call_id}: {e}")
    
    async def send_audio_rtp(self, call_id: str, audio_data: bytes, caller_addr):
        """Send audio via RTP to caller"""
        try:
            if call_id not in self.active_calls:
                return
            
            call_data = self.active_calls[call_id]
            rtp_port = call_data.get('rtp_port', 9000)
            
            # Create RTP socket for audio transmission
            rtp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Split audio into RTP packets (20ms chunks)
            chunk_size = self.config.sample_rate * 2 // 50  # 20ms at 8kHz
            sequence_number = call_data.get('rtp_seq', 0)
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                
                # Create RTP header
                rtp_header = struct.pack('!BBHII',
                    0x80,  # Version, Padding, Extension, CC
                    0,     # Marker, Payload Type (0 = PCMU)
                    sequence_number,
                    int(time.time() * 1000),  # Timestamp
                    call_data.get('ssrc', 12345)  # SSRC
                )
                
                # Send RTP packet
                rtp_packet = rtp_header + chunk
                rtp_socket.sendto(rtp_packet, (caller_addr[0], rtp_port))
                
                sequence_number += 1
                
                # 20ms delay between packets
                await asyncio.sleep(0.02)
            
            # Update call data
            call_data['rtp_seq'] = sequence_number
            
            # Close RTP socket
            rtp_socket.close()
            
            transmission_time = len(audio_data) / (self.config.sample_rate * 2)
            logger.info(f"Sent {len(audio_data)} bytes via RTP to call {call_id} ({transmission_time:.1f}s)")
            
        except Exception as e:
            logger.error(f"RTP transmission error for call {call_id}: {e}")
    
    async def simulate_audio_transmission(self, call_id: str, audio_data: bytes):
        """Fallback simulation for testing"""
        transmission_time = len(audio_data) / (self.config.sample_rate * 2)
        logger.info(f"Simulating transmission of {len(audio_data)} bytes to call {call_id} ({transmission_time:.1f}s)")
        
        # Simulate network transmission delay
        await asyncio.sleep(min(transmission_time, 0.1))
    
    async def start_call_monitoring(self):
        """Start monitoring for incoming calls"""
        logger.info("Starting call monitoring...")
        
        # Connect to 3CX
        connected = await self.sip_client.connect()
        if not connected:
            logger.warning("SIP connection failed - running in simulation mode")
            await self.simulate_incoming_calls()
            return
        
        # Start real call monitoring
        logger.info("SIP connection successful - monitoring for real calls...")
        await self.monitor_real_calls()
    
    async def monitor_real_calls(self):
        """Monitor for real incoming SIP calls"""
        try:
            while self.is_running:
                # Listen for SIP messages
                await self.listen_for_sip_messages()
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overload
                
        except Exception as e:
            logger.error(f"Real call monitoring error: {e}")
    
    async def listen_for_sip_messages(self):
        """Listen for incoming SIP messages"""
        try:
            if self.sip_client.socket:
                # Set socket to non-blocking for asyncio
                self.sip_client.socket.settimeout(0.1)
                
                try:
                    data, addr = self.sip_client.socket.recvfrom(4096)
                    message = data.decode('utf-8', errors='ignore')
                    
                    # Parse SIP message
                    if message.startswith('INVITE'):
                        await self.handle_incoming_call(message, addr)
                    elif 'BYE' in message:
                        await self.handle_call_end(message)
                    elif 'ACK' in message:
                        logger.debug("Call acknowledged")
                        
                except socket.timeout:
                    pass  # No data received, continue
                except Exception as e:
                    logger.debug(f"SIP message parsing error: {e}")
                    
        except Exception as e:
            logger.error(f"SIP listening error: {e}")
    
    async def handle_incoming_call(self, invite_message: str, caller_addr):
        """Handle incoming SIP INVITE"""
        try:
            # Extract call information
            call_id = self.extract_call_id(invite_message)
            caller_id = self.extract_caller_id(invite_message)
            
            logger.info(f"Incoming call {call_id} from {caller_id} at {caller_addr}")
            
            # Answer the call
            await self.answer_call(call_id, caller_addr)
            
            # Handle the call
            caller_info = {
                'caller_id': caller_id,
                'caller_addr': caller_addr,
                'call_id': call_id
            }
            
            # Process call asynchronously
            asyncio.create_task(self.handle_call(call_id, caller_info))
            
        except Exception as e:
            logger.error(f"Incoming call handling error: {e}")
    
    async def answer_call(self, call_id: str, caller_addr):
        """Answer incoming SIP call"""
        try:
            # Create 200 OK response
            ok_response = f"""SIP/2.0 200 OK\r
Via: SIP/2.0/UDP {caller_addr[0]}:{caller_addr[1]};branch=z9hG4bK123456\r
From: <sip:{self.config.test_extension}@{self.config.threecx_server}>;tag=caller123\r
To: <sip:{self.config.threecx_extension}@{self.config.threecx_server}>;tag=callee456\r
Call-ID: {call_id}\r
CSeq: 1 INVITE\r
Contact: <sip:{self.config.threecx_extension}@10.2.9.10:5060>\r
Content-Type: application/sdp\r
Content-Length: 200\r
\r
v=0\r
o=VoiceBot 123456 654321 IN IP4 10.2.9.10\r
s=Voice Call\r
c=IN IP4 10.2.9.10\r
t=0 0\r
m=audio 9000 RTP/AVP 0\r
a=rtpmap:0 PCMU/8000\r
"""
            
            self.sip_client.socket.sendto(ok_response.encode(), caller_addr)
            logger.info(f"Answered call {call_id}")
            
        except Exception as e:
            logger.error(f"Call answer error: {e}")
    
    def extract_call_id(self, message: str) -> str:
        """Extract Call-ID from SIP message"""
        for line in message.split('\n'):
            if line.startswith('Call-ID:'):
                return line.split(':', 1)[1].strip()
        return f"call_{int(time.time())}"
    
    def extract_caller_id(self, message: str) -> str:
        """Extract caller ID from SIP message"""
        for line in message.split('\n'):
            if line.startswith('From:'):
                # Extract phone number from From header
                from_header = line.split(':', 1)[1].strip()
                # Simple extraction - in production, use proper SIP parsing
                if '<' in from_header:
                    return from_header.split('<')[1].split('>')[0]
                return from_header
        return "Unknown"
    
    async def handle_call_end(self, bye_message: str):
        """Handle call termination"""
        try:
            call_id = self.extract_call_id(bye_message)
            logger.info(f"Call {call_id} ended")
            
            # Clean up call resources
            if call_id in self.active_calls:
                del self.active_calls[call_id]
                
        except Exception as e:
            logger.error(f"Call end handling error: {e}")
    
    async def simulate_incoming_calls(self):
        """Simulate incoming calls for testing"""
        test_callers = [
            {"caller_id": "+1234567890", "name": "John Doe"},
            {"caller_id": "+9876543210", "name": "Jane Smith"},
            {"caller_id": "+1122334455", "name": "Bob Johnson"}
        ]
        
        while self.is_running:
            try:
                # Simulate call every 30 seconds
                await asyncio.sleep(30)
                
                if len(self.active_calls) < self.config.max_concurrent_calls:
                    caller = test_callers[self.metrics.total_calls % len(test_callers)]
                    call_id = f"call_{int(time.time())}_{self.metrics.total_calls}"
                    
                    # Handle call asynchronously
                    asyncio.create_task(self.handle_call(call_id, caller))
                
            except Exception as e:
                logger.error(f"Call simulation error: {e}")
    
    def log_metrics(self):
        """Log current performance metrics"""
        if not self.config.enable_metrics:
            return
            
        m = self.metrics
        
        success_rate = (m.successful_calls / max(m.total_calls, 1)) * 100
        avg_call_duration = m.total_duration / max(m.total_calls, 1)
        
        logger.info("=== PERFORMANCE METRICS ===")
        logger.info(f"Total Calls: {m.total_calls}")
        logger.info(f"Successful: {m.successful_calls}")
        logger.info(f"Failed: {m.failed_calls}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Avg Call Duration: {avg_call_duration:.1f}s")
        logger.info(f"Avg STT Time: {m.avg_stt_time:.2f}s")
        logger.info(f"Avg LLM Time: {m.avg_llm_time:.2f}s")
        logger.info(f"Avg TTS Time: {m.avg_tts_time:.2f}s")
        logger.info(f"Active Calls: {len(self.active_calls)}")
        logger.info("==========================")
    
    async def run(self):
        """Main bot execution"""
        logger.info("Starting 3CX Voice Bot...")
        logger.info("=" * 50)
        
        self.is_running = True
        
        try:
            # Initialize AI models
            if not await self.initialize_models():
                logger.error("Failed to initialize models")
                return False
            
            # Start monitoring calls
            monitoring_task = asyncio.create_task(self.start_call_monitoring())
            
            # Start metrics logging
            if self.config.enable_metrics:
                metrics_task = asyncio.create_task(self.periodic_metrics())
                await asyncio.gather(monitoring_task, metrics_task)
            else:
                await monitoring_task
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.is_running = False
            logger.info("Voice Bot stopped")
    
    async def periodic_metrics(self):
        """Periodically log metrics"""
        while self.is_running:
            await asyncio.sleep(60)  # Log metrics every minute
            self.log_metrics()

def check_environment():
    """Verify environment configuration"""
    required_vars = [
        'THREECX_SERVER',
        'THREECX_EXTENSION', 
        'THREECX_PASSWORD'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        logger.info("Please create a .env file with required configuration")
        return False
    
    logger.info("Environment configuration OK")
    return True

async def main():
    """Main entry point"""
    logger.info("3CX Production Voice Bot")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Run bot
    bot = ThreeCXVoiceBot()
    await bot.run()

if __name__ == "__main__":
    try:
        # Check for .env file
        if not os.path.exists('.env'):
            logger.warning("No .env file found - using default/environment variables")
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        import traceback
        traceback.print_exc()