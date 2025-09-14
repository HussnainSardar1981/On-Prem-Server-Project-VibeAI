#!/usr/bin/env python3
"""
Production 3CX Voice Bot - Fixed Raw Socket Implementation
Uses corrected SIP authentication based on ChatGPT's analysis
"""

import asyncio
import time
import logging
import os
import socket
import re
import hashlib
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import collections

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_bot.log'),
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
        
        self.whisper_model = os.getenv('WHISPER_MODEL', 'base')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'orca2:7b')
        self.tts_model = os.getenv('TTS_MODEL', 'tts_models/en/ljspeech/tacotron2-DDC')
        
        self.sample_rate = int(os.getenv('SAMPLE_RATE', '8000'))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1024'))
        self.vad_aggressiveness = int(os.getenv('VAD_AGGRESSIVENESS', '2'))
        
        self.max_concurrent_calls = int(os.getenv('MAX_CONCURRENT_CALLS', '5'))
        self.response_timeout = int(os.getenv('RESPONSE_TIMEOUT', '30'))
        
        self.force_gpu = os.getenv('FORCE_GPU', 'false').lower() == 'true'
        self.gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
        
        logger.info(f"Configuration loaded - Server: {self.threecx_server}:{self.threecx_port}")
        logger.info(f"Extension: {self.threecx_extension}, Auth ID: {self.threecx_auth_id}")

class SIPClient:
    """Fixed SIP client with proper 3CX authentication"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.socket = None
        self.registered = False
        self.call_id_counter = 1
        self.local_ip = None
        
    def get_local_ip(self):
        """Get the actual local IP address for Via headers"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((self.config.threecx_server, self.config.threecx_port))
                self.local_ip = s.getsockname()[0]
            logger.info(f"Local IP detected: {self.local_ip}")
            return self.local_ip
        except Exception as e:
            logger.warning(f"Could not detect local IP: {e}, using fallback")
            self.local_ip = "10.2.9.10"
            return self.local_ip
    
    async def connect(self):
        """Connect to 3CX SIP server"""
        try:
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
                    logger.info("✅ Successfully registered with 3CX (no auth required)")
                elif "401 Unauthorized" in response_str or "407 Proxy Authentication Required" in response_str:
                    logger.info("🔐 Authentication challenge received")
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
            
            # Decide which header to use based on challenge type
            is_proxy = 'proxy-authenticate:' in challenge_response.lower()
            header_name = 'Proxy-Authorization' if is_proxy else 'Authorization'
            logger.info(f"Using {header_name} header for authentication")
            
            # Create authenticated REGISTER message
            auth_register = self.create_authenticated_register_message(auth_params, auth_response, header_name)
            
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
                    logger.info("✅ Successfully authenticated and registered with 3CX")
                else:
                    logger.error(f"❌ Authentication failed: {response_str[:200]}...")
                    
            except socket.timeout:
                logger.warning("No authentication response received")
                
        except Exception as e:
            logger.error(f"Digest authentication failed: {e}")
    
    def parse_auth_challenge(self, response: str):
        """Properly parse WWW-Authenticate or Proxy-Authenticate header"""
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
        try:
            realm = auth_params.get('realm', '')
            nonce = auth_params.get('nonce', '')
            algorithm = auth_params.get('algorithm', 'MD5').upper()
            
            if algorithm != 'MD5':
                logger.warning(f"Unsupported algorithm: {algorithm}, using MD5")
            
            # FIXED: Use Auth ID instead of extension for digest username
            username = self.config.threecx_auth_id or self.config.threecx_extension
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
    
    def create_authenticated_register_message(self, auth_params, auth_response, header_name='Authorization'):
        """Create authenticated REGISTER message with proper Authorization header"""
        call_id = f"call-{self.call_id_counter}@{self.local_ip}"
        self.call_id_counter += 1
        
        branch_id = f"z9hG4bK{random.randint(10000000, 99999999)}"
        tag_id = f"tag-{random.randint(100000, 999999)}"
        
        # FIXED: Use Auth ID instead of extension for digest username
        username = self.config.threecx_auth_id or self.config.threecx_extension
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
{header_name}: {auth_header}\r
Expires: 3600\r
Content-Length: 0\r
\r
"""
        return register
    
    def create_register_message(self):
        """Create initial SIP REGISTER message"""
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

class ThreeCXVoiceBot:
    """Production-ready 3CX Voice Bot with fixed authentication"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.sip_client = SIPClient(self.config)
        
        # AI Models
        self.device = 'cuda' if (torch.cuda.is_available() and self.config.force_gpu) or torch.cuda.is_available() else 'cpu'
        self.whisper_model = None
        self.tts_model = None
        
        # Call management
        self.active_calls: Dict[str, dict] = {}
        self.is_running = False
        
        # Metrics
        self.metrics = CallMetrics()
        
        logger.info(f"Voice Bot initialized - Device: {self.device}")
    
    async def initialize_models(self) -> bool:
        """Initialize AI models"""
        try:
            logger.info("Initializing AI models...")
            
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
            
            logger.info("✅ AI models ready")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    async def run(self):
        """Main bot execution"""
        logger.info("🚀 Starting 3CX Voice Bot...")
        logger.info("=" * 50)
        
        self.is_running = True
        
        try:
            # Initialize AI models
            if not await self.initialize_models():
                logger.error("Failed to initialize models")
                return False
            
            # Connect to 3CX
            logger.info("📡 Connecting to 3CX SIP server...")
            connected = await self.sip_client.connect()
            
            if connected and self.sip_client.registered:
                logger.info("✅ SIP connection successful - bot is ready!")
                logger.info("🎯 Waiting for incoming calls...")
                
                # Keep running
                while self.is_running:
                    await asyncio.sleep(1)
            else:
                logger.error("❌ SIP connection failed")
                return False
                
        except KeyboardInterrupt:
            logger.info(" Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.is_running = False
            logger.info("Voice Bot stopped")

async def main():
    """Main entry point"""
    logger.info("3CX Production Voice Bot - Fixed Authentication")
    logger.info("=" * 50)
    
    # Run bot
    bot = ThreeCXVoiceBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        import traceback
        traceback.print_exc()
