#!/usr/bin/env python3
"""
Production 3CX Voice Bot - pjsua2 Implementation
Uses proven pjsua2 for SIP communication
"""

import asyncio
import time
import signal
import sys
import logging
import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import collections

# pjsua2 for SIP communication
import pjsua2 as pj

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

class PJSIPClient(pj.Account):
    """pjsua2-based SIP client for 3CX integration"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.registered = False
        
    def onRegState(self, prm):
        """Handle registration state changes"""
        info = self.getInfo()
        if info.regIsActive:
            self.registered = True
            logger.info(f"Successfully registered with 3CX: {info.regIsActive}")
        else:
            self.registered = False
            logger.warning(f"Registration failed: {info.regStatusText}")
    
    def onIncomingCall(self, prm):
        """Handle incoming calls"""
        call = Call(self.config, prm.callId)
        call.answer(200)
        logger.info(f"Incoming call from: {prm.callId}")

class Call(pj.Call):
    """Handle individual calls"""
    
    def __init__(self, config: ConfigManager, call_id: int):
        super().__init__()
        self.config = config
        self.call_id = call_id
        
    def onCallState(self):
        """Handle call state changes"""
        info = self.getInfo()
        logger.info(f"Call {self.call_id} state: {info.stateText}")
        
        if info.state == pj.PJSIP_INV_STATE_CONFIRMED:
            logger.info(f"Call {self.call_id} confirmed - starting conversation")
            # Start conversation processing here
        elif info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            logger.info(f"Call {self.call_id} ended")
    
    def onCallMediaState(self):
        """Handle media state changes"""
        info = self.getInfo()
        for mi in info.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO:
                if mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                    logger.info(f"Audio media active for call {self.call_id}")
                    # Start audio processing here

class ThreeCXVoiceBot:
    """Production-ready 3CX Voice Bot with pjsua2"""
    
    def __init__(self):
        self.config = ConfigManager()
        
        # Initialize pjsua2
        self.ep = pj.Endpoint()
        self.ep.libCreate()
        self.ep.libInit(pj.EpConfig())
        
        # Create UDP transport
        tcfg = pj.TransportConfig()
        tcfg.port = 0  # Let system choose port
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        self.ep.libStart()
        
        # Create SIP account
        self.sip_client = self.create_sip_account()
        
        # AI Models
        self.device = 'cuda' if (torch.cuda.is_available() and self.config.force_gpu) or torch.cuda.is_available() else 'cpu'
        self.whisper_model = None
        self.tts_model = None
        
        # Call management
        self.active_calls: Dict[int, Call] = {}
        self.is_running = False
        
        # Metrics
        self.metrics = CallMetrics()
        
        logger.info(f"Voice Bot initialized - Device: {self.device}")
    
    def create_sip_account(self):
        """Create and configure SIP account"""
        acfg = pj.AccountConfig()
        acfg.idUri = f"sip:{self.config.threecx_extension}@{self.config.threecx_server}"
        acfg.regConfig.registrarUri = f"sip:{self.config.threecx_server}"
        acfg.regConfig.registerOnAdd = True
        
        # Add authentication credentials
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "3CXPhoneSystem", 
                           self.config.threecx_auth_id, 0, 
                           self.config.threecx_password)
        )
        
        # Create account
        acc = PJSIPClient(self.config)
        acc.create(acfg)
        
        return acc
    
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
            
            logger.info("AI models ready")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down voice bot...")
        self.ep.hangupAllCalls()
        self.ep.libDestroy()
        logger.info("Voice bot stopped")
    
    async def run(self):
        """Main bot execution"""
        logger.info("Starting 3CX Voice Bot with pjsua2...")
        logger.info("=" * 50)
        
        self.is_running = True
        
        try:
            # Initialize AI models
            if not await self.initialize_models():
                logger.error("Failed to initialize models")
                return False
            
            # Wait for registration
            logger.info("Waiting for SIP registration...")
            await asyncio.sleep(3)
            
            if self.sip_client.registered:
                logger.info("SIP registration successful - bot is ready!")
                logger.info("Waiting for incoming calls...")
                
                # Keep running
                while self.is_running:
                    await asyncio.sleep(1)
            else:
                logger.error("SIP registration failed")
                return False
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.is_running = False
            self.shutdown()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    logger.info("3CX Production Voice Bot - pjsua2 Version")
    logger.info("=" * 50)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
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
