#!/usr/bin/env python3
"""
3CX AI Voice Bot - Production PJSUA2 Implementation
Milestone 2: Complete SIP + AI pipeline integration
"""

import os
import sys
import time
import threading
import queue
import signal
import logging
import argparse
import subprocess
from typing import Optional, Tuple, List
from dataclasses import dataclass
from dotenv import load_dotenv

import pjsua2 as pj
import alsaaudio
import numpy as np
import webrtcvad
import whisper
import requests
from TTS.api import TTS
import librosa

# Load environment variables
load_dotenv()

# Configuration
@dataclass
class Config:
    # 3CX Settings
    server: str = os.getenv('THREECX_SERVER', 'mtipbx.ny.3cx.us')
    port: int = int(os.getenv('THREECX_PORT', '5060'))
    extension: str = os.getenv('THREECX_EXTENSION', '1600')
    password: str = os.getenv('THREECX_PASSWORD', '')
    auth_id: str = os.getenv('THREECX_AUTH_ID', '')
    realm: str = "3CXPhoneSystem"
    
    # Audio Settings
    sample_rate: int = 8000
    channels: int = 1
    frame_size: int = 160  # 20ms at 8kHz
    vad_aggressiveness: int = int(os.getenv('VAD_AGGRESSIVENESS', '2'))
    
    # AI Models
    whisper_model: str = os.getenv('WHISPER_MODEL', 'base')
    ollama_url: str = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
    ollama_model: str = os.getenv('OLLAMA_MODEL', 'orca2:7b')
    tts_model: str = os.getenv('TTS_MODEL', 'tts_models/en/ljspeech/tacotron2-DDC')
    
    # Echo test
    echo_extension: str = os.getenv('ECHO_EXTENSION', '*777')

# Logging setup
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles ALSA audio I/O and AI pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.capture_device = None
        self.playback_device = None
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.whisper_model = None
        self.tts = None
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.silence_frames = 0
        self.silence_threshold = 8  # ~160ms at 20ms frames
        self.is_speaking = False
        self.running = False
        
    def initialize_ai_models(self):
        """Initialize AI models"""
        try:
            logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.whisper_model = whisper.load_model(self.config.whisper_model)
            
            logger.info(f"Loading TTS model: {self.config.tts_model}")
            self.tts = TTS(self.config.tts_model)
            
            logger.info("AI models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            raise
            
    def setup_alsa_devices(self, capture_dev: str, playback_dev: str):
        """Setup ALSA capture and playback devices"""
        try:
            # Capture device (PBX → AI)
            self.capture_device = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE, 
                alsaaudio.PCM_NONBLOCK,
                device=capture_dev
            )
            self.capture_device.setchannels(self.config.channels)
            self.capture_device.setrate(self.config.sample_rate)
            self.capture_device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self.capture_device.setperiodsize(self.config.frame_size)
            
            # Playback device (AI → PBX)
            self.playback_device = alsaaudio.PCM(
                alsaaudio.PCM_PLAYBACK,
                alsaaudio.PCM_NONBLOCK,
                device=playback_dev
            )
            self.playback_device.setchannels(self.config.channels)
            self.playback_device.setrate(self.config.sample_rate)
            self.playback_device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
            self.playback_device.setperiodsize(self.config.frame_size)
            
            logger.info(f"ALSA devices configured: capture={capture_dev}, playback={playback_dev}")
            
        except Exception as e:
            logger.error(f"Failed to setup ALSA devices: {e}")
            raise
            
    def start_audio_processing(self):
        """Start audio processing thread"""
        self.running = True
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        logger.info("Audio processing started")
        
    def stop_audio_processing(self):
        """Stop audio processing"""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2)
        logger.info("Audio processing stopped")
        
    def _audio_loop(self):
        """Main audio processing loop"""
        while self.running:
            try:
                # Read audio from capture device
                length, data = self.capture_device.read()
                
                if length > 0:
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Ensure we have exactly 160 samples for VAD
                    if len(audio_data) == self.config.frame_size:
                        self._process_audio_frame(audio_data)
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
                
    def _process_audio_frame(self, audio_frame: np.ndarray):
        """Process a single audio frame with VAD"""
        try:
            # VAD requires bytes
            frame_bytes = audio_frame.tobytes()
            is_speech = self.vad.is_speech(frame_bytes, self.config.sample_rate)
            
            if is_speech:
                self.speech_buffer.append(audio_frame)
                self.silence_frames = 0
                if not self.is_speaking:
                    self.is_speaking = True
                    logger.debug("Speech detected")
            else:
                if self.is_speaking:
                    self.silence_frames += 1
                    if self.silence_frames >= self.silence_threshold:
                        # End of speech detected
                        self._process_speech()
                        self.is_speaking = False
                        self.speech_buffer = []
                        self.silence_frames = 0
                        
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            
    def _process_speech(self):
        """Process collected speech through AI pipeline"""
        if not self.speech_buffer:
            return
            
        start_time = time.time()
        
        try:
            # Concatenate speech buffer
            speech_audio = np.concatenate(self.speech_buffer)
            logger.debug(f"Processing speech: {len(speech_audio)} samples")
            
            # STT: Upsample to 16kHz for Whisper
            speech_16k = librosa.resample(
                speech_audio.astype(np.float32) / 32768.0,
                orig_sr=self.config.sample_rate,
                target_sr=16000
            )
            
            # Whisper transcription
            stt_start = time.time()
            result = self.whisper_model.transcribe(speech_16k)
            text = result['text'].strip()
            stt_time = time.time() - stt_start
            
            if not text:
                logger.debug("Empty transcription, skipping")
                return
                
            logger.info(f"STT ({stt_time:.2f}s): {text}")
            
            # LLM processing
            llm_start = time.time()
            response = self._get_llm_response(text)
            llm_time = time.time() - llm_start
            
            if not response:
                logger.warning("Empty LLM response")
                return
                
            logger.info(f"LLM ({llm_time:.2f}s): {response}")
            
            # TTS generation
            tts_start = time.time()
            self._generate_and_play_tts(response)
            tts_time = time.time() - tts_start
            
            total_time = time.time() - start_time
            logger.info(f"Turn completed - STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s, Total: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            
    def _get_llm_response(self, text: str) -> str:
        """Get response from Ollama LLM"""
        try:
            prompt = f"You are a helpful AI assistant. Give a brief, conversational response (2-3 sentences max) to: {text}"
            
            payload = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 100  # Limit response length
                }
            }
            
            response = requests.post(self.config.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"LLM request error: {e}")
            return "I'm sorry, I didn't catch that. Could you please repeat?"
            
    def _generate_and_play_tts(self, text: str):
        """Generate TTS and play through ALSA"""
        try:
            # Generate TTS audio
            tts_audio = self.tts.tts(text)
            
            # Convert to numpy array and resample to 8kHz
            if isinstance(tts_audio, list):
                tts_audio = np.array(tts_audio, dtype=np.float32)
                
            # Resample to 8kHz
            tts_8k = librosa.resample(tts_audio, orig_sr=22050, target_sr=self.config.sample_rate)
            
            # Convert to int16
            tts_int16 = (tts_8k * 32767).astype(np.int16)
            
            # Play in chunks of 160 samples
            chunk_size = self.config.frame_size
            for i in range(0, len(tts_int16), chunk_size):
                chunk = tts_int16[i:i+chunk_size]
                
                # Pad last chunk if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Write to playback device
                self.playback_device.write(chunk.tobytes())
                time.sleep(0.02)  # 20ms delay to maintain pacing
                
        except Exception as e:
            logger.error(f"TTS generation/playback error: {e}")
            
    def play_greeting(self):
        """Play initial greeting"""
        greeting = "Hello! I'm your AI assistant. How can I help you today?"
        self._generate_and_play_tts(greeting)

class VoiceBotAccount(pj.Account):
    """Custom Account class for handling SIP events"""
    
    def __init__(self, voicebot):
        pj.Account.__init__(self)
        self.voicebot = voicebot
        
    def onRegState(self, prm):
        logger.info(f"Registration state: {prm.code} - {prm.reason}")
        
    def onIncomingCall(self, prm):
        call = VoiceBotCall(self.voicebot, self, prm.callId)
        call_info = call.getInfo()
        logger.info(f"Incoming call from: {call_info.remoteUri}")
        
        # Auto-answer
        call_prm = pj.CallOpParam()
        call_prm.statusCode = 200
        try:
            call.answer(call_prm)
            logger.info("Call answered automatically")
        except Exception as e:
            logger.error(f"Failed to answer call: {e}")

class VoiceBotCall(pj.Call):
    """Custom Call class for handling call events"""
    
    def __init__(self, voicebot, account, call_id=pj.PJSUA_INVALID_ID):
        pj.Call.__init__(self, account, call_id)
        self.voicebot = voicebot
        self.media_connected = False
        
    def onCallState(self, prm):
        call_info = self.getInfo()
        state_name = {
            pj.PJSIP_INV_STATE_CALLING: "CALLING",
            pj.PJSIP_INV_STATE_INCOMING: "INCOMING", 
            pj.PJSIP_INV_STATE_EARLY: "EARLY",
            pj.PJSIP_INV_STATE_CONNECTING: "CONNECTING",
            pj.PJSIP_INV_STATE_CONFIRMED: "CONFIRMED",
            pj.PJSIP_INV_STATE_DISCONNECTED: "DISCONNECTED"
        }.get(call_info.state, f"UNKNOWN({call_info.state})")
        
        logger.info(f"Call state changed to: {state_name}")
        
        if call_info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self.voicebot.audio_processor.stop_audio_processing()
            
    def onCallMediaState(self, prm):
        call_info = self.getInfo()
        logger.info(f"Media state: active={call_info.media[0].status}")
        
        if (call_info.media[0].status == pj.PJSUA_CALL_MEDIA_ACTIVE and 
            not self.media_connected):
            
            try:
                # Get media objects
                call_media = self.getMedia(0)
                aud_dev_mgr = pj.Endpoint.instance().audDevManager()
                
                # Get capture and playback media ports
                capture_port = aud_dev_mgr.getCaptureDevMedia()
                playback_port = aud_dev_mgr.getPlaybackDevMedia()
                
                # Connect audio paths
                # Call media to playback (RTP → Loopback for AI to capture)
                call_media.startTransmit(playback_port)
                # Capture to call media (AI TTS → RTP)
                capture_port.startTransmit(call_media)
                
                logger.info("Audio media connected successfully")
                self.media_connected = True
                
                # Start AI processing and play greeting
                self.voicebot.audio_processor.start_audio_processing()
                threading.Timer(1.0, self.voicebot.audio_processor.play_greeting).start()
                
            except Exception as e:
                logger.error(f"Failed to connect media: {e}")

class VoiceBot:
    """Main VoiceBot class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.endpoint = None
        self.account = None
        self.transport = None
        self.audio_processor = AudioProcessor(config)
        self.running = False
        
    def find_loopback_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """Find ALSA Loopback devices in pjsua2 device list"""
        try:
            aud_dev_mgr = self.endpoint.audDevManager()
            device_count = aud_dev_mgr.getDevCount()
            
            capture_dev_id = None
            playback_dev_id = None
            
            logger.info("Available audio devices:")
            for i in range(device_count):
                dev_info = aud_dev_mgr.getDevInfo(i)
                logger.info(f"  {i}: {dev_info.name} (in:{dev_info.inputCount}, out:{dev_info.outputCount})")
                
                if "Loopback" in dev_info.name:
                    if dev_info.inputCount > 0 and capture_dev_id is None:
                        capture_dev_id = i
                    if dev_info.outputCount > 0 and playback_dev_id is None:
                        playback_dev_id = i
                        
            if capture_dev_id is None or playback_dev_id is None:
                logger.error("Could not find suitable Loopback devices")
                return None, None
                
            # Get device info for logging
            cap_info = aud_dev_mgr.getDevInfo(capture_dev_id)
            pb_info = aud_dev_mgr.getDevInfo(playback_dev_id)
            
            logger.info(f"Selected devices - Capture: {cap_info.name}, Playback: {pb_info.name}")
            
            return capture_dev_id, playback_dev_id
            
        except Exception as e:
            logger.error(f"Error finding Loopback devices: {e}")
            return None, None
            
    def initialize_pjsua2(self, dry_run: bool = False) -> bool:
        """Initialize PJSUA2 endpoint and account"""
        try:
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Initialize endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 3
            ep_cfg.logConfig.consoleLevel = 3
            
            # Media config
            ep_cfg.medConfig.clockRate = self.config.sample_rate
            ep_cfg.medConfig.audioFramePtime = 20  # 20ms frames
            
            # Echo cancellation (optional - may not be available in all builds)
            try:
                ep_cfg.medConfig.ecOptions = pj.PJMEDIA_ECHO_CANCEL
                logger.info("Echo cancellation enabled")
            except AttributeError:
                logger.warning("Echo cancellation not available in this pjsua2 build - continuing without it")
                ep_cfg.medConfig.ecOptions = 0
            
            self.endpoint.libInit(ep_cfg)
            
            # Find and configure audio devices
            capture_id, playback_id = self.find_loopback_devices()
            if capture_id is None or playback_id is None:
                logger.error("Failed to find Loopback devices - check snd-aloop module")
                return False
                
            # Set audio devices in pjsua2
            aud_dev_mgr = self.endpoint.audDevManager()
            aud_dev_mgr.setCaptureDevId(capture_id)
            aud_dev_mgr.setPlaybackDevId(playback_id)
            
            # Setup ALSA devices for direct I/O
            # Note: We use opposite mapping for direct ALSA access
            # pjsua2 playback goes to Loopback,1,0 so we capture from there
            # pjsua2 capture comes from Loopback,0,0 so we write TTS there
            self.audio_processor.setup_alsa_devices("hw:Loopback,1,0", "hw:Loopback,0,0")
            
            if not dry_run:
                # Initialize AI models
                self.audio_processor.initialize_ai_models()
            
            # Create UDP transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = 0  # Any available port
            self.transport = self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
            
            # Start library
            self.endpoint.libStart()
            logger.info("PJSUA2 initialized successfully")
            
            if dry_run:
                logger.info("Dry run mode - skipping account registration")
                return True
            
            # Create and configure account
            self.account = VoiceBotAccount(self)
            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{self.config.extension}@{self.config.server}"
            acc_cfg.regConfig.registrarUri = f"sip:{self.config.server}:{self.config.port}"
            
            # Authentication
            cred = pj.AuthCredInfo()
            cred.scheme = "digest"
            cred.realm = self.config.realm
            cred.username = self.config.auth_id
            cred.data = self.config.password
            acc_cfg.sipConfig.authCreds.append(cred)
            
            # Create account
            self.account.create(acc_cfg)
            
            logger.info(f"Account created for {self.config.extension}@{self.config.server}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PJSUA2: {e}")
            return False
            
    def start(self, dry_run: bool = False):
        """Start the voice bot"""
        logger.info("Starting 3CX Voice Bot...")
        
        if not self.initialize_pjsua2(dry_run):
            logger.error("Failed to initialize - exiting")
            return False
            
        if dry_run:
            logger.info("Dry run completed successfully")
            self.shutdown()
            return True
            
        self.running = True
        logger.info("Voice bot started successfully")
        logger.info(f"Registered as extension {self.config.extension}")
        logger.info("Waiting for incoming calls...")
        
        return True
        
    def make_echo_test_call(self):
        """Make a test call to echo service"""
        if not self.account:
            logger.error("No account available for outbound call")
            return
            
        try:
            call = VoiceBotCall(self, self.account)
            call_prm = pj.CallOpParam()
            call_prm.opt.audioCount = 1
            call_prm.opt.videoCount = 0
            
            dest_uri = f"sip:{self.config.echo_extension}@{self.config.server}"
            call.makeCall(dest_uri, call_prm)
            logger.info(f"Echo test call initiated to {dest_uri}")
            
        except Exception as e:
            logger.error(f"Failed to make echo test call: {e}")
            
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down voice bot...")
        self.running = False
        
        try:
            if self.audio_processor:
                self.audio_processor.stop_audio_processing()
                
            if self.endpoint:
                # Hangup all calls
                self.endpoint.hangupAllCalls()
                time.sleep(2)  # Give time for cleanup
                
                # Destroy library
                self.endpoint.libDestroy()
                
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            
        logger.info("Voice bot stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    global bot
    if bot:
        bot.shutdown()
    sys.exit(0)

def validate_environment():
    """Validate required environment variables"""
    required_vars = ['THREECX_PASSWORD', 'THREECX_AUTH_ID']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please check your .env file")
        return False
        
    return True

def check_loopback_module():
    """Check if snd-aloop module is loaded"""
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'snd_aloop' not in result.stdout:
            logger.error("snd-aloop module not loaded")
            logger.error("Run: sudo modprobe snd-aloop")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check snd-aloop module: {e}")
        return True  # Assume it's okay

def main():
    parser = argparse.ArgumentParser(description='3CX AI Voice Bot')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Initialize and validate environment without starting AI processing')
    parser.add_argument('--echo-test', action='store_true',
                       help='Make a test call to echo service (*777)')
    args = parser.parse_args()
    
    # Validate environment
    if not validate_environment():
        return 1
        
    if not check_loopback_module():
        return 1
    
    # Create config and bot
    config = Config()
    global bot
    bot = VoiceBot(config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the bot
        if not bot.start(dry_run=args.dry_run):
            return 1
            
        if args.dry_run:
            return 0
            
        # Handle echo test
        if args.echo_test:
            time.sleep(3)  # Wait for registration
            bot.make_echo_test_call()
        
        # Keep running
        while bot.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        bot.shutdown()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
