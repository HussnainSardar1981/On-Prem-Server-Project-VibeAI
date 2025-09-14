#!/usr/bin/env python3
"""
Production 3CX Voice Bot using pjsua2
Complete SIP integration with ALSA Loopback audio routing and AI pipeline
"""

import os
import sys
import time
import logging
import argparse
import asyncio
import threading
import queue
import collections
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json

# Core dependencies
import numpy as np
import torch
import librosa
import webrtcvad
import requests

# AI components
import whisper
from TTS.api import TTS

# Audio processing
import soundfile as sf
import pyaudio

# SIP integration
import pjsua2 as pj

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voicebot_pjsua2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration for 3CX compatibility"""
    sample_rate: int = 8000
    channels: int = 1
    format: int = pyaudio.paInt16
    chunk_size: int = 160  # 20ms at 8kHz
    vad_frame_size: int = 160  # 20ms frames for VAD
    silence_threshold: int = 15  # frames of silence before processing

@dataclass
class SIPConfig:
    """SIP configuration for 3CX"""
    server: str = "mtipbx.ny.3cx.us"
    port: int = 5060
    extension: str = "1600"
    auth_id: str = "qpZh2VS624"
    password: str = "FcHw0P2FHK"
    realm: str = "3CXPhoneSystem"

class AudioDeviceManager:
    """Manages ALSA Loopback device selection and validation"""
    
    def __init__(self, audio_config: AudioConfig):
        self.config = audio_config
        self.capture_dev = None
        self.playback_dev = None
        self.capture_stream = None
        self.playback_stream = None
        self.pyaudio = pyaudio.PyAudio()
        
    def enumerate_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """Enumerate and select ALSA Loopback devices"""
        logger.info("Enumerating audio devices...")
        
        capture_dev = None
        playback_dev = None
        
        device_count = self.pyaudio.get_device_count()
        logger.info(f"Found {device_count} audio devices")
        
        # First pass: find all loopback devices
        loopback_devices = []
        for i in range(device_count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info['name'].lower()
            
            if 'loopback' in name:
                loopback_devices.append((i, info))
                logger.info(f"Device {i}: {info['name']} - Inputs: {info['maxInputChannels']}, Outputs: {info['maxOutputChannels']}")
        
        if not loopback_devices:
            raise RuntimeError("No ALSA Loopback devices found. Ensure snd-aloop is loaded.")
        
        # ALSA Loopback device selection strategy
        logger.info("Selecting loopback devices...")
        
        # Strategy 1: Look for specific hw device patterns (hw:0,0 and hw:0,1)
        for i, info in loopback_devices:
            name = info['name'].lower()
            logger.info(f"Evaluating device {i}: {info['name']}")
            logger.info(f"  - Input channels: {info['maxInputChannels']}")
            logger.info(f"  - Output channels: {info['maxOutputChannels']}")
            
            # Look for hw:0,0 pattern (typically capture)
            if 'hw:0,0' in name and info['maxInputChannels'] > 0 and capture_dev is None:
                capture_dev = i
                logger.info(f"✓ Selected capture device: {info['name']} (hw:0,0)")
            
            # Look for hw:0,1 pattern (typically playback)
            elif 'hw:0,1' in name and info['maxOutputChannels'] > 0 and playback_dev is None:
                playback_dev = i
                logger.info(f"✓ Selected playback device: {info['name']} (hw:0,1)")
        
        # Strategy 2: Fallback to any available devices if specific patterns not found
        if capture_dev is None or playback_dev is None:
            logger.info("Fallback: Using any available loopback devices")
            
            for i, info in loopback_devices:
                name = info['name'].lower()
                
                # Select first available input device
                if info['maxInputChannels'] > 0 and capture_dev is None:
                    capture_dev = i
                    logger.info(f"✓ Fallback capture device: {info['name']}")
                
                # Select first available output device
                if info['maxOutputChannels'] > 0 and playback_dev is None:
                    playback_dev = i
                    logger.info(f"✓ Fallback playback device: {info['name']}")
                
                # If we have both, we can stop
                if capture_dev is not None and playback_dev is not None:
                    break
        
        # CRITICAL FIX: Use 'is None' instead of 'not' to avoid treating device ID 0 as False
        if capture_dev is None or playback_dev is None:
            logger.error(f"Device selection failed!")
            logger.error(f"  - capture_dev: {capture_dev}")
            logger.error(f"  - playback_dev: {playback_dev}")
            logger.error(f"  - Found {len(loopback_devices)} loopback devices")
            for i, (dev_idx, dev_info) in enumerate(loopback_devices):
                logger.error(f"    Device {i}: {dev_info['name']} (idx={dev_idx}) - I:{dev_info['maxInputChannels']}/O:{dev_info['maxOutputChannels']}")
            raise RuntimeError(f"Could not find suitable ALSA Loopback devices. Found {len(loopback_devices)} loopback devices but missing input or output capability.")
        
        # Ensure we're not using the same device for both input and output
        if capture_dev == playback_dev:
            logger.warning("Same device selected for both input and output, this may cause issues")
        
        # Get device names for logging
        capture_info = self.pyaudio.get_device_info_by_index(capture_dev)
        playback_info = self.pyaudio.get_device_info_by_index(playback_dev)
        
        logger.info(f"Final device selection:")
        logger.info(f"  - Capture device: {capture_dev} ({capture_info['name']})")
        logger.info(f"  - Playback device: {playback_dev} ({playback_info['name']})")
        
        return capture_dev, playback_dev
    
    def validate_device_capabilities(self, device_id: int, is_input: bool) -> bool:
        """Validate device supports required audio format"""
        try:
            info = self.pyaudio.get_device_info_by_index(device_id)
            device_name = info['name']
            
            logger.info(f"Validating device {device_id}: {device_name}")
            logger.info(f"  - Max Input Channels: {info['maxInputChannels']}")
            logger.info(f"  - Max Output Channels: {info['maxOutputChannels']}")
            logger.info(f"  - Default Sample Rate: {info['defaultSampleRate']}")
            
            # For loopback devices, use very lenient validation
            if 'loopback' in device_name.lower():
                logger.info(f"  - Loopback device detected, using lenient validation")
                
                # Just check if it has the right direction capability
                if is_input and info['maxInputChannels'] > 0:
                    logger.info(f"  - Input validation passed ({info['maxInputChannels']} channels)")
                    return True
                elif not is_input and info['maxOutputChannels'] > 0:
                    logger.info(f"  - Output validation passed ({info['maxOutputChannels']} channels)")
                    return True
                else:
                    logger.warning(f"  - Wrong direction for loopback device")
                    return False
            
            # For other devices, try to open a test stream
            try:
                logger.info(f"  - Testing stream with {self.config.sample_rate}Hz, {self.config.channels} channel(s)")
                
                stream_params = {
                    'format': self.config.format,
                    'channels': self.config.channels,
                    'rate': self.config.sample_rate,
                    'input': is_input,
                    'output': not is_input,
                    'frames_per_buffer': self.config.chunk_size
                }
                
                if is_input:
                    stream_params['input_device_index'] = device_id
                else:
                    stream_params['output_device_index'] = device_id
                
                test_stream = self.pyaudio.open(**stream_params)
                test_stream.close()
                logger.info(f"  - Stream test passed")
                return True
                
            except Exception as stream_error:
                logger.warning(f"  - Stream test failed: {stream_error}")
                # For loopback devices, this might still work, so don't fail immediately
                if 'loopback' in device_name.lower():
                    logger.info(f"  - Allowing loopback device despite stream test failure")
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Device validation failed for device {device_id}: {e}")
            # For loopback devices, be extra lenient
            if 'loopback' in str(e).lower():
                logger.info(f"  - Allowing loopback device despite validation error")
                return True
            return False
    
    def initialize_streams(self, capture_dev: int, playback_dev: int) -> bool:
        """Initialize audio streams with selected devices"""
        try:
            logger.info(f"Initializing audio streams: capture={capture_dev}, playback={playback_dev}")
            
            # Validate devices first (but be lenient for loopback devices)
            capture_valid = self.validate_device_capabilities(capture_dev, True)
            playback_valid = self.validate_device_capabilities(playback_dev, False)
            
            if not capture_valid:
                logger.warning("Capture device validation failed, but attempting to continue")
            
            if not playback_valid:
                logger.warning("Playback device validation failed, but attempting to continue")
            
            # Initialize capture stream with error handling
            try:
                logger.info("Opening capture stream...")
                self.capture_stream = self.pyaudio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    input_device_index=capture_dev,
                    frames_per_buffer=self.config.chunk_size,
                    # Add some tolerance for problematic devices
                    stream_callback=None
                )
                logger.info("Capture stream opened successfully")
                
            except Exception as e:
                logger.error(f"Failed to open capture stream: {e}")
                # Try without specifying device index (use default)
                logger.info("Retrying capture stream with default device...")
                try:
                    self.capture_stream = self.pyaudio.open(
                        format=self.config.format,
                        channels=self.config.channels,
                        rate=self.config.sample_rate,
                        input=True,
                        frames_per_buffer=self.config.chunk_size
                    )
                    logger.info("Capture stream opened with default device")
                except Exception as e2:
                    logger.error(f"Failed to open capture stream with default device: {e2}")
                    return False
            
            # Initialize playback stream with error handling
            try:
                logger.info("Opening playback stream...")
                self.playback_stream = self.pyaudio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    output=True,
                    output_device_index=playback_dev,
                    frames_per_buffer=self.config.chunk_size
                )
                logger.info("Playback stream opened successfully")
                
            except Exception as e:
                logger.error(f"Failed to open playback stream: {e}")
                # Try without specifying device index (use default)
                logger.info("Retrying playback stream with default device...")
                try:
                    self.playback_stream = self.pyaudio.open(
                        format=self.config.format,
                        channels=self.config.channels,
                        rate=self.config.sample_rate,
                        output=True,
                        frames_per_buffer=self.config.chunk_size
                    )
                    logger.info("Playback stream opened with default device")
                except Exception as e2:
                    logger.error(f"Failed to open playback stream with default device: {e2}")
                    # Clean up capture stream
                    if self.capture_stream:
                        try:
                            self.capture_stream.close()
                        except:
                            pass
                    return False
            
            self.capture_dev = capture_dev
            self.playback_dev = playback_dev
            
            logger.info(f"Audio streams initialized successfully")
            logger.info(f"  - Capture device: {capture_dev}")
            logger.info(f"  - Playback device: {playback_dev}")
            logger.info(f"  - Sample rate: {self.config.sample_rate}Hz")
            logger.info(f"  - Channels: {self.config.channels}")
            logger.info(f"  - Chunk size: {self.config.chunk_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize audio streams: {e}")
            self.cleanup_streams()
            return False
    
    def cleanup_streams(self):
        """Clean up individual streams"""
        try:
            if self.capture_stream:
                self.capture_stream.stop_stream()
                self.capture_stream.close()
                self.capture_stream = None
        except Exception as e:
            logger.error(f"Error cleaning up capture stream: {e}")
        
        try:
            if self.playback_stream:
                self.playback_stream.stop_stream()
                self.playback_stream.close()
                self.playback_stream = None
        except Exception as e:
            logger.error(f"Error cleaning up playback stream: {e}")
    
    def read_audio(self, frames: int) -> bytes:
        """Read audio from capture device"""
        try:
            if not self.capture_stream:
                logger.error("Capture stream not initialized")
                return b''
            
            return self.capture_stream.read(frames, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Audio read error: {e}")
            return b''
    
    def write_audio(self, audio_data: bytes) -> bool:
        """Write audio to playback device"""
        try:
            if not self.playback_stream:
                logger.error("Playback stream not initialized")
                return False
            
            self.playback_stream.write(audio_data)
            return True
        except Exception as e:
            logger.error(f"Audio write error: {e}")
            return False
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            logger.info("Cleaning up audio resources...")
            self.cleanup_streams()
            
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
                
            logger.info("Audio cleanup completed")
            
        except Exception as e:
            logger.error(f"Audio cleanup error: {e}")

class AIPipeline:
    """AI pipeline for STT, LLM, and TTS processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu'
        
        # AI models
        self.whisper_model = None
        self.tts_model = None
        
        # VAD
        self.vad = webrtcvad.Vad(2)
        
        # Performance metrics
        self.stt_times = collections.deque(maxlen=100)
        self.llm_times = collections.deque(maxlen=100)
        self.tts_times = collections.deque(maxlen=100)
        
        logger.info(f"AI Pipeline initialized on device: {self.device}")
    
    def initialize_models(self) -> bool:
        """Initialize AI models"""
        try:
            logger.info("Initializing AI models...")
            
            # Load Whisper
            model_name = self.config.get('whisper_model', 'base')
            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name, device=self.device)
            
            # Load TTS
            tts_model_name = self.config.get('tts_model', 'tts_models/en/ljspeech/tacotron2-DDC')
            logger.info(f"Loading TTS model: {tts_model_name}")
            self.tts_model = TTS(
                model_name=tts_model_name,
                gpu=(self.device == 'cuda'),
                progress_bar=False
            )
            
            # Warm up models
            logger.info("Warming up models...")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            _ = self.whisper_model.transcribe(dummy_audio)
            logger.info("Models warmed up successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"AI model initialization failed: {e}")
            return False
    
    def speech_to_text(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Convert speech to text using Whisper"""
        start_time = time.time()
        
        try:
            if len(audio_data) == 0:
                return "", 0.0
            
            # Resample to 16kHz for Whisper
            if len(audio_data) > 0:
                audio_16k = librosa.resample(
                    audio_data, 
                    orig_sr=8000, 
                    target_sr=16000
                )
                audio_16k = audio_16k.astype(np.float32)
            else:
                return "", 0.0
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_16k, language='en')
            text = result['text'].strip()
            
            processing_time = time.time() - start_time
            self.stt_times.append(processing_time)
            
            logger.info(f"STT ({processing_time:.2f}s): '{text}'")
            return text, processing_time
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            return "", time.time() - start_time
    
    def generate_response(self, user_text: str) -> Tuple[str, float]:
        """Generate AI response using Ollama"""
        start_time = time.time()
        
        try:
            if not user_text.strip():
                return "I didn't catch that. Could you please repeat?", 0.0
            
            # System prompt
            system_prompt = """You are Alexis, a professional IT support assistant for NETOVO. 
            Provide helpful, concise responses to customer inquiries. 
            Be friendly, professional, and solution-oriented. 
            Keep responses under 2 sentences."""
            
            payload = {
                "model": self.config.get('ollama_model', 'orca2:7b'),
                "prompt": f"{system_prompt}\n\nCustomer said: {user_text}\n\nRespond as Alexis:",
                "stream": False
            }
            
            response = requests.post(
                self.config.get('ollama_url', 'http://127.0.0.1:11434/api/generate'),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            bot_response = response.json().get("response", "").strip()
            processing_time = time.time() - start_time
            self.llm_times.append(processing_time)
            
            logger.info(f"LLM ({processing_time:.2f}s): '{bot_response}'")
            return bot_response or "I'm here to help. How can I assist you today?", processing_time
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I'm experiencing some technical difficulties. Please try again.", time.time() - start_time
    
    def text_to_speech(self, text: str) -> Tuple[np.ndarray, float]:
        """Convert text to speech using Coqui TTS"""
        start_time = time.time()
        
        try:
            if not text.strip():
                return np.array([]), 0.0
            
            # Generate speech at 22kHz
            wav = self.tts_model.tts(text=text)
            
            # Convert to 8kHz for 3CX
            wav_8k = librosa.resample(wav, orig_sr=22050, target_sr=8000)
            wav_int16 = (wav_8k * 32767).astype(np.int16)
            
            processing_time = time.time() - start_time
            self.tts_times.append(processing_time)
            
            logger.info(f"TTS ({processing_time:.2f}s): {len(wav_int16)} samples")
            return wav_int16, processing_time
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return np.array([]), time.time() - start_time
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return {
            'avg_stt_time': np.mean(self.stt_times) if self.stt_times else 0.0,
            'avg_llm_time': np.mean(self.llm_times) if self.llm_times else 0.0,
            'avg_tts_time': np.mean(self.tts_times) if self.tts_times else 0.0,
            'total_processing': len(self.stt_times)
        }

class VoiceBotCall(pj.Call):
    """pjsua2 Call implementation for voice bot"""
    
    def __init__(self, acc, call_id, voice_bot):
        pj.Call.__init__(self, acc, call_id)
        self.voice_bot = voice_bot
        self.audio_bridge = None
        self.is_active = False
        
    def onCallState(self):
        """Handle call state changes"""
        ci = self.getInfo()
        logger.info(f"Call state: {ci.stateText} ({ci.state})")
        
        if ci.state == pj.PJSIP_INV_STATE_CALLING:
            logger.info("Incoming call detected")
        elif ci.state == pj.PJSIP_INV_STATE_EARLY:
            logger.info("Call is ringing")
        elif ci.state == pj.PJSIP_INV_STATE_CONNECTING:
            logger.info("Call is connecting")
        elif ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            logger.info("Call is active - starting audio processing")
            self.is_active = True
            self.voice_bot.start_call_processing(self)
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            logger.info("Call ended")
            self.is_active = False
            if self.audio_bridge:
                self.audio_bridge.stop()
    
    def onCallMediaState(self):
        """Handle media state changes"""
        ci = self.getInfo()
        
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                logger.info("Audio media is active - setting up audio bridge")
                self.setup_audio_bridge()
    
    def setup_audio_bridge(self):
        """Setup audio bridge between pjsua2 and ALSA Loopback"""
        try:
            # Get media
            call_media = self.getMedia(0)
            aud_dev_manager = pj.AudioDevManager.instance()
            
            # Get capture and playback media
            cap_media = aud_dev_manager.getCaptureDevMedia()
            pb_media = aud_dev_manager.getPlaybackDevMedia()
            
            # Connect audio streams
            call_media.startTransmit(pb_media)
            cap_media.startTransmit(call_media)
            
            logger.info("Audio bridge established")
            
        except Exception as e:
            logger.error(f"Failed to setup audio bridge: {e}")

class VoiceBotAccount(pj.Account):
    """pjsua2 Account implementation for 3CX registration"""
    
    def __init__(self, voice_bot):
        pj.Account.__init__(self)
        self.voice_bot = voice_bot
    
    def onRegState(self, prm):
        """Handle registration state changes"""
        info = prm.reason
        logger.info(f"Registration state: {info}")
        
        if prm.code == 200:
            logger.info("Successfully registered with 3CX")
            self.voice_bot.registration_success = True
        elif prm.code in [401, 407]:
            logger.warning(f"Authentication required: {prm.code}")
        else:
            logger.error(f"Registration failed: {prm.code} - {info}")

class VoiceBot:
    """Main voice bot class using pjsua2"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sip_config = SIPConfig()
        self.audio_config = AudioConfig()
        
        # Components
        self.endpoint = None
        self.account = None
        self.audio_manager = None
        self.ai_pipeline = None
        
        # State
        self.registration_success = False
        self.is_running = False
        self.active_calls = {}
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.speech_buffer = collections.deque(maxlen=50)
        self.silence_frames = 0
        
        logger.info("Voice Bot initialized")
    
    def initialize_pjsua2(self) -> bool:
        """Initialize pjsua2 endpoint and transport"""
        try:
            logger.info("Initializing pjsua2...")
            
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Configure logging and initialize with robust error handling
            try:
                # Try newer pjsua2 API with EpConfig
                ep_cfg = pj.EpConfig()
                ep_cfg.logConfig.level = 4  # INFO level
                ep_cfg.logConfig.consoleLevel = 4
                self.endpoint.libInit(ep_cfg)
                logger.info("pjsua2 initialized with EpConfig")
            except (AttributeError, TypeError) as e:
                logger.warning(f"EpConfig approach failed: {e}")
                try:
                    # Fallback: try with individual configs
                    ua_cfg = pj.UAConfig()
                    log_cfg = pj.LogConfig()
                    log_cfg.level = 4  # INFO level
                    log_cfg.consoleLevel = 4
                    media_cfg = pj.MediaConfig()
                    self.endpoint.libInit(ua_cfg, log_cfg, media_cfg)
                    logger.info("pjsua2 initialized with separate configs")
                except (AttributeError, TypeError) as e2:
                    logger.warning(f"Separate configs approach failed: {e2}")
                    try:
                        # Final fallback - minimal EpConfig
                        ep_cfg = pj.EpConfig()
                        self.endpoint.libInit(ep_cfg)
                        logger.info("pjsua2 initialized with minimal EpConfig")
                    except Exception as e3:
                        logger.error(f"All pjsua2 initialization attempts failed: {e3}")
                        raise RuntimeError(f"Failed to initialize pjsua2: {e3}")
            
            # Create UDP transport
            tcfg = pj.TransportConfig()
            tcfg.port = self.sip_config.port
            self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
            
            # Start library
            self.endpoint.libStart()
            
            logger.info("pjsua2 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"pjsua2 initialization failed: {e}")
            return False
    
    def setup_audio_devices(self) -> bool:
        """Setup ALSA Loopback audio devices"""
        try:
            logger.info("Setting up audio devices...")
            
            # Initialize audio manager
            self.audio_manager = AudioDeviceManager(self.audio_config)
            
            # Enumerate and select devices
            capture_dev, playback_dev = self.audio_manager.enumerate_devices()
            
            # Configure pjsua2 audio devices BEFORE initializing PyAudio streams
            try:
                aud_dev_manager = pj.Endpoint.instance().audDevManager()
                aud_dev_manager.setCaptureDev(capture_dev)
                aud_dev_manager.setPlaybackDev(playback_dev)
                logger.info(f"pjsua2 audio devices configured - Capture: {capture_dev}, Playback: {playback_dev}")
            except Exception as pj_error:
                logger.warning(f"Failed to configure pjsua2 audio devices: {pj_error}")
                # Continue anyway - PyAudio might still work
            
            # Initialize PyAudio streams AFTER pjsua2 configuration
            if not self.audio_manager.initialize_streams(capture_dev, playback_dev):
                raise RuntimeError("Failed to initialize audio streams")
            
            logger.info(f"Audio devices configured - Capture: {capture_dev}, Playback: {playback_dev}")
            return True
            
        except Exception as e:
            logger.error(f"Audio device setup failed: {e}")
            return False
    
    def register_account(self) -> bool:
        """Register with 3CX SIP server"""
        try:
            logger.info("Registering with 3CX...")
            
            # Create account configuration
            acc_cfg = pj.AccountConfig()
            acc_cfg.idUri = f"sip:{self.sip_config.extension}@{self.sip_config.server}"
            acc_cfg.regConfig.registrarUri = f"sip:{self.sip_config.server}"
            acc_cfg.regConfig.registerOnAdd = True
            
            # Add authentication credentials
            auth_cred = pj.AuthCredInfo()
            auth_cred.scheme = "digest"
            auth_cred.realm = self.sip_config.realm
            auth_cred.username = self.sip_config.auth_id
            auth_cred.dataType = pj.PJSIP_CRED_DATA_PLAIN_PASSWD
            auth_cred.data = self.sip_config.password
            acc_cfg.sipConfig.authCreds.append(auth_cred)
            
            # Create and register account
            self.account = VoiceBotAccount(self)
            self.account.create(acc_cfg)
            
            # Wait for registration
            max_wait = 10
            for _ in range(max_wait):
                if self.registration_success:
                    return True
                time.sleep(1)
            
            logger.error("Registration timeout")
            return False
            
        except Exception as e:
            logger.error(f"Account registration failed: {e}")
            return False
    
    def start_call_processing(self, call: VoiceBotCall):
        """Start audio processing for an active call"""
        try:
            logger.info("Starting call audio processing...")
            
            # Start audio processing thread
            processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True
            )
            processing_thread.start()
            
            # Send greeting
            self._send_greeting()
            
        except Exception as e:
            logger.error(f"Failed to start call processing: {e}")
    
    def _audio_processing_loop(self):
        """Main audio processing loop"""
        try:
            while self.is_running:
                # Read audio from ALSA Loopback
                audio_data = self.audio_manager.read_audio(self.audio_config.chunk_size)
                
                if not audio_data:
                    time.sleep(0.01)
                    continue
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                # Voice Activity Detection
                if self._detect_speech(audio_array):
                    self.speech_buffer.extend(audio_float)
                    self.silence_frames = 0
                else:
                    self.silence_frames += 1
                
                # Process speech when silence detected
                if (self.silence_frames >= self.audio_config.silence_threshold and 
                    len(self.speech_buffer) > 0):
                    self._process_speech()
                
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Audio processing loop error: {e}")
    
    def _detect_speech(self, audio_data: np.ndarray) -> bool:
        """Detect speech using WebRTC VAD"""
        try:
            # Convert to 16kHz for VAD
            if len(audio_data) > 0:
                audio_16k = librosa.resample(
                    audio_data.astype(np.float32) / 32767.0,
                    orig_sr=8000,
                    target_sr=16000
                )
                audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
                
                # VAD requires exactly 160 samples at 16kHz (10ms)
                if len(audio_16k_int16) >= 160:
                    return self.ai_pipeline.vad.is_speech(
                        audio_16k_int16[:160].tobytes(), 16000
                    )
            
            return False
            
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return False
    
    def _process_speech(self):
        """Process accumulated speech through AI pipeline"""
        try:
            if len(self.speech_buffer) == 0:
                return
            
            # Convert buffer to numpy array
            audio_data = np.array(list(self.speech_buffer))
            self.speech_buffer.clear()
            self.silence_frames = 0
            
            # Process through AI pipeline
            user_text, stt_time = self.ai_pipeline.speech_to_text(audio_data)
            
            if user_text:
                # Generate response
                bot_response, llm_time = self.ai_pipeline.generate_response(user_text)
                
                # Convert to speech
                response_audio, tts_time = self.ai_pipeline.text_to_speech(bot_response)
                
                if len(response_audio) > 0:
                    # Play response through ALSA Loopback
                    self._play_response(response_audio)
                    
                    total_time = stt_time + llm_time + tts_time
                    logger.info(f"Turn completed in {total_time:.2f}s (STT: {stt_time:.2f}s, LLM: {llm_time:.2f}s, TTS: {tts_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
    
    def _play_response(self, audio_data: np.ndarray):
        """Play response audio through ALSA Loopback"""
        try:
            # Convert to bytes and chunk for streaming
            audio_bytes = audio_data.astype(np.int16).tobytes()
            chunk_size = self.audio_config.chunk_size * 2  # 2 bytes per sample
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                self.audio_manager.write_audio(chunk)
                time.sleep(0.02)  # 20ms delay between chunks
                
        except Exception as e:
            logger.error(f"Response playback error: {e}")
    
    def _send_greeting(self):
        """Send initial greeting to caller"""
        try:
            greeting = "Hello! This is Alexis, your NETOVO IT support assistant. How can I help you today?"
            response_audio, _ = self.ai_pipeline.text_to_speech(greeting)
            
            if len(response_audio) > 0:
                self._play_response(response_audio)
                logger.info("Greeting sent to caller")
                
        except Exception as e:
            logger.error(f"Failed to send greeting: {e}")
    
    def make_test_call(self, extension: str = "*777"):
        """Make a test call to echo service"""
        try:
            logger.info(f"Making test call to {extension}...")
            
            call_uri = f"sip:{extension}@{self.sip_config.server}"
            call = self.endpoint.utilVerifySipUri(call_uri)
            
            if call:
                call_prm = pj.CallOpParam()
                call_prm.opt.audioCount = 1
                call_prm.opt.videoCount = 0
                
                new_call = VoiceBotCall(self.account, pj.PJSUA_INVALID_ID, self)
                new_call.makeCall(call_uri, call_prm)
                
                logger.info(f"Test call initiated to {extension}")
                return True
            else:
                logger.error(f"Invalid SIP URI: {call_uri}")
                return False
                
        except Exception as e:
            logger.error(f"Test call failed: {e}")
            return False
    
    def run(self, dry_run: bool = False) -> bool:
        """Main run method"""
        try:
            logger.info("Starting Voice Bot...")
            
            # Initialize pjsua2
            if not self.initialize_pjsua2():
                return False
            
            # Setup audio devices
            if not self.setup_audio_devices():
                return False
            
            # Initialize AI pipeline
            self.ai_pipeline = AIPipeline(self.config)
            if not self.ai_pipeline.initialize_models():
                return False
            
            if dry_run:
                logger.info("Dry run completed successfully - all components initialized")
                return True
            
            # Register with 3CX
            if not self.register_account():
                return False
            
            # Set call handler
            self.endpoint.setCallHandler(VoiceBotCall)
            
            self.is_running = True
            logger.info("Voice Bot is running and ready for calls")
            logger.info(f"Extension: {self.sip_config.extension}")
            logger.info(f"Server: {self.sip_config.server}")
            
            # Keep running
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
            
            return True
            
        except Exception as e:
            logger.error(f"Voice Bot run error: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up resources...")
            
            self.is_running = False
            
            # Hangup all calls
            if self.endpoint:
                self.endpoint.hangupAllCalls()
                time.sleep(1)
                self.endpoint.libDestroy()
            
            # Cleanup audio
            if self.audio_manager:
                self.audio_manager.cleanup()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment"""
    return {
        'use_gpu': os.getenv('USE_GPU', 'true').lower() == 'true',
        'whisper_model': os.getenv('WHISPER_MODEL', 'base'),
        'tts_model': os.getenv('TTS_MODEL', 'tts_models/en/ljspeech/tacotron2-DDC'),
        'ollama_model': os.getenv('OLLAMA_MODEL', 'orca2:7b'),
        'ollama_url': os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO')
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='3CX Voice Bot with pjsua2')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Initialize components without starting call processing')
    parser.add_argument('--test-call', action='store_true',
                       help='Make a test call to echo service')
    parser.add_argument('--extension', default='*777',
                       help='Extension to call for test (default: *777)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create and run voice bot
    voice_bot = VoiceBot(config)
    
    if args.dry_run:
        success = voice_bot.run(dry_run=True)
    elif args.test_call:
        success = voice_bot.run(dry_run=True)
        if success:
            voice_bot.make_test_call(args.extension)
    else:
        success = voice_bot.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
