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
import signal
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
    """SIP configuration for 3CX - loads from environment with validation"""
    def __init__(self):
        # Require all SIP credentials from environment - no defaults
        required_vars = ['SIP_SERVER', 'SIP_EXTENSION', 'SIP_AUTH_ID', 'SIP_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Please set all required SIP credentials in your .env file")
            sys.exit(f"Missing SIP environment variables: {missing_vars}")
        
        self.server: str = os.getenv('SIP_SERVER')
        self.port: int = int(os.getenv('SIP_PORT', '5060'))
        self.extension: str = os.getenv('SIP_EXTENSION')
        self.auth_id: str = os.getenv('SIP_AUTH_ID')
        self.password: str = os.getenv('SIP_PASSWORD')
        self.realm: str = os.getenv('SIP_REALM', '3CXPhoneSystem')

class AudioDeviceManager:
    """Manages ALSA Loopback device selection and validation"""
    
    def __init__(self, audio_config: AudioConfig):
        logger.info("AudioDeviceManager.__init__: Starting initialization...")
        self.config = audio_config
        self.capture_dev = None
        self.playback_dev = None
        self.capture_stream = None
        self.playback_stream = None
        
        logger.info("AudioDeviceManager.__init__: Creating PyAudio instance...")
        try:
            self.pyaudio = pyaudio.PyAudio()
            logger.info("AudioDeviceManager.__init__: PyAudio instance created successfully")
        except Exception as e:
            logger.error(f"AudioDeviceManager.__init__: Failed to create PyAudio instance: {e}")
            raise
        logger.info("AudioDeviceManager.__init__: Initialization completed")
    
    def _pick_pjsua2_loopback(self):
        """Pick loopback devices from pjsua2's device list"""
        try:
            adm = pj.Endpoint.instance().audDevManager()
            n = adm.getDevCount()
            cap_id = pb_id = None
            cap_name = pb_name = None
            
            logger.info(f"Scanning {n} pjsua2 devices for loopback...")
            
            for i in range(n):
                di = adm.getDevInfo(i)
                name = (di.name or "").lower()
                driver = (di.driver or "").lower()
                
                logger.info(f"pjsua2 device {i}: '{di.name}' driver='{di.driver}' in={di.inputCount} out={di.outputCount}")
                
                if ("loopback" in name or "alsa" in driver):
                    if di.inputCount > 0 and cap_id is None:
                        cap_id, cap_name = i, di.name
                        logger.info(f"Selected pjsua2 capture: {i} ({di.name})")
                    if di.outputCount > 0 and pb_id is None:
                        pb_id, pb_name = i, di.name
                        logger.info(f"Selected pjsua2 playback: {i} ({di.name})")
            
            return (cap_id, cap_name), (pb_id, pb_name)
            
        except Exception as e:
            logger.error(f"Failed to pick pjsua2 loopback devices: {e}")
            return (None, None), (None, None)
    
    def _pyaudio_index_by_name(self, friendly_name, want_input):
        """Find PyAudio device index by matching name"""
        if not friendly_name:
            return None
            
        friendly = friendly_name.lower()
        logger.info(f"Looking for PyAudio device matching '{friendly_name}' (input={want_input})")
        
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            name = (info["name"] or "").lower()
            
            if friendly in name or name in friendly:
                if want_input and info["maxInputChannels"] > 0:
                    logger.info(f"Found PyAudio input match: {i} ({info['name']})")
                    return i
                if not want_input and info["maxOutputChannels"] > 0:
                    logger.info(f"Found PyAudio output match: {i} ({info['name']})")
                    return i
        
        logger.warning(f"No PyAudio device found matching '{friendly_name}'")
        return None
        
    def enumerate_devices(self) -> Tuple[Optional[int], Optional[int]]:
        """Enumerate and select ALSA Loopback devices"""
        logger.info("=== ENUMERATE_DEVICES METHOD STARTED ===")
        logger.info("Enumerating audio devices...")
        
        capture_dev = None
        playback_dev = None
        
        logger.info("Getting device count from PyAudio...")
        try:
            device_count = self.pyaudio.get_device_count()
            logger.info(f"Found {device_count} audio devices")
        except Exception as e:
            logger.error(f"Failed to get device count: {e}")
            raise
        
        # First pass: find all loopback devices
        loopback_devices = []
        logger.info("=== DEVICE ENUMERATION DEBUG ===")
        for i in range(device_count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info['name'].lower()
            
            logger.info(f"Device {i}: '{info['name']}'")
            logger.info(f"  - Name (lower): '{name}'")
            logger.info(f"  - Input channels: {info['maxInputChannels']}")
            logger.info(f"  - Output channels: {info['maxOutputChannels']}")
            logger.info(f"  - Default sample rate: {info['defaultSampleRate']}")
            logger.info(f"  - Contains 'loopback': {'loopback' in name}")
            
            if 'loopback' in name:
                loopback_devices.append((i, info))
                logger.info(f"  *** ADDED TO LOOPBACK LIST ***")
            logger.info("  ---")
        
        logger.info(f"=== FOUND {len(loopback_devices)} LOOPBACK DEVICES ===")
        
        if not loopback_devices:
            raise RuntimeError("No ALSA Loopback devices found. Ensure snd-aloop is loaded.")
        
        # ALSA Loopback device selection strategy
        logger.info("Selecting loopback devices...")
        
        # Strategy 1: Look for specific hw device patterns (hw:0,0 and hw:0,1)
        logger.info("=== STRATEGY 1: Looking for hw:0,0 and hw:0,1 patterns ===")
        for i, info in loopback_devices:
            name = info['name'].lower()
            logger.info(f"Evaluating device {i}: {info['name']}")
            logger.info(f"  - Name (lower): '{name}'")
            logger.info(f"  - Input channels: {info['maxInputChannels']}")
            logger.info(f"  - Output channels: {info['maxOutputChannels']}")
            logger.info(f"  - Contains 'hw:0,0': {'hw:0,0' in name}")
            logger.info(f"  - Contains 'hw:0,1': {'hw:0,1' in name}")
            logger.info(f"  - Current capture_dev: {capture_dev}")
            logger.info(f"  - Current playback_dev: {playback_dev}")
            
            # Look for hw:0,0 pattern (typically capture)
            if 'hw:0,0' in name and info['maxInputChannels'] > 0 and capture_dev is None:
                capture_dev = i
                logger.info(f"✓ Selected capture device: {info['name']} (hw:0,0)")
            
            # Look for hw:0,1 pattern (typically playback)
            elif 'hw:0,1' in name and info['maxOutputChannels'] > 0 and playback_dev is None:
                playback_dev = i
                logger.info(f"✓ Selected playback device: {info['name']} (hw:0,1)")
            
            logger.info(f"  - After evaluation - capture_dev: {capture_dev}, playback_dev: {playback_dev}")
            logger.info("  ---")
        
        # Strategy 2: Fallback to any available devices if specific patterns not found
        if capture_dev is None or playback_dev is None:
            logger.info("=== STRATEGY 2: Fallback to any available devices ===")
            logger.info(f"Before fallback - capture_dev: {capture_dev}, playback_dev: {playback_dev}")
            
            for i, info in loopback_devices:
                name = info['name'].lower()
                logger.info(f"Fallback evaluating device {i}: {info['name']}")
                logger.info(f"  - Input channels: {info['maxInputChannels']}")
                logger.info(f"  - Output channels: {info['maxOutputChannels']}")
                logger.info(f"  - Current capture_dev: {capture_dev}")
                logger.info(f"  - Current playback_dev: {playback_dev}")
                
                # Select first available input device
                if info['maxInputChannels'] > 0 and capture_dev is None:
                    capture_dev = i
                    logger.info(f"✓ Fallback capture device: {info['name']}")
                
                # Select first available output device
                if info['maxOutputChannels'] > 0 and playback_dev is None:
                    playback_dev = i
                    logger.info(f"✓ Fallback playback device: {info['name']}")
                
                logger.info(f"  - After fallback evaluation - capture_dev: {capture_dev}, playback_dev: {playback_dev}")
                
                # If we have both, we can stop
                if capture_dev is not None and playback_dev is not None:
                    logger.info("  - Both devices selected, stopping fallback")
                    break
                logger.info("  ---")
        
        logger.info(f"=== FINAL DEVICE SELECTION VALIDATION ===")
        logger.info(f"capture_dev: {capture_dev}")
        logger.info(f"playback_dev: {playback_dev}")
        logger.info(f"capture_dev is None: {capture_dev is None}")
        logger.info(f"playback_dev is None: {playback_dev is None}")
        logger.info(f"capture_dev is falsy: {not capture_dev}")
        logger.info(f"playback_dev is falsy: {not playback_dev}")
        
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
        
        logger.info(f"=== FINAL DEVICE SELECTION SUCCESS ===")
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
            logger.info("Outgoing call - dialing")
        elif ci.state == pj.PJSIP_INV_STATE_INCOMING:
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
            
            # Clean up PyAudio streams when call ends - ChatGPT's fix
            if self.voice_bot.audio_manager:
                logger.info("Cleaning up PyAudio streams after call")
                self.voice_bot.audio_manager.cleanup_streams()
                
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
            # Open PyAudio streams now (first time) - ChatGPT's deferred opening fix
            if self.voice_bot.audio_manager.capture_stream is None:
                cap_id, pb_id = getattr(self.voice_bot, "_pending_pa_ids", (None, None))
                if cap_id is None or pb_id is None:
                    logger.warning("PyAudio device ids missing; attempting defaults")
                    # let initialize_streams fall back to defaults
                    self.voice_bot.audio_manager.initialize_streams(0, 0)
                else:
                    logger.info(f"Opening PyAudio streams: cap={cap_id}, pb={pb_id}")
                    self.voice_bot.audio_manager.initialize_streams(cap_id, pb_id)
            
            # Get media
            call_media = self.getMedia(0)
            aud_dev_manager = pj.Endpoint.instance().audDevManager()
            
            # Get capture and playback media
            cap_media = aud_dev_manager.getCaptureDevMedia()
            pb_media = aud_dev_manager.getPlaybackDevMedia()
            
            # Connect audio streams
            call_media.startTransmit(pb_media)
            cap_media.startTransmit(call_media)
            
            logger.info("Audio bridge established (call ↔ loopback)")
            
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
    
    def onIncomingCall(self, prm):
        """Handle incoming calls - auto-answer them"""
        logger.info(f"Incoming call received from {prm.rdata.srcAddress}")
        
        try:
            # Create call object
            call = VoiceBotCall(self, prm.callId, self.voice_bot)
            
            # Auto-answer the call with 200 OK
            op = pj.CallOpParam()
            op.statusCode = 200
            call.answer(op)
            
            logger.info("Incoming call auto-answered successfully")
            
        except Exception as e:
            logger.error(f"Failed to answer incoming call: {e}")

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
        
        # Safety options
        self.use_null_audio = config.get('use_null_audio', False)
        
        logger.info("Voice Bot initialized")
    
    def initialize_pjsua2(self) -> bool:
        """Initialize pjsua2 endpoint and transport"""
        try:
            logger.info("Initializing pjsua2...")
            
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Configure logging and initialize with safe defaults
            try:
                # Use safe default MediaConfig - don't force custom settings that can destabilize pjmedia
                ua_cfg = pj.UAConfig()
                log_cfg = pj.LogConfig()
                log_cfg.level = 4  # INFO level
                log_cfg.consoleLevel = 4
                med_cfg = pj.MediaConfig()  # Use defaults - most stable path
                
                self.endpoint.libInit(ua_cfg, log_cfg, med_cfg)
                logger.info("pjsua2 initialized with default MediaConfig (most stable)")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Individual configs approach failed: {e}")
                try:
                    # Fallback - minimal EpConfig with defaults
                    ep_cfg = pj.EpConfig()
                    # Only set logging, leave media config as default
                    ep_cfg.logConfig.level = 4
                    ep_cfg.logConfig.consoleLevel = 4
                    
                    self.endpoint.libInit(ep_cfg)
                    logger.info("pjsua2 initialized with minimal EpConfig")
                except Exception as e2:
                    logger.error(f"All pjsua2 initialization attempts failed: {e2}")
                    raise RuntimeError(f"Failed to initialize pjsua2: {e2}")
            
            # Create UDP transport
            tcfg = pj.TransportConfig()
            tcfg.port = self.sip_config.port
            self.endpoint.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
            
            # Start library
            self.endpoint.libStart()
            
            # Optional null device mode for testing
            if self.use_null_audio:
                logger.info("Using null audio device for testing (no real audio)")
                pj.Endpoint.instance().audDevManager().setNullDev()
            
            logger.info("pjsua2 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"pjsua2 initialization failed: {e}")
            return False
    
    def setup_audio_devices(self) -> bool:
        """Setup ALSA Loopback audio devices with enhanced debugging"""
        logger.info("=== STARTING AUDIO DEVICE SETUP ===")
        
        try:
            logger.info("Step 2.1: Testing basic PyAudio initialization...")
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                count = pa.get_device_count()
                logger.info(f"PyAudio basic test: {count} devices found")
                pa.terminate()
                logger.info("Step 2.1 SUCCESS: PyAudio basic test passed")
            except Exception as pa_error:
                logger.error(f"Step 2.1 FAILED: PyAudio basic test failed: {pa_error}")
                raise
            
            logger.info("Step 2.2: Creating AudioDeviceManager...")
            
            # Initialize audio manager
            self.audio_manager = AudioDeviceManager(self.audio_config)
            logger.info("Step 2.2 SUCCESS: AudioDeviceManager created")
            
            logger.info("Step 2.3: Selecting devices using name-based mapping...")
            
            # Use name-based device mapping (ChatGPT's fix for segfault)
            try:
                (pj_cap, pj_cap_name), (pj_pb, pj_pb_name) = self.audio_manager._pick_pjsua2_loopback()
                
                if pj_cap is not None and pj_pb is not None:
                    logger.info("Step 2.4: Binding pjsua2 to loopback devices...")
                    adm = pj.Endpoint.instance().audDevManager()
                    adm.setCaptureDev(pj_cap)
                    adm.setPlaybackDev(pj_pb)
                    logger.info(f"Step 2.4 SUCCESS: Bound pjsua2 to: cap={pj_cap}({pj_cap_name}), pb={pj_pb}({pj_pb_name})")
                    
                    # Verify the binding worked
                    current_cap = adm.getCaptureDev()
                    current_pb = adm.getPlaybackDev()
                    logger.info(f"Step 2.4 VERIFY: Current devices after binding - cap={current_cap}, pb={current_pb}")
                    
                    # Map to PyAudio by name (not number) and store for later
                    pa_cap = self.audio_manager._pyaudio_index_by_name(pj_cap_name, True)
                    pa_pb = self.audio_manager._pyaudio_index_by_name(pj_pb_name, False)
                    self._pending_pa_ids = (pa_cap, pa_pb)  # store for later, on call
                    
                    logger.info(f"Step 2.5: PyAudio device mapping: cap={pa_cap}, pb={pa_pb}")
                    logger.info("Step 2.5 SUCCESS: Device mapping completed - streams will open on call")
                    
                else:
                    logger.warning("Step 2.4 WARNING: Could not find Loopback in pjsua2 device list; leaving defaults")
                    self._pending_pa_ids = (None, None)
                    
                    # Add safety option to use null device if no loopback found
                    logger.info("Step 2.4 SAFETY: Consider using setNullDev() for testing if segfaults persist")
                    
            except Exception as mapping_error:
                logger.error(f"Step 2.3 FAILED: Device mapping failed: {mapping_error}")
                logger.error(f"Exception type: {type(mapping_error)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise  # Re-raise to see full stack trace
            
            logger.info(f"=== AUDIO DEVICE SETUP COMPLETE ===")
            logger.info(f"Device mapping completed - streams will open during calls")
            return True
            
        except Exception as e:
            logger.error(f"=== AUDIO DEVICE SETUP FAILED ===")
            logger.error(f"Error: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
            logger.info("=== VOICEBOT RUN METHOD STARTED ===")
            logger.info(f"Dry run mode: {dry_run}")
            logger.info("Starting Voice Bot...")
            
            # Initialize pjsua2
            logger.info("Step 1: Initializing pjsua2...")
            if not self.initialize_pjsua2():
                logger.error("Step 1 FAILED: pjsua2 initialization failed")
                return False
            logger.info("Step 1 SUCCESS: pjsua2 initialized")
            
            # Setup audio devices
            logger.info("Step 2: Setting up audio devices...")
            logger.info("About to call self.setup_audio_devices()...")
            setup_result = self.setup_audio_devices()
            logger.info(f"setup_audio_devices() returned: {setup_result}")
            if not setup_result:
                logger.error("Step 2 FAILED: Audio device setup failed")
                return False
            logger.info("Step 2 SUCCESS: Audio devices configured")
            
            # Initialize AI pipeline
            logger.info("Step 3: Initializing AI pipeline...")
            self.ai_pipeline = AIPipeline(self.config)
            if not self.ai_pipeline.initialize_models():
                logger.error("Step 3 FAILED: AI pipeline initialization failed")
                return False
            logger.info("Step 3 SUCCESS: AI pipeline initialized")
            
            if dry_run:
                logger.info("=== DRY RUN SUCCESS ===")
                logger.info("✓ pjsua2 initialized successfully")
                logger.info("✓ Audio devices configured successfully") 
                logger.info("✓ AI pipeline initialized successfully")
                logger.info("Dry run completed successfully - all components initialized")
                logger.info("NOTE: Dry run does NOT test SIP registration or audio bridging!")
                return True
            
            # Register with 3CX
            logger.info("Step 4: Registering with 3CX...")
            if not self.register_account():
                logger.error("Step 4 FAILED: SIP registration failed")
                return False
            logger.info("Step 4 SUCCESS: SIP registration completed")
            
            # Call handler is set via Account.onIncomingCall() - no need for setCallHandler
            
            self.is_running = True
            logger.info("Voice Bot is running and ready for calls")
            logger.info(f"Extension: {self.sip_config.extension}")
            logger.info(f"Server: {self.sip_config.server}")
            
            # Setup signal handler for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.is_running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Keep running
            try:
                logger.info("Voice bot is running. Press Ctrl+C to stop gracefully.")
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
                self.is_running = False
            
            return True
            
        except Exception as e:
            logger.error(f"Voice Bot run error: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources in proper order to prevent segfaults"""
        try:
            logger.info("Starting cleanup sequence...")
            
            self.is_running = False
            
            # Step 1: Stop all calls first
            if self.endpoint:
                try:
                    logger.info("Hanging up all calls...")
                    self.endpoint.hangupAllCalls()
                    time.sleep(0.5)  # Shorter wait
                except Exception as e:
                    logger.warning(f"Error hanging up calls: {e}")
            
            # Step 2: Cleanup audio streams BEFORE destroying pjsua2
            if self.audio_manager:
                try:
                    logger.info("Cleaning up audio streams...")
                    self.audio_manager.cleanup_streams()  # Just streams, not PyAudio yet
                except Exception as e:
                    logger.warning(f"Error cleaning up audio streams: {e}")
            
            # Step 3: Destroy pjsua2 endpoint with extra safety
            if self.endpoint:
                try:
                    logger.info("Destroying pjsua2 endpoint...")
                    # Force garbage collection before destroying
                    import gc
                    gc.collect()
                    
                    self.endpoint.libDestroy()
                    self.endpoint = None
                    
                    # Extra cleanup time and force GC again
                    time.sleep(1.0)  # Longer wait for pjsua2 cleanup
                    gc.collect()
                    
                except Exception as e:
                    logger.warning(f"Error destroying pjsua2: {e}")
            
            # Step 4: Now cleanup PyAudio safely
            if self.audio_manager:
                try:
                    logger.info("Cleaning up PyAudio...")
                    if self.audio_manager.pyaudio:
                        self.audio_manager.pyaudio.terminate()
                        self.audio_manager.pyaudio = None
                except Exception as e:
                    logger.warning(f"Error cleaning up PyAudio: {e}")
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            # Don't re-raise - we're cleaning up anyway

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
    parser.add_argument('--init-only', action='store_true',
                       help='Initialize all components including SIP registration, then exit')
    parser.add_argument('--test-call', action='store_true',
                       help='Make a test call to echo service')
    parser.add_argument('--extension', default='*777',
                       help='Extension to call for test (default: *777)')
    parser.add_argument('--null-audio', action='store_true',
                       help='Use null audio device for testing (isolates audio issues)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Add null audio option from command line
    if args.null_audio:
        config['use_null_audio'] = True
        logger.info("Null audio mode enabled - will isolate audio device issues")
    
    # Create voice bot
    voice_bot = VoiceBot(config)
    
    # Global cleanup function
    def cleanup_and_exit():
        try:
            logger.info("Performing final cleanup...")
            voice_bot.cleanup()
            logger.info("Final cleanup completed")
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")
        finally:
            sys.exit(0)
    
    # Register cleanup on normal exit
    import atexit
    atexit.register(lambda: voice_bot.cleanup() if voice_bot else None)
    
    if args.dry_run:
        success = voice_bot.run(dry_run=True)
    elif args.init_only:
        logger.info("=== INIT-ONLY MODE ===")
        logger.info("This will test SIP registration and audio device configuration")
        success = voice_bot.run()
        if success:
            logger.info("=== INIT-ONLY SUCCESS ===")
            logger.info("✓ All components initialized successfully")
            logger.info("✓ SIP registration completed")
            logger.info("✓ Audio bridging can be tested with incoming calls")
            voice_bot.cleanup()
    elif args.test_call:
        logger.info("=== TEST CALL MODE ===")
        logger.info("This will register with 3CX and make a test call")
        success = voice_bot.run()  # NOT dry-run - need registration for test calls
        if success:
            voice_bot.make_test_call(args.extension)
    else:
        try:
            success = voice_bot.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            success = True
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            success = False
        finally:
            # Ensure cleanup happens
            try:
                voice_bot.cleanup()
            except:
                pass
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
