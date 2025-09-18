#!/usr/bin/env python3
"""
GPU-Accelerated TTS Processor for NETOVO VoiceBot
Uses Coqui TTS with NVIDIA H100 optimization
"""

import os
import tempfile
import subprocess
import logging
from typing import Optional
from TTS.api import TTS
import torch

logger = logging.getLogger('GPU_TTS')

class GPUTTSProcessor:
    """High-performance TTS using NVIDIA H100 GPU"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = None
        self.model_loaded = False
        
        logger.info(f"GPU TTS Processor initializing on {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def load_model(self) -> bool:
        """Load TTS model with GPU acceleration"""
        try:
            if self.model_loaded:
                return True
                
            logger.info("Loading Coqui TTS model on GPU...")
            
            # Use high-quality model optimized for telephony
            self.tts_model = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC_ph",
                gpu=True if self.device == "cuda" else False
            )
            
            self.model_loaded = True
            logger.info("GPU TTS model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            return False
    
    def generate_speech(self, text: str) -> Optional[str]:
        """Generate high-quality speech using GPU acceleration"""
        try:
            if not self.model_loaded:
                if not self.load_model():
                    return None
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Generate speech with GPU acceleration
            logger.info(f"Generating speech: {text[:50]}...")
            
            self.tts_model.tts_to_file(
                text=text,
                file_path=temp_file.name
            )
            
            # Convert to telephony format (8kHz, mono, 16-bit)
            telephony_file = self.convert_for_asterisk(temp_file.name)
            
            # Cleanup original file
            os.unlink(temp_file.name)
            
            if telephony_file:
                logger.info("Speech generation successful")
                return telephony_file
            else:
                logger.error("Format conversion failed")
                return None
                
        except Exception as e:
            logger.error(f"Speech generation error: {e}")
            return None
    
    def convert_for_asterisk(self, input_file: str) -> Optional[str]:
        """Convert audio to Asterisk-compatible format"""
        try:
            output_file = input_file.replace('.wav', '_asterisk.wav')
            
            # Use sox for high-quality conversion
            sox_cmd = [
                'sox', input_file,
                '-r', '8000',      # 8kHz sample rate for telephony
                '-c', '1',         # Mono
                '-b', '16',        # 16-bit depth
                output_file,
                'gain', '-n',      # Normalize audio
                'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2'  # Compression for telephony
            ]
            
            result = subprocess.run(sox_cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(output_file):
                logger.info(f"Audio converted for Asterisk: {output_file}")
                return output_file
            else:
                logger.error(f"Sox conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None
