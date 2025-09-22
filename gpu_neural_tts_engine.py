#!/usr/bin/env python3
"""
NETOVO VoiceBot - GPU-Accelerated Neural Text-to-Speech Engine
Professional-quality TTS using H100 GPU optimized for 8kHz telephony output
Targets <2s latency with natural, non-robotic voice quality
"""

import torch
import numpy as np
import soundfile as sf
import time
import logging
import threading
import queue
import asyncio
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass
import tempfile
import subprocess
import io

# Try importing neural TTS libraries
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("TTS library not available. Install with: pip install TTS")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class TTSResult:
    """Text-to-speech result with metadata"""
    audio_data: np.ndarray
    sample_rate: int
    processing_time: float
    text: str
    voice_model: str
    audio_file_path: Optional[str] = None


class GPUNeuralTTSEngine:
    """
    High-performance GPU-accelerated Neural Text-to-Speech engine
    optimized for professional telephony applications on H100 GPU.
    """

    def __init__(self,
                 model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
                 device: str = "cuda",
                 vocoder_name: Optional[str] = None,
                 target_sample_rate: int = 8000):
        """
        Initialize GPU Neural TTS engine

        Args:
            model_name: TTS model name (Coqui TTS format)
            device: torch device (cuda for GPU, cpu for fallback)
            vocoder_name: Optional vocoder model
            target_sample_rate: Target sample rate for telephony (8000 Hz)
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.target_sample_rate = target_sample_rate

        # Performance settings
        self.max_concurrent_requests = 4
        self.processing_queue = queue.Queue(maxsize=20)
        self.result_cache = {}
        self.cache_timeout = 300  # 5 minutes

        # Audio processing settings
        self.chunk_size = 1024
        self.overlap_samples = 256

        # Initialize GPU and models
        self._initialize_gpu()
        self._initialize_tts_engines()
        self._start_processing_threads()

    def _initialize_gpu(self):
        """Initialize GPU settings for optimal H100 performance"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            return

        # Check for H100 GPU
        gpu_name = torch.cuda.get_device_name(0)
        self.logger.info(f"GPU detected: {gpu_name}")

        if "H100" in gpu_name:
            self.logger.info("H100 GPU detected - enabling optimized settings")
            # H100-specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Display GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.info(f"GPU memory available: {total_memory:.1f} GB")

    def _initialize_tts_engines(self):
        """Initialize multiple TTS engines with fallbacks"""
        self.tts_engines = {}
        self.primary_engine = None

        # 1. Try Coqui TTS (Neural, GPU-accelerated)
        if TTS_AVAILABLE:
            try:
                self._initialize_coqui_tts()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Coqui TTS: {e}")

        # 2. Initialize fallback engines
        self._initialize_fallback_engines()

        # Set primary engine
        if "coqui" in self.tts_engines:
            self.primary_engine = "coqui"
        elif "espeak" in self.tts_engines:
            self.primary_engine = "espeak"
        else:
            raise RuntimeError("No TTS engines available")

        self.logger.info(f"Primary TTS engine: {self.primary_engine}")

    def _initialize_coqui_tts(self):
        """Initialize Coqui TTS neural models"""
        if not TTS_AVAILABLE:
            return

        start_time = time.time()
        self.logger.info("Initializing Coqui TTS neural models...")

        try:
            # Try multiple model configurations for best quality
            model_configs = [
                # High-quality models (if available)
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ek1/tacotron2",
                # Fallback models
                "tts_models/en/ljspeech/speedy-speech",
            ]

            for model_config in model_configs:
                try:
                    self.logger.info(f"Attempting to load model: {model_config}")

                    # Initialize TTS model
                    tts = TTS(
                        model_name=model_config,
                        gpu=(self.device == "cuda")
                    )

                    # Test the model with a short phrase
                    test_audio = tts.tts("Hello")

                    if test_audio is not None and len(test_audio) > 0:
                        self.tts_engines["coqui"] = {
                            "engine": tts,
                            "model_name": model_config,
                            "quality": "neural_high"
                        }

                        load_time = time.time() - start_time
                        self.logger.info(f"Coqui TTS loaded successfully in {load_time:.2f}s")
                        self.logger.info(f"Using model: {model_config}")
                        return

                except Exception as e:
                    self.logger.debug(f"Model {model_config} failed: {e}")
                    continue

            self.logger.warning("No Coqui TTS models could be loaded")

        except Exception as e:
            self.logger.error(f"Coqui TTS initialization failed: {e}")

    def _initialize_fallback_engines(self):
        """Initialize fallback TTS engines for reliability"""

        # Enhanced espeak with better voice settings
        try:
            # Test espeak availability
            result = subprocess.run(
                ["espeak", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self.tts_engines["espeak"] = {
                    "command": "espeak",
                    "quality": "enhanced_synthetic",
                    "voices": {
                        "female": "en+f3",
                        "male": "en+m3"
                    }
                }
                self.logger.info("Enhanced espeak TTS initialized")

        except Exception as e:
            self.logger.debug(f"espeak not available: {e}")

        # Festival TTS
        try:
            result = subprocess.run(
                ["festival", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self.tts_engines["festival"] = {
                    "command": "festival",
                    "quality": "concatenative"
                }
                self.logger.info("Festival TTS initialized")

        except Exception as e:
            self.logger.debug(f"Festival not available: {e}")

        # Flite (lightweight)
        try:
            result = subprocess.run(
                ["flite", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                self.tts_engines["flite"] = {
                    "command": "flite",
                    "quality": "lightweight"
                }
                self.logger.info("Flite TTS initialized")

        except Exception as e:
            self.logger.debug(f"Flite not available: {e}")

    def _start_processing_threads(self):
        """Start background threads for concurrent TTS processing"""
        self.processing_threads = []

        for i in range(self.max_concurrent_requests):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"TTS-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)

        self.logger.info(f"Started {self.max_concurrent_requests} TTS processing threads")

    def _processing_worker(self):
        """Background worker thread for TTS processing"""
        while True:
            try:
                request = self.processing_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break

                text, engine_name, voice_options, result_queue, request_id = request

                try:
                    result = self._synthesize_speech_sync(
                        text, engine_name, voice_options, request_id
                    )
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"TTS worker error: {e}")

    def _synthesize_speech_sync(self,
                              text: str,
                              engine_name: Optional[str] = None,
                              voice_options: Optional[Dict] = None,
                              request_id: str = "default") -> TTSResult:
        """Synchronous speech synthesis with GPU optimization"""
        start_time = time.time()

        if engine_name is None:
            engine_name = self.primary_engine

        if voice_options is None:
            voice_options = {}

        try:
            self.logger.debug(f"Synthesizing: '{text[:50]}...' using {engine_name}")

            # Use appropriate engine
            if engine_name == "coqui" and "coqui" in self.tts_engines:
                audio_data = self._coqui_synthesize(text, voice_options)
                sample_rate = 22050  # Coqui default
                voice_model = self.tts_engines["coqui"]["model_name"]

            elif engine_name in ["espeak", "festival", "flite"]:
                audio_data, sample_rate = self._fallback_synthesize(
                    text, engine_name, voice_options
                )
                voice_model = f"{engine_name}_enhanced"

            else:
                # Fallback to primary engine
                if self.primary_engine == "coqui":
                    audio_data = self._coqui_synthesize(text, voice_options)
                    sample_rate = 22050
                    voice_model = "coqui_fallback"
                else:
                    audio_data, sample_rate = self._fallback_synthesize(
                        text, self.primary_engine, voice_options
                    )
                    voice_model = f"{self.primary_engine}_fallback"

            # Post-process audio for telephony
            processed_audio = self._post_process_audio(audio_data, sample_rate)

            processing_time = time.time() - start_time

            return TTSResult(
                audio_data=processed_audio,
                sample_rate=self.target_sample_rate,
                processing_time=processing_time,
                text=text,
                voice_model=voice_model
            )

        except Exception as e:
            self.logger.error(f"TTS synthesis failed: {e}")
            raise

    def _coqui_synthesize(self, text: str, voice_options: Dict) -> np.ndarray:
        """Synthesize speech using Coqui TTS neural models"""
        if "coqui" not in self.tts_engines:
            raise RuntimeError("Coqui TTS not available")

        tts_engine = self.tts_engines["coqui"]["engine"]

        try:
            # Configure voice options
            tts_kwargs = {}

            # Speaker/voice selection if supported
            if "speaker" in voice_options:
                tts_kwargs["speaker"] = voice_options["speaker"]

            # Language selection
            if "language" in voice_options:
                tts_kwargs["language"] = voice_options["language"]

            # Emotion/style if supported
            if "emotion" in voice_options:
                tts_kwargs["emotion"] = voice_options["emotion"]

            # Generate audio with GPU acceleration
            if self.device == "cuda":
                torch.cuda.synchronize()

            audio_data = tts_engine.tts(text, **tts_kwargs)

            if self.device == "cuda":
                torch.cuda.synchronize()

            # Convert to numpy array if needed
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

            return np.array(audio_data, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Coqui TTS synthesis failed: {e}")
            raise

    def _fallback_synthesize(self,
                           text: str,
                           engine_name: str,
                           voice_options: Dict) -> tuple[np.ndarray, int]:
        """Synthesize speech using fallback engines"""
        if engine_name not in self.tts_engines:
            raise RuntimeError(f"TTS engine {engine_name} not available")

        engine_config = self.tts_engines[engine_name]

        # Create temporary file for audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            if engine_name == "espeak":
                success = self._espeak_synthesize(text, temp_path, voice_options)
            elif engine_name == "festival":
                success = self._festival_synthesize(text, temp_path, voice_options)
            elif engine_name == "flite":
                success = self._flite_synthesize(text, temp_path, voice_options)
            else:
                raise RuntimeError(f"Unknown fallback engine: {engine_name}")

            if not success:
                raise RuntimeError(f"{engine_name} synthesis failed")

            # Load generated audio with better validation
            if Path(temp_path).exists():
                file_size = Path(temp_path).stat().st_size
                self.logger.debug(f"Generated audio file size: {file_size} bytes")

                if file_size > 100:  # Lower threshold for shorter test phrases
                    try:
                        audio_data, sample_rate = sf.read(temp_path)
                        if len(audio_data) > 0:
                            return audio_data.astype(np.float32), sample_rate
                        else:
                            self.logger.warning(f"Audio data is empty from {engine_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to read audio file: {e}")

                self.logger.warning(f"Generated audio file is too small: {file_size} bytes")

            # Generate a simple beep as fallback
            self.logger.info("Generating fallback beep tone")
            duration = min(len(text) * 0.1, 2.0)  # 100ms per character, max 2s
            sample_rate = 8000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * 800 * t)  # 800Hz tone
            return audio_data.astype(np.float32), sample_rate

        finally:
            # Cleanup temporary file
            try:
                Path(temp_path).unlink()
            except:
                pass

    def _espeak_synthesize(self, text: str, output_path: str, voice_options: Dict) -> bool:
        """Enhanced espeak synthesis with better voice quality"""
        try:
            # Build espeak command with optimized settings
            cmd = [
                "espeak",
                text,
                "-w", output_path,
                "-s", str(voice_options.get("speed", 160)),  # Slightly faster
                "-p", str(voice_options.get("pitch", 45)),   # Slightly lower pitch
                "-a", str(voice_options.get("amplitude", 100)),
                "-g", "5",  # 5ms gap between words
                "-k", "5"   # Capitalize emphasis
            ]

            # Voice selection
            voice = voice_options.get("voice", "en-us+f3")  # Default female voice
            cmd.extend(["-v", voice])

            # Execute espeak
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"espeak synthesis failed: {e}")
            return False

    def _festival_synthesize(self, text: str, output_path: str, voice_options: Dict) -> bool:
        """Festival TTS synthesis"""
        try:
            # Create Festival script
            festival_script = f"""
(voice_ked_diphone)
(Parameter.set 'Audio_Method 'Audio_Command)
(Parameter.set 'Audio_Command "sox -t raw -r 16000 -s -w - -r 8000 {output_path}")
(SayText "{text}")
"""

            # Execute Festival
            result = subprocess.run(
                ["festival"],
                input=festival_script,
                text=True,
                capture_output=True,
                timeout=15
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Festival synthesis failed: {e}")
            return False

    def _flite_synthesize(self, text: str, output_path: str, voice_options: Dict) -> bool:
        """Flite TTS synthesis"""
        try:
            cmd = [
                "flite",
                "-t", text,
                "-o", output_path,
                "-voice", voice_options.get("voice", "slt")  # Default voice
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Flite synthesis failed: {e}")
            return False

    def _post_process_audio(self, audio_data: np.ndarray, original_sample_rate: int) -> np.ndarray:
        """Post-process audio for telephony optimization"""
        try:
            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95

            # Resample to target sample rate (8kHz for telephony)
            if original_sample_rate != self.target_sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data) * self.target_sample_rate / original_sample_rate)
                audio_data = signal.resample(audio_data, num_samples)

            # Apply telephony band-pass filter (300-3400 Hz)
            nyquist = self.target_sample_rate / 2
            low_cutoff = 300 / nyquist
            high_cutoff = 3400 / nyquist

            from scipy import signal
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            audio_data = signal.filtfilt(b, a, audio_data)

            # Apply mild compression to even out volume
            audio_data = np.tanh(audio_data * 1.5) * 0.8

            # Ensure no clipping
            audio_data = np.clip(audio_data, -1.0, 1.0)

            return audio_data

        except Exception as e:
            self.logger.error(f"Audio post-processing failed: {e}")
            return audio_data  # Return original if processing fails

    async def synthesize_speech_async(self,
                                    text: str,
                                    engine_name: Optional[str] = None,
                                    voice_options: Optional[Dict] = None,
                                    request_id: Optional[str] = None) -> TTSResult:
        """Asynchronous speech synthesis"""
        if request_id is None:
            request_id = f"tts_{int(time.time() * 1000)}"

        if voice_options is None:
            voice_options = {}

        # Check cache first
        cache_key = f"{request_id}_{hash(text)}_{engine_name}"
        if cache_key in self.result_cache:
            cache_entry = self.result_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_timeout:
                self.logger.debug(f"Returning cached TTS result for {request_id}")
                return cache_entry["result"]

        # Create result queue
        result_queue = queue.Queue()

        # Submit to processing queue
        try:
            self.processing_queue.put(
                (text, engine_name, voice_options, result_queue, request_id),
                timeout=5
            )
        except queue.Full:
            raise RuntimeError("TTS processing queue full - too many concurrent requests")

        # Wait for result asynchronously
        loop = asyncio.get_event_loop()

        def get_result():
            return result_queue.get(timeout=15)

        try:
            status, result = await loop.run_in_executor(None, get_result)

            if status == "error":
                raise RuntimeError(f"TTS processing failed: {result}")

            # Cache result
            self.result_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }

            return result

        except queue.Empty:
            raise TimeoutError("TTS processing timeout")

    def synthesize_speech(self,
                         text: str,
                         engine_name: Optional[str] = None,
                         voice_options: Optional[Dict] = None,
                         request_id: Optional[str] = None) -> TTSResult:
        """Synchronous speech synthesis"""
        if request_id is None:
            request_id = f"tts_{int(time.time() * 1000)}"

        if voice_options is None:
            voice_options = {}

        return self._synthesize_speech_sync(text, engine_name, voice_options, request_id)

    def synthesize_to_file(self,
                          text: str,
                          output_path: str,
                          engine_name: Optional[str] = None,
                          voice_options: Optional[Dict] = None) -> TTSResult:
        """Synthesize speech and save to file"""
        result = self.synthesize_speech(text, engine_name, voice_options)

        # Save audio to file
        sf.write(output_path, result.audio_data, result.sample_rate)
        result.audio_file_path = output_path

        return result

    def get_available_engines(self) -> Dict[str, Dict]:
        """Get list of available TTS engines and their capabilities"""
        return self.tts_engines.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "primary_engine": self.primary_engine,
            "available_engines": list(self.tts_engines.keys()),
            "queue_size": self.processing_queue.qsize(),
            "cache_entries": len(self.result_cache),
            "worker_threads": len(self.processing_threads),
            "target_sample_rate": self.target_sample_rate
        }

        if self.device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1e9,
                "gpu_name": torch.cuda.get_device_name(0)
            })

        return stats

    def cleanup(self):
        """Cleanup resources and stop processing threads"""
        self.logger.info("Cleaning up TTS engine...")

        # Stop processing threads
        for _ in self.processing_threads:
            self.processing_queue.put(None)

        for thread in self.processing_threads:
            thread.join(timeout=5)

        # Clear GPU cache
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear result cache
        self.result_cache.clear()

        self.logger.info("TTS engine cleanup completed")


if __name__ == "__main__":
    # Test the GPU Neural TTS engine
    logging.basicConfig(level=logging.INFO)

    print("=== Testing GPU Neural TTS Engine ===")

    # Initialize TTS engine
    tts_engine = GPUNeuralTTSEngine(device="cuda")

    # Test synthesis
    test_text = "Hello! I'm Alexis from NETOVO IT Services. How can I help you today?"

    start_time = time.time()
    result = tts_engine.synthesize_speech(
        test_text,
        voice_options={"voice": "en-us+f3", "speed": 160, "pitch": 45}
    )
    total_time = time.time() - start_time

    print(f"Text: '{test_text}'")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Audio length: {len(result.audio_data) / result.sample_rate:.2f}s")
    print(f"Voice model: {result.voice_model}")
    print(f"Sample rate: {result.sample_rate} Hz")

    # Performance stats
    stats = tts_engine.get_performance_stats()
    print(f"Available engines: {stats['available_engines']}")
    print(f"Primary engine: {stats['primary_engine']}")

    # Test different engines if available
    for engine_name in stats['available_engines']:
        try:
            start_time = time.time()
            test_result = tts_engine.synthesize_speech(
                "Testing engine quality",
                engine_name=engine_name
            )
            engine_time = time.time() - start_time
            print(f"{engine_name}: {test_result.processing_time:.3f}s processing, {engine_time:.3f}s total")
        except Exception as e:
            print(f"{engine_name}: Failed - {e}")

    # Cleanup
    tts_engine.cleanup()
    print("=== TTS Test Complete ===")
