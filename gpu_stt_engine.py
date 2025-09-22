#!/usr/bin/env python3
"""
NETOVO VoiceBot - GPU-Accelerated Speech-to-Text Engine
High-performance STT using Whisper Large on H100 GPU with sub-300ms turnaround
"""

import torch
import whisper
import numpy as np
import soundfile as sf
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import threading
import queue
import asyncio
from dataclasses import dataclass


@dataclass
class STTResult:
    """Speech-to-text result with metadata"""
    text: str
    confidence: float
    processing_time: float
    language: str = "en"
    segments: list = None


class GPUSTTEngine:
    """
    High-performance GPU-accelerated Speech-to-Text engine optimized for
    telephony applications using Whisper Large on H100 GPU.
    """

    def __init__(self,
                 model_size: str = "large",
                 device: str = "cuda",
                 compute_type: str = "float32",
                 batch_size: int = 1):
        """
        Initialize GPU STT engine

        Args:
            model_size: Whisper model size (base, small, medium, large, large-v2)
            device: torch device (cuda for GPU, cpu for fallback)
            compute_type: Precision (float16 for speed, float32 for accuracy)
            batch_size: Concurrent transcription batch size
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.model_size = model_size

        # Performance settings for H100 optimization
        self.max_concurrent_requests = 4
        self.processing_queue = queue.Queue(maxsize=50)
        self.result_cache = {}
        self.cache_timeout = 300  # 5 minutes

        # Initialize GPU and model
        self._initialize_gpu()
        self._load_whisper_model()

        # Start background processing threads
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

            # Set optimal batch size for H100
            self.batch_size = min(self.batch_size, 8)

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Display GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.info(f"GPU memory available: {total_memory:.1f} GB")

    def _load_whisper_model(self):
        """Load and optimize Whisper model for GPU inference"""
        start_time = time.time()

        try:
            self.logger.info(f"Loading Whisper {self.model_size} model on {self.device}")

            # Load model with GPU optimization
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=None  # Use default cache
            )

            # Optimize model for inference
            if self.device == "cuda":
                self.model = self.model.cuda()

                # Enable half precision for H100 if specified
                if self.compute_type == "float16":
                    self.model = self.model.half()
                    self.logger.info("Enabled FP16 precision for faster inference")

            # Set model to evaluation mode
            self.model.eval()

            load_time = time.time() - start_time
            self.logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")

            # Test model with dummy audio to warm up GPU
            self._warmup_model()

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _warmup_model(self):
        """Warm up model with dummy audio for consistent performance"""
        self.logger.info("Warming up Whisper model...")

        # Create dummy 1-second audio (8kHz, telephony standard)
        dummy_audio = np.random.randn(8000).astype(np.float32)

        try:
            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start_time = time.time()
                result = self.model.transcribe(
                    dummy_audio,
                    language="en",
                    task="transcribe",
                    fp16=False  # Force float32 to avoid type conflicts
                )

                if self.device == "cuda":
                    torch.cuda.synchronize()

                warmup_time = time.time() - start_time
                self.logger.info(f"Model warmup completed in {warmup_time:.3f}s")

        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

    def _start_processing_threads(self):
        """Start background threads for concurrent STT processing"""
        self.processing_threads = []

        for i in range(self.max_concurrent_requests):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"STT-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)

        self.logger.info(f"Started {self.max_concurrent_requests} STT processing threads")

    def _processing_worker(self):
        """Background worker thread for STT processing"""
        while True:
            try:
                # Get processing request
                request = self.processing_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break

                audio_data, result_queue, request_id = request

                # Process STT
                try:
                    result = self._transcribe_audio_sync(audio_data, request_id)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"STT worker error: {e}")

    def _transcribe_audio_sync(self, audio_data: np.ndarray, request_id: str) -> STTResult:
        """Synchronous transcription with GPU optimization"""
        start_time = time.time()

        try:
            # Ensure audio is in correct format (float32, normalized)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio to [-1, 1] range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Transcribe with Whisper
            with torch.no_grad():
                if self.device == "cuda":
                    torch.cuda.synchronize()

                transcribe_options = {
                    "language": "en",
                    "task": "transcribe",
                    "fp16": False,  # Disable half precision to fix Float/Half type conflict
                    "verbose": False,
                    "word_timestamps": False,  # Disable for speed
                    "condition_on_previous_text": False  # Disable for telephony
                }

                result = self.model.transcribe(audio_data, **transcribe_options)

                if self.device == "cuda":
                    torch.cuda.synchronize()

            processing_time = time.time() - start_time

            # Extract results
            transcribed_text = result["text"].strip()
            language = result.get("language", "en")

            # Calculate confidence (Whisper doesn't provide direct confidence)
            # Use audio quality metrics and text length as proxy
            confidence = self._estimate_confidence(audio_data, transcribed_text, result)

            self.logger.debug(f"STT completed in {processing_time:.3f}s: '{transcribed_text}'")

            return STTResult(
                text=transcribed_text,
                confidence=confidence,
                processing_time=processing_time,
                language=language,
                segments=result.get("segments", [])
            )

        except Exception as e:
            self.logger.error(f"STT transcription failed: {e}")
            raise

    def _estimate_confidence(self, audio_data: np.ndarray, text: str, whisper_result: dict) -> float:
        """Estimate transcription confidence based on audio and text quality"""
        confidence = 0.5  # Base confidence

        # Audio quality factors
        audio_level = np.mean(np.abs(audio_data))
        if audio_level > 0.01:  # Good audio level
            confidence += 0.2

        # Text quality factors
        if len(text.strip()) > 0:
            confidence += 0.2

            # Check for reasonable word count (not too short/long)
            word_count = len(text.split())
            if 1 <= word_count <= 50:  # Reasonable length
                confidence += 0.1

        # Whisper segments information
        segments = whisper_result.get("segments", [])
        if segments:
            # Average probability from segments if available
            try:
                avg_prob = np.mean([seg.get("avg_logprob", 0) for seg in segments])
                # Convert log prob to linear scale (approximate)
                segment_confidence = min(1.0, max(0.0, np.exp(avg_prob / 2)))
                confidence = 0.7 * confidence + 0.3 * segment_confidence
            except:
                pass

        return min(1.0, max(0.0, confidence))

    async def transcribe_audio_async(self, audio_data: np.ndarray,
                                   request_id: Optional[str] = None) -> STTResult:
        """
        Asynchronous audio transcription for non-blocking operation

        Args:
            audio_data: Audio data as numpy array (float32, 8kHz)
            request_id: Optional request identifier for caching

        Returns:
            STTResult with transcription and metadata
        """
        if request_id is None:
            request_id = f"stt_{int(time.time() * 1000)}"

        # Check cache first
        if request_id in self.result_cache:
            cache_entry = self.result_cache[request_id]
            if time.time() - cache_entry["timestamp"] < self.cache_timeout:
                self.logger.debug(f"Returning cached STT result for {request_id}")
                return cache_entry["result"]

        # Create result queue
        result_queue = queue.Queue()

        # Submit to processing queue
        try:
            self.processing_queue.put((audio_data, result_queue, request_id), timeout=5)
        except queue.Full:
            raise RuntimeError("STT processing queue full - too many concurrent requests")

        # Wait for result asynchronously
        loop = asyncio.get_event_loop()

        def get_result():
            return result_queue.get(timeout=10)  # 10 second timeout

        try:
            status, result = await loop.run_in_executor(None, get_result)

            if status == "error":
                raise RuntimeError(f"STT processing failed: {result}")

            # Cache result
            self.result_cache[request_id] = {
                "result": result,
                "timestamp": time.time()
            }

            return result

        except queue.Empty:
            raise TimeoutError("STT processing timeout")

    def transcribe_audio(self, audio_data: np.ndarray,
                        request_id: Optional[str] = None) -> STTResult:
        """
        Synchronous audio transcription (blocking)

        Args:
            audio_data: Audio data as numpy array (float32, 8kHz)
            request_id: Optional request identifier

        Returns:
            STTResult with transcription and metadata
        """
        if request_id is None:
            request_id = f"stt_{int(time.time() * 1000)}"

        return self._transcribe_audio_sync(audio_data, request_id)

    def transcribe_file(self, audio_file_path: str) -> STTResult:
        """
        Transcribe audio from file

        Args:
            audio_file_path: Path to audio file

        Returns:
            STTResult with transcription and metadata
        """
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Resample to 16kHz if needed (Whisper's expected rate)
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio_data) * 16000 / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)

            # Transcribe
            request_id = f"file_{Path(audio_file_path).stem}_{int(time.time())}"
            return self.transcribe_audio(audio_data, request_id)

        except Exception as e:
            self.logger.error(f"Failed to transcribe file {audio_file_path}: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not hasattr(self, 'model'):
            return {"status": "not_initialized"}

        stats = {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "queue_size": self.processing_queue.qsize(),
            "cache_entries": len(self.result_cache),
            "worker_threads": len(self.processing_threads)
        }

        if self.device == "cuda":
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1e9,
                "gpu_name": torch.cuda.get_device_name(0)
            })

        return stats

    def cleanup(self):
        """Cleanup resources and stop processing threads"""
        self.logger.info("Cleaning up STT engine...")

        # Stop processing threads
        for _ in self.processing_threads:
            self.processing_queue.put(None)  # Shutdown signal

        for thread in self.processing_threads:
            thread.join(timeout=5)

        # Clear GPU cache
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Clear result cache
        self.result_cache.clear()

        self.logger.info("STT engine cleanup completed")


class TelephonySTTOptimizer:
    """Specialized optimizations for telephony audio (8kHz, compressed)"""

    @staticmethod
    def preprocess_telephony_audio(audio_data: np.ndarray,
                                 sample_rate: int = 8000) -> np.ndarray:
        """
        Preprocess telephony audio for optimal STT performance

        Args:
            audio_data: Raw telephony audio
            sample_rate: Audio sample rate

        Returns:
            Optimized audio data
        """
        # Convert to float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply telephony-specific filtering
        # High-pass filter to remove low-frequency noise
        from scipy import signal

        # Design high-pass filter (remove < 100 Hz)
        nyquist = sample_rate / 2
        high_cutoff = 100 / nyquist
        b, a = signal.butter(4, high_cutoff, btype='high')
        audio_data = signal.filtfilt(b, a, audio_data)

        # Low-pass filter for telephony band (remove > 3400 Hz)
        low_cutoff = 3400 / nyquist
        b, a = signal.butter(4, low_cutoff, btype='low')
        audio_data = signal.filtfilt(b, a, audio_data)

        # Upsample to 16kHz for Whisper
        if sample_rate != 16000:
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)

        return audio_data

    @staticmethod
    def enhance_audio_quality(audio_data: np.ndarray) -> np.ndarray:
        """Apply audio enhancement for better STT recognition"""
        # Noise reduction using spectral subtraction (simplified)
        # This is a basic implementation - could be enhanced with ML models

        # Apply mild compression to even out levels
        audio_data = np.tanh(audio_data * 2) / 2

        # Remove silence gaps (helps with chunked processing)
        silence_threshold = 0.01
        non_silent = np.abs(audio_data) > silence_threshold

        if np.any(non_silent):
            # Keep some context around speech
            context_samples = int(0.1 * 16000)  # 100ms context

            # Find speech boundaries
            speech_start = max(0, np.where(non_silent)[0][0] - context_samples)
            speech_end = min(len(audio_data), np.where(non_silent)[0][-1] + context_samples)

            audio_data = audio_data[speech_start:speech_end]

        return audio_data


if __name__ == "__main__":
    # Test the GPU STT engine
    logging.basicConfig(level=logging.INFO)

    print("=== Testing GPU STT Engine ===")

    # Initialize STT engine
    stt_engine = GPUSTTEngine(model_size="large", device="cuda")

    # Test with dummy audio
    test_audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

    # Synchronous test
    start_time = time.time()
    result = stt_engine.transcribe_audio(test_audio)
    processing_time = time.time() - start_time

    print(f"Transcription: '{result.text}'")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Total time: {processing_time:.3f}s")

    # Performance stats
    stats = stt_engine.get_performance_stats()
    print(f"Performance stats: {stats}")

    # Cleanup
    stt_engine.cleanup()
    print("=== STT Test Complete ===")
    
