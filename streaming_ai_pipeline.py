#!/usr/bin/env python3
"""
NETOVO VoiceBot - Streaming AI Pipeline
High-performance streaming pipeline binding Whisper STT + Ollama LLM + Neural TTS
Optimized for real-time telephony with <2s TTS latency and sub-300ms STT turnaround
"""

import asyncio
import time
import logging
import threading
import queue
import numpy as np
from typing import Optional, Dict, Any, Callable, List, AsyncGenerator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path

# Import our custom modules
from conversation_context_manager import ConversationContextManager, session_manager
from gpu_stt_engine import GPUSTTEngine, STTResult
from gpu_neural_tts_engine import GPUNeuralTTSEngine, TTSResult
from telephony_audio_processor import TelephonyAudioProcessor, AudioMetrics

# Ollama integration
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Configuration for the streaming AI pipeline"""
    # Performance targets
    stt_timeout: float = 0.3  # 300ms STT target
    tts_timeout: float = 2.0  # 2s TTS target
    llm_timeout: float = 5.0  # 5s LLM timeout

    # Audio settings
    sample_rate: int = 8000  # Telephony standard
    chunk_size: int = 1024   # Audio chunk size

    # Model settings
    whisper_model: str = "large"
    ollama_model: str = "orca2:7b"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"

    # Pipeline settings
    max_concurrent_calls: int = 4
    enable_streaming: bool = True
    enable_caching: bool = True

    # Quality settings
    min_audio_duration: float = 0.1  # Minimum audio length
    max_audio_duration: float = 10.0  # Maximum audio length
    min_stt_confidence: float = 0.3   # Minimum STT confidence


@dataclass
class StreamingResult:
    """Result from streaming pipeline processing"""
    session_id: str
    turn_number: int
    user_text: str
    bot_response: str
    audio_data: np.ndarray
    processing_times: Dict[str, float]
    confidence_scores: Dict[str, float]
    audio_metrics: AudioMetrics
    success: bool = True
    error_message: Optional[str] = None


class OllamaLLMEngine:
    """Ollama LLM integration for conversation processing"""

    def __init__(self, model_name: str = "orca2:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("requests library not available")

            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.logger.info("Ollama connection successful")
            else:
                raise RuntimeError(f"Ollama server error: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Ollama connection failed: {e}")
            raise

    async def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using Ollama"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["\n\n", "Customer:", "Human:", "User:"]
                }
            }

            # Make async request
            loop = asyncio.get_event_loop()

            def make_request():
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=10
                )
                return response.json()

            result = await loop.run_in_executor(None, make_request)

            if "response" in result:
                return result["response"].strip()
            else:
                raise RuntimeError(f"Invalid Ollama response: {result}")

        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise


class StreamingAIPipeline:
    """
    High-performance streaming AI pipeline for real-time telephony conversations.
    Integrates STT, LLM, and TTS with conversation context management.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.pipeline_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "avg_stt_time": 0.0,
            "avg_llm_time": 0.0,
            "avg_tts_time": 0.0,
            "avg_total_time": 0.0
        }

        # Initialize components
        self._initialize_components()

        # Threading for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_calls * 3)

        # Processing queues for pipeline stages
        self.stt_queue = asyncio.Queue(maxsize=20)
        self.llm_queue = asyncio.Queue(maxsize=20)
        self.tts_queue = asyncio.Queue(maxsize=20)

        # Active processing tracking
        self.active_sessions: Dict[str, Dict] = {}

        self.logger.info("Streaming AI Pipeline initialized successfully")

    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            # Initialize STT engine
            self.logger.info("Initializing STT engine...")
            self.stt_engine = GPUSTTEngine(
                model_size=self.config.whisper_model,
                device="cuda"
            )

            # Initialize TTS engine
            self.logger.info("Initializing TTS engine...")
            self.tts_engine = GPUNeuralTTSEngine(
                model_name=self.config.tts_model,
                device="cuda",
                target_sample_rate=self.config.sample_rate
            )

            # Initialize LLM engine
            self.logger.info("Initializing LLM engine...")
            self.llm_engine = OllamaLLMEngine(model_name=self.config.ollama_model)

            # Initialize audio processor
            self.logger.info("Initializing audio processor...")
            self.audio_processor = TelephonyAudioProcessor(
                target_sample_rate=self.config.sample_rate
            )

            self.logger.info("All pipeline components initialized")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def process_conversation_turn(self,
                                      session_id: str,
                                      audio_data: np.ndarray,
                                      input_sample_rate: int = 8000) -> StreamingResult:
        """
        Process a complete conversation turn: audio in → text → LLM → TTS → audio out

        Args:
            session_id: Unique session identifier
            audio_data: Input audio data
            input_sample_rate: Input audio sample rate

        Returns:
            StreamingResult with complete processing results
        """
        start_time = time.time()
        processing_times = {}
        confidence_scores = {}

        try:
            # Get or create conversation context
            context_manager = session_manager.get_or_create_session(session_id)
            turn_number = len(context_manager.conversation_history) + 1

            self.logger.info(f"Processing turn {turn_number} for session {session_id}")

            # Step 1: Audio preprocessing
            preprocess_start = time.time()
            processed_audio = self.audio_processor.process_for_telephony(
                audio_data, input_sample_rate
            )
            audio_metrics = self.audio_processor.analyze_audio_quality(processed_audio)
            processing_times["audio_preprocessing"] = time.time() - preprocess_start

            # Check audio quality
            if audio_metrics.duration < self.config.min_audio_duration:
                return StreamingResult(
                    session_id=session_id,
                    turn_number=turn_number,
                    user_text="",
                    bot_response="I didn't catch that. Could you please repeat?",
                    audio_data=np.array([]),
                    processing_times=processing_times,
                    confidence_scores=confidence_scores,
                    audio_metrics=audio_metrics,
                    success=False,
                    error_message="Audio too short"
                )

            # Step 2: Speech-to-Text
            stt_start = time.time()
            stt_result = await asyncio.wait_for(
                self.stt_engine.transcribe_audio_async(processed_audio),
                timeout=self.config.stt_timeout
            )
            processing_times["stt"] = time.time() - stt_start
            confidence_scores["stt"] = stt_result.confidence

            self.logger.info(f"STT result: '{stt_result.text}' (confidence: {stt_result.confidence:.2f})")

            # Check STT confidence
            if stt_result.confidence < self.config.min_stt_confidence:
                fallback_response = "I'm having trouble understanding. Could you speak more clearly?"

                # Generate fallback TTS
                tts_result = await self._generate_tts_response(
                    fallback_response, session_id, processing_times
                )

                return StreamingResult(
                    session_id=session_id,
                    turn_number=turn_number,
                    user_text=stt_result.text,
                    bot_response=fallback_response,
                    audio_data=tts_result.audio_data,
                    processing_times=processing_times,
                    confidence_scores=confidence_scores,
                    audio_metrics=audio_metrics,
                    success=False,
                    error_message="Low STT confidence"
                )

            # Step 3: Check for escalation
            if context_manager.should_escalate():
                escalation_reason = context_manager.get_escalation_reason()
                escalation_response = self._get_escalation_response(escalation_reason)

                # Generate escalation TTS
                tts_result = await self._generate_tts_response(
                    escalation_response, session_id, processing_times
                )

                # Mark session for escalation
                context_manager.escalation_requested = True

                return StreamingResult(
                    session_id=session_id,
                    turn_number=turn_number,
                    user_text=stt_result.text,
                    bot_response=escalation_response,
                    audio_data=tts_result.audio_data,
                    processing_times=processing_times,
                    confidence_scores=confidence_scores,
                    audio_metrics=audio_metrics,
                    success=True
                )

            # Step 4: Generate LLM response
            llm_start = time.time()
            llm_prompt = context_manager.format_prompt_for_llm(stt_result.text)

            bot_response = await asyncio.wait_for(
                self.llm_engine.generate_response(llm_prompt),
                timeout=self.config.llm_timeout
            )
            processing_times["llm"] = time.time() - llm_start
            confidence_scores["llm"] = 0.9  # LLM confidence (could be enhanced)

            self.logger.info(f"LLM response: '{bot_response[:50]}...'")

            # Step 5: Generate TTS audio
            tts_result = await self._generate_tts_response(
                bot_response, session_id, processing_times
            )

            # Step 6: Update conversation context
            context_manager.add_turn(
                stt_result.text,
                bot_response,
                stt_result.confidence
            )

            # Calculate total processing time
            total_time = time.time() - start_time
            processing_times["total"] = total_time

            # Update pipeline statistics
            self._update_pipeline_stats(processing_times, True)

            self.logger.info(
                f"Turn {turn_number} completed in {total_time:.3f}s "
                f"(STT: {processing_times['stt']:.3f}s, "
                f"LLM: {processing_times['llm']:.3f}s, "
                f"TTS: {processing_times['tts']:.3f}s)"
            )

            return StreamingResult(
                session_id=session_id,
                turn_number=turn_number,
                user_text=stt_result.text,
                bot_response=bot_response,
                audio_data=tts_result.audio_data,
                processing_times=processing_times,
                confidence_scores=confidence_scores,
                audio_metrics=audio_metrics,
                success=True
            )

        except asyncio.TimeoutError as e:
            error_msg = f"Pipeline timeout: {str(e)}"
            self.logger.error(error_msg)
            self._update_pipeline_stats(processing_times, False)

            return StreamingResult(
                session_id=session_id,
                turn_number=turn_number,
                user_text="",
                bot_response="I'm experiencing technical difficulties. Please hold on.",
                audio_data=np.array([]),
                processing_times=processing_times,
                confidence_scores=confidence_scores,
                audio_metrics=AudioMetrics(0, 0, 0, 0, self.config.sample_rate, 0),
                success=False,
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.logger.error(error_msg)
            self._update_pipeline_stats(processing_times, False)

            return StreamingResult(
                session_id=session_id,
                turn_number=turn_number,
                user_text="",
                bot_response="I apologize, but I'm having technical issues. Let me transfer you to a human agent.",
                audio_data=np.array([]),
                processing_times=processing_times,
                confidence_scores=confidence_scores,
                audio_metrics=AudioMetrics(0, 0, 0, 0, self.config.sample_rate, 0),
                success=False,
                error_message=error_msg
            )

    async def _generate_tts_response(self,
                                   response_text: str,
                                   session_id: str,
                                   processing_times: Dict[str, float]) -> TTSResult:
        """Generate TTS audio for response text"""
        tts_start = time.time()

        # Use professional voice settings
        voice_options = {
            "voice": "en-us+f3",  # Professional female voice
            "speed": 160,         # Slightly faster than default
            "pitch": 45,          # Slightly lower pitch
            "amplitude": 100
        }

        tts_result = await asyncio.wait_for(
            self.tts_engine.synthesize_speech_async(
                response_text,
                voice_options=voice_options,
                request_id=f"{session_id}_{int(time.time())}"
            ),
            timeout=self.config.tts_timeout
        )

        processing_times["tts"] = time.time() - tts_start
        return tts_result

    def _get_escalation_response(self, reason: str) -> str:
        """Get appropriate escalation response based on reason"""
        escalation_responses = {
            "customer_request": "I understand you'd like to speak with a human agent. Let me transfer you to someone who can help you right away.",
            "conversation_length": "I want to make sure you get the best help possible. Let me connect you with one of our technical specialists.",
            "call_duration": "I'd like to transfer you to a human agent who can give you more detailed assistance.",
            "unknown": "Let me connect you with a human agent who can provide additional support."
        }

        return escalation_responses.get(reason, escalation_responses["unknown"])

    def _update_pipeline_stats(self, processing_times: Dict[str, float], success: bool):
        """Update pipeline performance statistics"""
        self.pipeline_stats["total_calls"] += 1

        if success:
            self.pipeline_stats["successful_calls"] += 1

            # Update timing averages
            if "stt" in processing_times:
                self.pipeline_stats["avg_stt_time"] = (
                    self.pipeline_stats["avg_stt_time"] + processing_times["stt"]
                ) / 2

            if "llm" in processing_times:
                self.pipeline_stats["avg_llm_time"] = (
                    self.pipeline_stats["avg_llm_time"] + processing_times["llm"]
                ) / 2

            if "tts" in processing_times:
                self.pipeline_stats["avg_tts_time"] = (
                    self.pipeline_stats["avg_tts_time"] + processing_times["tts"]
                ) / 2

            if "total" in processing_times:
                self.pipeline_stats["avg_total_time"] = (
                    self.pipeline_stats["avg_total_time"] + processing_times["total"]
                ) / 2

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = self.pipeline_stats.copy()

        # Add component stats
        stats["stt_stats"] = self.stt_engine.get_performance_stats()
        stats["tts_stats"] = self.tts_engine.get_performance_stats()

        # Add success rate
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0

        # Add session stats
        stats["active_sessions"] = len(session_manager.sessions)

        return stats

    async def start_streaming_session(self, session_id: str) -> AsyncGenerator[StreamingResult, np.ndarray]:
        """
        Start a streaming session for real-time audio processing

        Args:
            session_id: Unique session identifier

        Yields:
            StreamingResult objects for each processed audio chunk
        """
        self.logger.info(f"Starting streaming session: {session_id}")

        try:
            while True:
                # Wait for audio data
                audio_data = yield

                if audio_data is None:
                    break

                # Process the audio chunk
                result = await self.process_conversation_turn(session_id, audio_data)

                # Yield the result
                yield result

        except Exception as e:
            self.logger.error(f"Streaming session error: {e}")
            yield StreamingResult(
                session_id=session_id,
                turn_number=0,
                user_text="",
                bot_response="Technical error occurred",
                audio_data=np.array([]),
                processing_times={},
                confidence_scores={},
                audio_metrics=AudioMetrics(0, 0, 0, 0, self.config.sample_rate, 0),
                success=False,
                error_message=str(e)
            )

        finally:
            self.logger.info(f"Streaming session ended: {session_id}")
            session_manager.end_session(session_id)

    def save_audio_for_asterisk(self,
                               audio_data: np.ndarray,
                               output_path: str) -> str:
        """Save processed audio in format suitable for Asterisk"""
        return self.audio_processor.save_for_asterisk(
            audio_data, output_path, "wav"
        )

    def cleanup(self):
        """Cleanup all pipeline resources"""
        self.logger.info("Cleaning up streaming pipeline...")

        # Cleanup components
        if hasattr(self, 'stt_engine'):
            self.stt_engine.cleanup()

        if hasattr(self, 'tts_engine'):
            self.tts_engine.cleanup()

        # Cleanup executor
        self.executor.shutdown(wait=True)

        # Cleanup sessions
        session_manager.sessions.clear()

        self.logger.info("Pipeline cleanup completed")


# Factory function for easy pipeline creation
def create_production_pipeline() -> StreamingAIPipeline:
    """Create a production-ready streaming pipeline with optimized settings"""
    config = PipelineConfig(
        # Performance targets for production
        stt_timeout=0.25,  # 250ms STT target
        tts_timeout=1.5,   # 1.5s TTS target
        llm_timeout=3.0,   # 3s LLM timeout

        # High-quality models
        whisper_model="large",
        ollama_model="orca2:7b",
        tts_model="tts_models/en/ljspeech/tacotron2-DDC",

        # Production settings
        max_concurrent_calls=4,
        enable_streaming=True,
        enable_caching=True,

        # Quality thresholds
        min_stt_confidence=0.4,
        min_audio_duration=0.2,
        max_audio_duration=8.0
    )

    return StreamingAIPipeline(config)


if __name__ == "__main__":
    # Test the streaming pipeline
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test_pipeline():
        print("=== Testing Streaming AI Pipeline ===")

        # Create pipeline
        pipeline = create_production_pipeline()

        # Create test audio (1 second of silence with some noise)
        sample_rate = 8000
        duration = 1.0
        test_audio = 0.1 * np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Add some speech-like signal
        t = np.linspace(0, duration, len(test_audio))
        speech_signal = 0.3 * np.sin(2 * np.pi * 800 * t)  # 800 Hz tone
        test_audio += speech_signal

        print(f"Test audio: {len(test_audio)} samples at {sample_rate} Hz")

        # Process conversation turn
        session_id = "test_session_001"

        try:
            result = await pipeline.process_conversation_turn(
                session_id, test_audio, sample_rate
            )

            print(f"Processing result:")
            print(f"  Success: {result.success}")
            print(f"  User text: '{result.user_text}'")
            print(f"  Bot response: '{result.bot_response}'")
            print(f"  Processing times: {result.processing_times}")
            print(f"  Audio output length: {len(result.audio_data)} samples")

            # Get pipeline stats
            stats = pipeline.get_pipeline_stats()
            print(f"Pipeline stats: {stats}")

        except Exception as e:
            print(f"Test failed: {e}")

        finally:
            pipeline.cleanup()

        print("=== Pipeline Test Complete ===")

    # Run test
    asyncio.run(test_pipeline())
