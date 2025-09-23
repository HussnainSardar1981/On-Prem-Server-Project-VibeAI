#!/usr/bin/env python3
"""
NETOVO VoiceBot - Production GPU-Accelerated AGI Implementation
Professional-grade AI VoiceBot with H100 GPU acceleration
Replaces espeak/festival with neural TTS and integrates all components
"""

import sys
import os
import time
import logging
import asyncio
import tempfile
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# AGI library with comprehensive fallback
try:
    from pyst2 import agi
    AGI_AVAILABLE = True
    print("Successfully imported pyst2 AGI library")
except ImportError as e:
    print(f"pyst2 import failed: {e}")
    try:
        import asterisk.agi as agi
        AGI_AVAILABLE = True
        print("Using asterisk.agi library as fallback")
    except ImportError as e2:
        print(f"asterisk.agi also failed: {e2}")
        print("Creating fallback AGI class for testing and compatibility")
        AGI_AVAILABLE = "fallback"

        # Create a comprehensive fallback AGI class
        class AGI:
            """Fallback AGI implementation for testing and compatibility"""

            def __init__(self, file=None):
                self.env = {}
                self._file = file or sys.stdin
                self.logger = logging.getLogger("FallbackAGI")
                self.logger.info("AGI: Initialized fallback AGI implementation")

                # Read AGI environment if available
                try:
                    while True:
                        line = self._file.readline().strip()
                        if not line:
                            break
                        if ':' in line:
                            key, value = line.split(':', 1)
                            self.env[key.strip()] = value.strip()
                except:
                    pass  # No stdin available in test mode

            def execute(self, command, *args):
                """Execute AGI command with logging"""
                cmd = f"{command} {' '.join(map(str, args))}"
                self.logger.info(f"AGI: {cmd}")
                print(f"AGI: {cmd}")
                return {'result': '0', 'data': ''}

            def answer(self):
                """Answer the call"""
                return self.execute("ANSWER")

            def stream_file(self, filename, escape_digits=""):
                """Stream audio file"""
                return self.execute("STREAM FILE", filename, escape_digits)

            def record_file(self, filename, format='wav', escape_digits='#', timeout=5000, offset=0, beep=1):
                """Record audio file"""
                return self.execute("RECORD FILE", filename, format, escape_digits, timeout, offset, beep)

            def get_data(self, filename, timeout=5000, max_digits=255):
                """Get DTMF data"""
                return self.execute("GET DATA", filename, timeout, max_digits)

            def say_number(self, number, escape_digits=""):
                """Say a number"""
                return self.execute("SAY NUMBER", number, escape_digits)

            def say_alpha(self, text, escape_digits=""):
                """Say alpha characters"""
                return self.execute("SAY ALPHA", text, escape_digits)

            def hangup(self, channel=""):
                """Hang up the call"""
                return self.execute("HANGUP", channel)

            def set_variable(self, name, value):
                """Set a channel variable"""
                return self.execute("SET VARIABLE", name, value)

            def get_variable(self, name):
                """Get a channel variable"""
                return self.execute("GET VARIABLE", name)

            def verbose(self, message, level=1):
                """Send verbose message"""
                return self.execute("VERBOSE", f'"{message}"', level)

            def noop(self, message=""):
                """No operation with message"""
                return self.execute("NOOP", f'"{message}"')

        # Create the agi module-like interface
        agi = AGI

# Import our streaming pipeline
from streaming_ai_pipeline import StreamingAIPipeline, create_production_pipeline, PipelineConfig
from conversation_context_manager import session_manager
import numpy as np
import soundfile as sf


class ProductionVoiceBotConfig:
    """Configuration for production VoiceBot deployment"""

    # Call management
    max_call_duration = 300  # 5 minutes
    max_conversation_turns = 8
    max_silent_attempts = 3
    record_timeout = 8  # seconds
    silence_threshold = 2  # seconds

    # Audio settings
    sample_rate = 8000
    min_recording_size = 1000  # bytes
    audio_format = "wav"

    # GPU Performance settings
    enable_gpu_acceleration = True
    gpu_warmup = True
    concurrent_processing = True

    # Quality thresholds
    min_stt_confidence = 0.4
    min_audio_duration = 0.2
    max_audio_duration = 8.0

    # Professional responses
    greeting_message = "Hello! I'm Alexis, your AI assistant from NETOVO IT Services. How can I help you today?"
    escalation_message = "I'd be happy to connect you with one of our technical specialists. Please hold while I transfer your call."
    technical_difficulty_message = "I'm experiencing technical difficulties. Let me transfer you to a human agent immediately."
    goodbye_message = "Thank you for calling NETOVO. Have a great day!"


class NetovoGPUVoiceBot:
    """
    Production-grade GPU-accelerated VoiceBot for NETOVO
    Integrates all components: GPU STT, Neural TTS, Conversation Management
    """

    def __init__(self, agi_instance):
        """Initialize the VoiceBot with AGI instance"""
        self.agi = agi_instance
        self.config = ProductionVoiceBotConfig()

        # Setup logging
        self._setup_logging()

        # Initialize session tracking
        self.session_id = self._generate_session_id()
        self.call_start_time = time.time()
        self.conversation_turns = 0
        self.silent_attempts = 0

        # Performance tracking
        self.performance_metrics = {
            "call_start": self.call_start_time,
            "total_processing_time": 0.0,
            "avg_turn_time": 0.0,
            "stt_times": [],
            "tts_times": [],
            "llm_times": []
        }

        # Initialize GPU streaming pipeline
        self.logger.info("Initializing GPU-accelerated streaming pipeline...")
        try:
            self.pipeline = create_production_pipeline()
            self.logger.info("Streaming pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise

        self.logger.info(f"NETOVO GPU VoiceBot initialized for session {self.session_id}")

    def _setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        log_dir = Path("/opt/voicebot/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_file = log_dir / f"netovo_gpu_voicebot_{int(time.time())}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("NetovoGPUVoiceBot")
        self.logger.info("Logging initialized")

    def _generate_session_id(self) -> str:
        """Generate unique session ID for this call"""
        timestamp = int(time.time())
        return f"netovo_call_{timestamp}_{os.getpid()}"

    def log_call_info(self, message: str):
        """Log call information with session context"""
        self.agi.verbose(f"NETOVO-GPU: {message}")
        self.logger.info(f"[{self.session_id}] {message}")

    def should_continue_call(self) -> bool:
        """Check if call should continue based on limits"""
        # Check call duration
        call_duration = time.time() - self.call_start_time
        if call_duration > self.config.max_call_duration:
            self.log_call_info(f"Call duration limit reached: {call_duration:.1f}s")
            return False

        # Check conversation turns
        if self.conversation_turns >= self.config.max_conversation_turns:
            self.log_call_info(f"Conversation turn limit reached: {self.conversation_turns}")
            return False

        # Check silent attempts
        if self.silent_attempts >= self.config.max_silent_attempts:
            self.log_call_info(f"Silent attempt limit reached: {self.silent_attempts}")
            return False

        return True

    def record_customer_audio(self) -> Optional[np.ndarray]:
        """Record audio from customer with enhanced error handling"""
        record_name = f"/tmp/customer_input_{self.session_id}_{int(time.time())}"

        try:
            self.log_call_info("Recording customer audio...")

            # Record audio using AGI
            result = self.agi.record_file(
                record_name,
                format=self.config.audio_format,
                escape_digits='#*',
                timeout=self.config.record_timeout * 1000,  # Convert to milliseconds
                offset=0,
                beep=True,
                silence=self.config.silence_threshold
            )

            self.log_call_info(f"Recording result: {result}")

            # Check if recording file exists and has content
            audio_file = f"{record_name}.{self.config.audio_format}"
            audio_path = Path(audio_file)

            if not audio_path.exists():
                self.log_call_info("Recording file does not exist")
                return None

            file_size = audio_path.stat().st_size
            if file_size < self.config.min_recording_size:
                self.log_call_info(f"Recording too small: {file_size} bytes")
                self.silent_attempts += 1
                return None

            # Load audio data
            try:
                audio_data, sample_rate = sf.read(audio_file)
                self.log_call_info(f"Audio loaded: {len(audio_data)} samples at {sample_rate} Hz")

                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Convert to float32
                audio_data = audio_data.astype(np.float32)

                # Reset silent attempts on successful recording
                self.silent_attempts = 0

                return audio_data

            except Exception as e:
                self.log_call_info(f"Failed to load audio file: {e}")
                return None

        except Exception as e:
            self.log_call_info(f"Recording failed: {e}")
            return None

        finally:
            # Cleanup recording file
            try:
                audio_file = f"{record_name}.{self.config.audio_format}"
                if Path(audio_file).exists():
                    Path(audio_file).unlink()
            except:
                pass

    def play_response_audio(self, audio_data: np.ndarray) -> bool:
        """Play response audio through Asterisk with proper telephony format conversion"""
        try:
            if len(audio_data) == 0:
                self.log_call_info("No audio data to play")
                return False

            # Create unique filenames
            timestamp = int(time.time())
            temp_raw = f"/tmp/voicebot_raw_{timestamp}.wav"
            asterisk_file = f"/var/lib/asterisk/sounds/voicebot_{timestamp}"

            try:
                # Step 1: Save GPU audio as raw file
                import soundfile as sf

                # Ensure audio is float32 and normalized
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize to prevent clipping
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95

                # Save raw audio (assume 22050 Hz from neural TTS)
                sf.write(temp_raw, audio_data, 22050)

                self.log_call_info(f"Raw audio saved: {temp_raw} ({len(audio_data)} samples)")

                # Step 2: Convert to telephony format using sox
                import subprocess
                conversion_cmd = [
                    'sox', temp_raw,
                    '-r', '8000',    # 8kHz sample rate for telephony
                    '-c', '1',       # Mono
                    '-b', '16',      # 16-bit
                    f"{asterisk_file}.wav"
                ]

                subprocess.run(conversion_cmd, check=True, timeout=10)
                self.log_call_info(f"Audio converted for telephony: {asterisk_file}.wav")

                # Step 3: Play through AGI (without .wav extension)
                result = self.agi.stream_file(f"voicebot_{timestamp}", "")
                self.log_call_info(f"AGI stream_file result: {result}")

                # Check if file exists and size
                if os.path.exists(f"{asterisk_file}.wav"):
                    file_size = os.path.getsize(f"{asterisk_file}.wav")
                    self.log_call_info(f"Final audio file size: {file_size} bytes")

                return True

            finally:
                # Cleanup files with delay to ensure Asterisk finishes playing
                import threading
                def delayed_cleanup():
                    time.sleep(3)  # Wait for playback to complete
                    try:
                        if os.path.exists(temp_raw):
                            os.remove(temp_raw)
                        if os.path.exists(f"{asterisk_file}.wav"):
                            os.remove(f"{asterisk_file}.wav")
                        self.log_call_info("Audio files cleaned up")
                    except Exception as cleanup_error:
                        self.log_call_info(f"Cleanup warning: {cleanup_error}")

                cleanup_thread = threading.Thread(target=delayed_cleanup)
                cleanup_thread.daemon = True
                cleanup_thread.start()

        except subprocess.CalledProcessError as e:
            self.log_call_info(f"Audio conversion failed: {e}")
            # Fallback to espeak
            return self.play_espeak_fallback(audio_data)
        except Exception as e:
            self.log_call_info(f"Audio playback failed: {e}")
            return False

    def play_espeak_fallback(self, audio_data: np.ndarray = None, text: str = "") -> bool:
        """Fallback audio using espeak when GPU pipeline fails"""
        try:
            if not text and audio_data is not None:
                # If we have audio data but no text, use generic message
                text = "Please hold while we process your request."

            if not text:
                text = "I'm sorry, I'm having audio difficulties."

            timestamp = int(time.time())
            espeak_file = f"/var/lib/asterisk/sounds/espeak_{timestamp}"

            # Generate audio with espeak directly in telephony format
            import subprocess
            espeak_cmd = [
                'espeak', text,
                '-s', '160',    # Speed
                '-p', '45',     # Pitch
                '-a', '100',    # Amplitude
                '-w', f"{espeak_file}.wav"
            ]

            subprocess.run(espeak_cmd, check=True, timeout=10)
            self.log_call_info(f"Espeak audio generated: {espeak_file}.wav")

            # Convert to proper telephony format
            conversion_cmd = [
                'sox', f"{espeak_file}.wav",
                '-r', '8000', '-c', '1', '-b', '16',
                f"{espeak_file}_final.wav"
            ]

            subprocess.run(conversion_cmd, check=True, timeout=10)

            # Play the final file
            result = self.agi.stream_file(f"espeak_{timestamp}_final", "")
            self.log_call_info(f"Espeak fallback playback result: {result}")

            # Cleanup
            import threading
            def cleanup():
                time.sleep(3)
                try:
                    if os.path.exists(f"{espeak_file}.wav"):
                        os.remove(f"{espeak_file}.wav")
                    if os.path.exists(f"{espeak_file}_final.wav"):
                        os.remove(f"{espeak_file}_final.wav")
                except:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

            return True

        except Exception as e:
            self.log_call_info(f"Espeak fallback failed: {e}")
            return False

    def play_text_message(self, message: str) -> bool:
        """Play text message using GPU-accelerated neural TTS"""
        try:
            self.log_call_info(f"Generating TTS for: '{message[:50]}...'")

            # Generate high-quality TTS using our neural engine
            tts_start = time.time()

            # Create voice options for professional female voice
            voice_options = {
                "voice": "en-us+f3",  # Professional female voice
                "speed": 160,         # Slightly faster for efficiency
                "pitch": 45,          # Professional pitch
                "amplitude": 100
            }

            # Use the TTS engine directly for immediate response
            tts_result = self.pipeline.tts_engine.synthesize_speech(
                message,
                voice_options=voice_options,
                request_id=f"{self.session_id}_tts_{int(time.time())}"
            )

            tts_time = time.time() - tts_start
            self.performance_metrics["tts_times"].append(tts_time)

            self.log_call_info(f"TTS generated in {tts_time:.3f}s")

            # Play the generated audio
            success = self.play_response_audio(tts_result.audio_data)

            if success:
                self.log_call_info("TTS playback successful")
                return True
            else:
                self.log_call_info("TTS playback failed - trying espeak fallback")
                return self.play_espeak_fallback(text=message)

        except Exception as e:
            self.log_call_info(f"TTS generation failed: {e} - trying espeak fallback")
            return self.play_espeak_fallback(text=message)

    async def process_conversation_turn_async(self, audio_data: np.ndarray) -> bool:
        """Process a complete conversation turn using the streaming pipeline"""
        try:
            turn_start = time.time()

            self.log_call_info(f"Processing conversation turn {self.conversation_turns + 1}")

            # Process through the streaming pipeline
            result = await self.pipeline.process_conversation_turn(
                self.session_id,
                audio_data,
                self.config.sample_rate
            )

            turn_time = time.time() - turn_start
            self.performance_metrics["total_processing_time"] += turn_time

            # Update performance metrics
            if "stt" in result.processing_times:
                self.performance_metrics["stt_times"].append(result.processing_times["stt"])

            if "llm" in result.processing_times:
                self.performance_metrics["llm_times"].append(result.processing_times["llm"])

            if "tts" in result.processing_times:
                self.performance_metrics["tts_times"].append(result.processing_times["tts"])

            # Log processing results
            self.log_call_info(
                f"Turn processed in {turn_time:.3f}s - "
                f"STT: '{result.user_text}' -> Bot: '{result.bot_response[:50]}...'"
            )

            if result.success:
                # Play the response
                if len(result.audio_data) > 0:
                    success = self.play_response_audio(result.audio_data)
                else:
                    # Fallback to direct TTS if no audio generated
                    success = self.play_text_message(result.bot_response)

                self.conversation_turns += 1
                return success

            else:
                # Handle processing failure
                self.log_call_info(f"Processing failed: {result.error_message}")

                if "confidence" in result.error_message or "audio too short" in result.error_message:
                    # Audio quality issue - increment silent attempts
                    self.silent_attempts += 1

                # Play fallback response
                if result.bot_response:
                    self.play_text_message(result.bot_response)
                else:
                    self.play_text_message(self.config.technical_difficulty_message)

                return False

        except Exception as e:
            self.log_call_info(f"Conversation turn processing failed: {e}")
            traceback.print_exc()
            return False

    def process_conversation_turn(self, audio_data: np.ndarray) -> bool:
        """Synchronous wrapper for conversation turn processing"""
        try:
            # Run async processing in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                return loop.run_until_complete(
                    self.process_conversation_turn_async(audio_data)
                )
            finally:
                loop.close()

        except Exception as e:
            self.log_call_info(f"Async processing failed: {e}")
            return False

    def transfer_to_human(self, reason: str) -> bool:
        """Transfer call to human agent"""
        try:
            self.log_call_info(f"Transferring to human agent - Reason: {reason}")

            # Play escalation message
            self.play_text_message(self.config.escalation_message)

            # Execute transfer (this would be configured in Asterisk dialplan)
            # For now, we'll just log and hangup gracefully
            self.log_call_info("Call transfer initiated")

            return True

        except Exception as e:
            self.log_call_info(f"Transfer failed: {e}")
            return False

    def log_performance_metrics(self):
        """Log comprehensive performance metrics"""
        call_duration = time.time() - self.call_start_time

        if self.conversation_turns > 0:
            avg_turn_time = self.performance_metrics["total_processing_time"] / self.conversation_turns
        else:
            avg_turn_time = 0

        metrics = {
            "session_id": self.session_id,
            "call_duration": call_duration,
            "conversation_turns": self.conversation_turns,
            "avg_turn_time": avg_turn_time,
            "total_processing_time": self.performance_metrics["total_processing_time"],
            "silent_attempts": self.silent_attempts
        }

        if self.performance_metrics["stt_times"]:
            metrics["avg_stt_time"] = np.mean(self.performance_metrics["stt_times"])

        if self.performance_metrics["tts_times"]:
            metrics["avg_tts_time"] = np.mean(self.performance_metrics["tts_times"])

        if self.performance_metrics["llm_times"]:
            metrics["avg_llm_time"] = np.mean(self.performance_metrics["llm_times"])

        self.log_call_info(f"Performance metrics: {metrics}")

    def handle_call(self):
        """Main call handling logic"""
        try:
            self.log_call_info("=== NETOVO GPU VoiceBot Call Started ===")

            # Initial greeting (only for first interaction)
            context_manager = session_manager.get_or_create_session(self.session_id)

            if context_manager.should_greet():
                self.log_call_info("Playing initial greeting")
                success = self.play_text_message(self.config.greeting_message)

                if not success:
                    self.log_call_info("Failed to play greeting - continuing anyway")

            # Main conversation loop
            while self.should_continue_call():
                self.log_call_info(f"Starting conversation turn {self.conversation_turns + 1}")

                # Record customer input
                audio_data = self.record_customer_audio()

                if audio_data is None:
                    self.log_call_info("No valid audio recorded")

                    if self.silent_attempts >= self.config.max_silent_attempts:
                        self.log_call_info("Maximum silent attempts reached")
                        break

                    continue

                # Process the conversation turn
                success = self.process_conversation_turn(audio_data)

                if not success:
                    self.log_call_info("Conversation turn processing failed")

                # Check if escalation is needed
                context_manager = session_manager.sessions.get(self.session_id)
                if context_manager and context_manager.should_escalate():
                    reason = context_manager.get_escalation_reason()
                    self.log_call_info(f"Escalation needed: {reason}")
                    self.transfer_to_human(reason)
                    break

            # End of conversation
            self.log_call_info("Conversation ended - playing goodbye message")
            self.play_text_message(self.config.goodbye_message)

        except Exception as e:
            self.log_call_info(f"Call handling error: {e}")
            traceback.print_exc()

            # Play technical difficulty message
            try:
                self.play_text_message(self.config.technical_difficulty_message)
            except:
                pass

        finally:
            # Log performance metrics
            self.log_performance_metrics()

            # Cleanup
            self.cleanup()

            self.log_call_info("=== NETOVO GPU VoiceBot Call Ended ===")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.log_call_info("Cleaning up VoiceBot resources...")

            # End session
            session_manager.end_session(self.session_id)

            # Get final pipeline stats
            pipeline_stats = self.pipeline.get_pipeline_stats()
            self.log_call_info(f"Final pipeline stats: {pipeline_stats}")

            # Note: Don't cleanup the pipeline itself as it may be shared
            # across multiple calls in a production environment

            self.log_call_info("Cleanup completed")

        except Exception as e:
            self.log_call_info(f"Cleanup error: {e}")


def signal_handler(signum, frame):
    """Handle system signals gracefully"""
    print(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point for AGI script"""
    if not AGI_AVAILABLE:
        print("ERROR: pyst2 library not available")
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize AGI
        agi_instance = agi.AGI()

        # Create and run VoiceBot
        voicebot = NetovoGPUVoiceBot(agi_instance)
        voicebot.handle_call()

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
