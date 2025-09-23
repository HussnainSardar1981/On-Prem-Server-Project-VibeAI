"""
Main Voice Bot Application
Orchestrates the entire voice pipeline
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

from riva_client import RivaASRClient, RivaTTSClient
from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
from audio_processing import AudioProcessor, AudioBuffer
from conversation_context import ConversationContext

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class VoiceBot:
    """Main Voice Bot orchestrator"""

    def __init__(self):
        # Initialize components
        self.riva_asr = RivaASRClient(
            server_url=os.getenv("RIVA_SERVER", "localhost:50051")
        )
        self.riva_tts = RivaTTSClient(
            server_url=os.getenv("RIVA_SERVER", "localhost:50051")
        )
        self.ollama = OllamaClient(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "orca2:7b")
        )
        self.audio_processor = AudioProcessor()
        self.conversation = ConversationContext()

        # Audio settings
        self.telephony_rate = 8000  # 3CX/telephony
        self.asr_rate = 16000       # Riva ASR
        self.tts_rate = 22050       # Riva TTS

        self.running = False

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Voice Bot components...")

        try:
            # Connect to Riva services
            await asyncio.get_event_loop().run_in_executor(None, self.riva_asr.connect)
            await asyncio.get_event_loop().run_in_executor(None, self.riva_tts.connect)

            # Test Ollama connection
            if not self.ollama.health_check():
                raise ConnectionError("Ollama not available")

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error("Initialization failed", error=str(e))
            return False

    async def process_audio_file(self, audio_file: Path) -> Path:
        """
        Test pipeline: WAV file → Riva ASR → LLM → Riva TTS → WAV file
        This tests the core pipeline without RTP complexity
        """
        logger.info("Processing audio file", file=str(audio_file))

        try:
            # TODO: Load audio file
            # audio_data = load_wav_file(audio_file)

            # TODO: Process through ASR
            # transcript = await self.process_asr(audio_data)

            # TODO: Get LLM response
            # response = await self.get_llm_response(transcript)

            # TODO: Synthesize TTS
            # tts_audio = await self.process_tts(response)

            # TODO: Save output file
            # output_file = audio_file.parent / f"output_{audio_file.name}"
            # save_wav_file(tts_audio, output_file)

            logger.info("Audio file processed successfully")
            # return output_file

        except Exception as e:
            logger.error("Audio processing failed", error=str(e))
            raise

    async def process_asr(self, audio_data):
        """Process audio through Riva ASR"""
        # TODO: Implement ASR processing
        logger.info("Processing ASR", audio_length=len(audio_data) if audio_data is not None else 0)
        return "Hello, this is a test transcription"

    async def get_llm_response(self, transcript: str) -> str:
        """Get response from Ollama LLM"""
        try:
            # Build context-aware prompt
            prompt = self.conversation.build_prompt(transcript, VOICE_BOT_SYSTEM_PROMPT)

            # Generate response
            response = await self.ollama.generate(
                prompt=transcript,
                system_prompt=VOICE_BOT_SYSTEM_PROMPT,
                max_tokens=50  # Keep responses short for telephony
            )

            # Add to conversation history
            self.conversation.add_turn(transcript, response)

            logger.info("LLM response generated",
                       input=transcript[:50],
                       output=response[:50])

            return response

        except Exception as e:
            logger.error("LLM processing failed", error=str(e))
            # Fallback response
            return "I'm sorry, I'm having trouble processing your request. Please try again."

    async def process_tts(self, text: str):
        """Process text through Riva TTS"""
        # TODO: Implement TTS processing
        logger.info("Processing TTS", text=text[:50])
        return self.riva_tts.synthesize(text)

    async def handle_call(self, call_context):
        """Handle a single phone call"""
        logger.info("Handling new call", context=call_context)

        try:
            # Reset conversation for new call
            self.conversation.reset_session()

            # TODO: Implement call handling logic
            # - Accept RTP stream from 3CX
            # - Process audio in real-time
            # - Handle barge-in
            # - Send TTS back to caller

            pass

        except Exception as e:
            logger.error("Call handling failed", error=str(e))
        finally:
            logger.info("Call ended")

    async def start_server(self):
        """Start the voice bot server"""
        logger.info("Starting Voice Bot server...")

        self.running = True

        try:
            while self.running:
                # TODO: Listen for incoming calls from 3CX/ARI
                # For now, just keep running
                await asyncio.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Voice Bot...")
        self.running = False

        # Close connections
        if hasattr(self.ollama, 'close'):
            self.ollama.close()

        logger.info("Voice Bot shutdown complete")

def setup_signal_handlers(voice_bot):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        voice_bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    logger.info("Starting Professional Voice Bot for NETOVO")

    voice_bot = VoiceBot()
    setup_signal_handlers(voice_bot)

    # Initialize components
    if not await voice_bot.initialize():
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    # Start server
    await voice_bot.start_server()

if __name__ == "__main__":
    asyncio.run(main())