"""
Professional audio processing module
Handles resampling, VAD, and format conversions
"""

import numpy as np
import soxr
import webrtcvad
import audioop
import structlog

logger = structlog.get_logger()

class AudioProcessor:
    """Handles all audio processing tasks"""

    def __init__(self):
        # Initialize VAD for barge-in detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
        self.vad_frame_duration = 20  # ms

    def resample(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
        """High-quality resampling using soxr"""
        if input_rate == output_rate:
            return audio

        try:
            resampled = soxr.resample(audio, input_rate, output_rate)
            logger.debug("Audio resampled",
                        input_rate=input_rate,
                        output_rate=output_rate,
                        input_len=len(audio),
                        output_len=len(resampled))
            return resampled
        except Exception as e:
            logger.error("Resampling failed", error=str(e))
            raise

    def ulaw_to_pcm16(self, ulaw_data: bytes) -> np.ndarray:
        """Convert μ-law to 16-bit PCM using stdlib"""
        pcm_bytes = audioop.ulaw2lin(ulaw_data, 2)  # 2 bytes = 16-bit
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        # Convert to float32 in range [-1, 1]
        return pcm_array.astype(np.float32) / 32768.0

    def pcm16_to_ulaw(self, pcm_array: np.ndarray) -> bytes:
        """Convert 16-bit PCM to μ-law using stdlib"""
        # Convert float32 [-1, 1] to int16
        pcm_int16 = (pcm_array * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
        return audioop.lin2ulaw(pcm_bytes, 2)

    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Detect speech using WebRTC VAD"""
        # VAD expects 16-bit PCM at specific rates (8000, 16000, 32000, 48000)
        if sample_rate not in [8000, 16000, 32000, 48000]:
            # Resample to nearest supported rate
            target_rate = min([8000, 16000, 32000, 48000],
                            key=lambda x: abs(x - sample_rate))
            audio = self.resample(audio, sample_rate, target_rate)
            sample_rate = target_rate

        # Convert to 16-bit PCM bytes
        pcm_int16 = (audio * 32767).astype(np.int16)

        # VAD frame size calculation
        frame_length = int(sample_rate * self.vad_frame_duration / 1000)

        # Process in frames
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(pcm_int16) - frame_length, frame_length):
            frame = pcm_int16[i:i + frame_length].tobytes()
            try:
                if self.vad.is_speech(frame, sample_rate):
                    speech_frames += 1
                total_frames += 1
            except:
                # Skip invalid frames
                continue

        # Return True if >30% of frames contain speech
        if total_frames == 0:
            return False

        speech_ratio = speech_frames / total_frames
        return speech_ratio > 0.3

class AudioBuffer:
    """Circular buffer for audio processing"""

    def __init__(self, max_duration_seconds: float = 10.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0

    def add_audio(self, audio: np.ndarray):
        """Add audio to circular buffer"""
        audio_len = len(audio)

        if audio_len > self.max_samples:
            # Audio longer than buffer, take the last part
            audio = audio[-self.max_samples:]
            audio_len = self.max_samples

        # Handle wraparound
        end_pos = self.write_pos + audio_len
        if end_pos <= self.max_samples:
            self.buffer[self.write_pos:end_pos] = audio
        else:
            # Split across wraparound
            first_part = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = audio[:first_part]
            self.buffer[:audio_len - first_part] = audio[first_part:]

        self.write_pos = end_pos % self.max_samples
        self.samples_written = min(self.samples_written + audio_len, self.max_samples)

    def get_latest(self, duration_seconds: float) -> np.ndarray:
        """Get the latest N seconds of audio"""
        samples_needed = int(duration_seconds * self.sample_rate)
        samples_needed = min(samples_needed, self.samples_written)

        if samples_needed == 0:
            return np.array([], dtype=np.float32)

        start_pos = (self.write_pos - samples_needed) % self.max_samples

        if start_pos + samples_needed <= self.max_samples:
            return self.buffer[start_pos:start_pos + samples_needed].copy()
        else:
            # Handle wraparound
            first_part = self.max_samples - start_pos
            result = np.zeros(samples_needed, dtype=np.float32)
            result[:first_part] = self.buffer[start_pos:]
            result[first_part:] = self.buffer[:samples_needed - first_part]
            return result