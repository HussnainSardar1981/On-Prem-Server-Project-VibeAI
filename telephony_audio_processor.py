#!/usr/bin/env python3
"""
NETOVO VoiceBot - Telephony Audio Processing Pipeline
Optimized audio processing for 8kHz telephony with G.711 codec support
Handles ulaw/alaw conversion, noise reduction, and audio enhancement
"""

import numpy as np
import soundfile as sf
import scipy.signal as signal
import logging
import time
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import struct
import tempfile


@dataclass
class AudioMetrics:
    """Audio quality metrics"""
    rms_level: float
    peak_level: float
    snr_estimate: float
    duration: float
    sample_rate: int
    dynamic_range: float


class TelephonyAudioProcessor:
    """
    Optimized audio processor for telephony applications.
    Handles conversion between different formats, noise reduction,
    and quality enhancement for 8kHz G.711 telephony.
    """

    def __init__(self, target_sample_rate: int = 8000):
        """
        Initialize telephony audio processor

        Args:
            target_sample_rate: Target sample rate for telephony (8000 Hz standard)
        """
        self.logger = logging.getLogger(__name__)
        self.target_sample_rate = target_sample_rate

        # Telephony band limits (300-3400 Hz for G.711)
        self.telephony_low_freq = 300
        self.telephony_high_freq = 3400

        # Audio processing parameters
        self.silence_threshold = -40  # dB
        self.noise_floor_db = -60
        self.target_rms_db = -20

        # G.711 codec parameters
        self.g711_max_value = 32767  # 16-bit signed max
        self.ulaw_bias = 0x84
        self.alaw_bias = 0x55

        # Initialize filters
        self._setup_filters()

    def _setup_filters(self):
        """Setup audio filters for telephony processing"""
        nyquist = self.target_sample_rate / 2

        # Telephony bandpass filter (300-3400 Hz)
        low_cutoff = self.telephony_low_freq / nyquist
        high_cutoff = self.telephony_high_freq / nyquist

        # Design Butterworth bandpass filter
        self.telephony_filter = signal.butter(
            6,  # 6th order for good rolloff
            [low_cutoff, high_cutoff],
            btype='band',
            output='sos'  # Second-order sections for numerical stability
        )

        # Pre-emphasis filter for speech clarity
        self.preemphasis_coeff = 0.97

        # De-emphasis filter (inverse of pre-emphasis)
        self.deemphasis_coeff = 0.97

        self.logger.info("Audio filters initialized for telephony processing")

    def resample_audio(self,
                      audio_data: np.ndarray,
                      original_sample_rate: int,
                      target_sample_rate: Optional[int] = None) -> np.ndarray:
        """
        High-quality audio resampling optimized for speech

        Args:
            audio_data: Input audio data
            original_sample_rate: Original sample rate
            target_sample_rate: Target sample rate (default: self.target_sample_rate)

        Returns:
            Resampled audio data
        """
        if target_sample_rate is None:
            target_sample_rate = self.target_sample_rate

        if original_sample_rate == target_sample_rate:
            return audio_data

        try:
            # Calculate resampling ratio
            ratio = target_sample_rate / original_sample_rate
            num_samples = int(len(audio_data) * ratio)

            # Use scipy's high-quality resampling
            resampled = signal.resample(audio_data, num_samples)

            self.logger.debug(
                f"Resampled audio: {original_sample_rate}Hz → {target_sample_rate}Hz "
                f"({len(audio_data)} → {len(resampled)} samples)"
            )

            return resampled.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Audio resampling failed: {e}")
            raise

    def apply_telephony_filter(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply telephony bandpass filter (300-3400 Hz)

        Args:
            audio_data: Input audio data

        Returns:
            Filtered audio data
        """
        try:
            # Apply bandpass filter using second-order sections
            filtered_audio = signal.sosfiltfilt(self.telephony_filter, audio_data)

            return filtered_audio.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Telephony filtering failed: {e}")
            return audio_data

    def apply_preemphasis(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to enhance high frequencies for speech clarity

        Args:
            audio_data: Input audio data

        Returns:
            Pre-emphasized audio data
        """
        try:
            # Pre-emphasis: y[n] = x[n] - α * x[n-1]
            emphasized = np.zeros_like(audio_data)
            emphasized[0] = audio_data[0]
            emphasized[1:] = audio_data[1:] - self.preemphasis_coeff * audio_data[:-1]

            return emphasized.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Pre-emphasis failed: {e}")
            return audio_data

    def apply_deemphasis(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply de-emphasis filter (inverse of pre-emphasis)

        Args:
            audio_data: Input audio data

        Returns:
            De-emphasized audio data
        """
        try:
            # De-emphasis: y[n] = x[n] + α * y[n-1]
            deemphasized = np.zeros_like(audio_data)
            deemphasized[0] = audio_data[0]

            for n in range(1, len(audio_data)):
                deemphasized[n] = audio_data[n] + self.deemphasis_coeff * deemphasized[n-1]

            return deemphasized.astype(np.float32)

        except Exception as e:
            self.logger.error(f"De-emphasis failed: {e}")
            return audio_data

    def noise_reduction(self, audio_data: np.ndarray, reduction_db: float = 10) -> np.ndarray:
        """
        Simple spectral noise reduction for telephony audio

        Args:
            audio_data: Input audio data
            reduction_db: Noise reduction in dB

        Returns:
            Noise-reduced audio data
        """
        try:
            # Convert to frequency domain
            fft_audio = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_audio)
            phase = np.angle(fft_audio)

            # Estimate noise floor (lowest 10% of magnitudes)
            sorted_mag = np.sort(magnitude)
            noise_floor = np.mean(sorted_mag[:len(sorted_mag)//10])

            # Calculate reduction factor
            reduction_factor = 10 ** (-reduction_db / 20)

            # Apply spectral subtraction
            enhanced_magnitude = magnitude.copy()
            noise_mask = magnitude < (noise_floor * 3)  # Identify noise regions
            enhanced_magnitude[noise_mask] *= reduction_factor

            # Reconstruct audio
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.fft.irfft(enhanced_fft, len(audio_data))

            return enhanced_audio.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Noise reduction failed: {e}")
            return audio_data

    def normalize_audio_level(self,
                            audio_data: np.ndarray,
                            target_rms_db: Optional[float] = None) -> np.ndarray:
        """
        Normalize audio to target RMS level

        Args:
            audio_data: Input audio data
            target_rms_db: Target RMS level in dB (default: self.target_rms_db)

        Returns:
            Normalized audio data
        """
        if target_rms_db is None:
            target_rms_db = self.target_rms_db

        try:
            # Calculate current RMS
            current_rms = np.sqrt(np.mean(audio_data ** 2))

            if current_rms > 0:
                # Calculate target RMS in linear scale
                target_rms_linear = 10 ** (target_rms_db / 20)

                # Calculate gain
                gain = target_rms_linear / current_rms

                # Apply gain with limiting
                normalized = audio_data * gain
                normalized = np.clip(normalized, -1.0, 1.0)

                self.logger.debug(f"Audio normalized: gain={gain:.3f}, target_rms={target_rms_db}dB")
                return normalized.astype(np.float32)
            else:
                return audio_data

        except Exception as e:
            self.logger.error(f"Audio normalization failed: {e}")
            return audio_data

    def apply_dynamic_range_compression(self,
                                      audio_data: np.ndarray,
                                      threshold_db: float = -20,
                                      ratio: float = 4.0,
                                      attack_time: float = 0.003,
                                      release_time: float = 0.1) -> np.ndarray:
        """
        Apply dynamic range compression for consistent audio levels

        Args:
            audio_data: Input audio data
            threshold_db: Compression threshold in dB
            ratio: Compression ratio
            attack_time: Attack time in seconds
            release_time: Release time in seconds

        Returns:
            Compressed audio data
        """
        try:
            # Convert threshold to linear scale
            threshold_linear = 10 ** (threshold_db / 20)

            # Calculate attack and release coefficients
            attack_coeff = np.exp(-1 / (attack_time * self.target_sample_rate))
            release_coeff = np.exp(-1 / (release_time * self.target_sample_rate))

            # Initialize envelope follower
            envelope = 0.0
            compressed = np.zeros_like(audio_data)

            for i in range(len(audio_data)):
                # Envelope detection
                input_level = abs(audio_data[i])

                if input_level > envelope:
                    envelope = attack_coeff * envelope + (1 - attack_coeff) * input_level
                else:
                    envelope = release_coeff * envelope + (1 - release_coeff) * input_level

                # Calculate gain reduction
                if envelope > threshold_linear:
                    gain_reduction = threshold_linear + (envelope - threshold_linear) / ratio
                    gain = gain_reduction / envelope if envelope > 0 else 1.0
                else:
                    gain = 1.0

                # Apply compression
                compressed[i] = audio_data[i] * gain

            return compressed.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Dynamic range compression failed: {e}")
            return audio_data

    def remove_silence(self,
                      audio_data: np.ndarray,
                      silence_threshold_db: Optional[float] = None,
                      min_silence_duration: float = 0.1,
                      padding_duration: float = 0.05) -> np.ndarray:
        """
        Remove silence from audio while preserving natural pauses

        Args:
            audio_data: Input audio data
            silence_threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration to remove (seconds)
            padding_duration: Padding to keep around speech (seconds)

        Returns:
            Audio with silence removed
        """
        if silence_threshold_db is None:
            silence_threshold_db = self.silence_threshold

        try:
            # Convert threshold to linear scale
            threshold_linear = 10 ** (silence_threshold_db / 20)

            # Calculate window size for silence detection
            window_size = int(0.02 * self.target_sample_rate)  # 20ms windows
            hop_size = window_size // 2

            # Detect speech regions
            speech_regions = []
            current_speech_start = None

            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))

                if rms > threshold_linear:  # Speech detected
                    if current_speech_start is None:
                        current_speech_start = max(0, i - int(padding_duration * self.target_sample_rate))
                else:  # Silence detected
                    if current_speech_start is not None:
                        speech_end = min(len(audio_data), i + int(padding_duration * self.target_sample_rate))
                        speech_regions.append((current_speech_start, speech_end))
                        current_speech_start = None

            # Handle case where speech continues to end of audio
            if current_speech_start is not None:
                speech_regions.append((current_speech_start, len(audio_data)))

            # Combine overlapping regions
            if speech_regions:
                combined_regions = [speech_regions[0]]
                for start, end in speech_regions[1:]:
                    if start <= combined_regions[-1][1]:
                        # Overlapping, merge regions
                        combined_regions[-1] = (combined_regions[-1][0], max(combined_regions[-1][1], end))
                    else:
                        combined_regions.append((start, end))

                # Extract speech regions
                speech_audio = []
                for start, end in combined_regions:
                    speech_audio.append(audio_data[start:end])

                if speech_audio:
                    result = np.concatenate(speech_audio)
                    self.logger.debug(f"Silence removal: {len(audio_data)} → {len(result)} samples")
                    return result.astype(np.float32)

            return audio_data

        except Exception as e:
            self.logger.error(f"Silence removal failed: {e}")
            return audio_data

    def pcm_to_ulaw(self, pcm_data: np.ndarray) -> np.ndarray:
        """
        Convert PCM audio to μ-law encoding

        Args:
            pcm_data: PCM audio data (float32, -1 to 1)

        Returns:
            μ-law encoded data (uint8)
        """
        try:
            # Convert float to 16-bit PCM
            pcm_16bit = np.clip(pcm_data * 32767, -32768, 32767).astype(np.int16)

            # μ-law encoding
            ulaw_data = np.zeros(len(pcm_16bit), dtype=np.uint8)

            for i, sample in enumerate(pcm_16bit):
                # Get sign and magnitude
                sign = 0x80 if sample < 0 else 0x00
                magnitude = abs(sample)

                # Add bias and find exponent
                magnitude += self.ulaw_bias
                if magnitude > 32767:
                    magnitude = 32767

                exponent = 7
                for exp in range(8):
                    if magnitude <= (33 << exp):
                        exponent = exp
                        break

                # Calculate mantissa
                mantissa = (magnitude >> (exponent + 3)) & 0x0F

                # Combine sign, exponent, and mantissa
                ulaw_data[i] = sign | (exponent << 4) | mantissa

            return ulaw_data

        except Exception as e:
            self.logger.error(f"PCM to μ-law conversion failed: {e}")
            raise

    def ulaw_to_pcm(self, ulaw_data: np.ndarray) -> np.ndarray:
        """
        Convert μ-law encoded audio to PCM

        Args:
            ulaw_data: μ-law encoded data (uint8)

        Returns:
            PCM audio data (float32, -1 to 1)
        """
        try:
            pcm_data = np.zeros(len(ulaw_data), dtype=np.int16)

            for i, ulaw_byte in enumerate(ulaw_data):
                # Extract components
                sign = (ulaw_byte & 0x80) != 0
                exponent = (ulaw_byte >> 4) & 0x07
                mantissa = ulaw_byte & 0x0F

                # Reconstruct magnitude
                magnitude = (33 << exponent) + (mantissa << (exponent + 3)) - self.ulaw_bias

                # Apply sign
                pcm_data[i] = -magnitude if sign else magnitude

            # Convert to float32 (-1 to 1)
            return (pcm_data / 32767.0).astype(np.float32)

        except Exception as e:
            self.logger.error(f"μ-law to PCM conversion failed: {e}")
            raise

    def pcm_to_alaw(self, pcm_data: np.ndarray) -> np.ndarray:
        """
        Convert PCM audio to A-law encoding

        Args:
            pcm_data: PCM audio data (float32, -1 to 1)

        Returns:
            A-law encoded data (uint8)
        """
        try:
            # Convert float to 16-bit PCM
            pcm_16bit = np.clip(pcm_data * 32767, -32768, 32767).astype(np.int16)

            # A-law encoding table (simplified implementation)
            alaw_data = np.zeros(len(pcm_16bit), dtype=np.uint8)

            for i, sample in enumerate(pcm_16bit):
                # Get sign and magnitude
                sign = 0x80 if sample >= 0 else 0x00
                magnitude = abs(sample)

                if magnitude >= 256:
                    exponent = 7
                    for exp in range(8):
                        if magnitude <= (256 << exp):
                            exponent = exp
                            break
                    mantissa = (magnitude >> (exponent + 4)) & 0x0F
                    alaw_byte = sign | (exponent << 4) | mantissa
                else:
                    alaw_byte = sign | (magnitude >> 4)

                # XOR with bias
                alaw_data[i] = alaw_byte ^ self.alaw_bias

            return alaw_data

        except Exception as e:
            self.logger.error(f"PCM to A-law conversion failed: {e}")
            raise

    def analyze_audio_quality(self, audio_data: np.ndarray) -> AudioMetrics:
        """
        Analyze audio quality metrics

        Args:
            audio_data: Audio data to analyze

        Returns:
            AudioMetrics object with quality measurements
        """
        try:
            # Basic level measurements
            rms_level = np.sqrt(np.mean(audio_data ** 2))
            peak_level = np.max(np.abs(audio_data))

            # Convert to dB
            rms_db = 20 * np.log10(rms_level) if rms_level > 0 else -np.inf
            peak_db = 20 * np.log10(peak_level) if peak_level > 0 else -np.inf

            # Estimate SNR using spectral analysis
            fft_audio = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_audio)

            # Estimate noise floor (bottom 10% of spectrum)
            sorted_mag = np.sort(magnitude)
            noise_floor = np.mean(sorted_mag[:len(sorted_mag)//10])
            signal_power = np.mean(sorted_mag[-len(sorted_mag)//10:])

            snr_estimate = 20 * np.log10(signal_power / noise_floor) if noise_floor > 0 else np.inf

            # Duration
            duration = len(audio_data) / self.target_sample_rate

            # Dynamic range
            dynamic_range = peak_db - rms_db

            return AudioMetrics(
                rms_level=rms_db,
                peak_level=peak_db,
                snr_estimate=snr_estimate,
                duration=duration,
                sample_rate=self.target_sample_rate,
                dynamic_range=dynamic_range
            )

        except Exception as e:
            self.logger.error(f"Audio quality analysis failed: {e}")
            return AudioMetrics(0, 0, 0, 0, self.target_sample_rate, 0)

    def process_for_telephony(self,
                            audio_data: np.ndarray,
                            input_sample_rate: int,
                            apply_noise_reduction: bool = True,
                            apply_compression: bool = True,
                            normalize_level: bool = True) -> np.ndarray:
        """
        Complete telephony audio processing pipeline

        Args:
            audio_data: Input audio data
            input_sample_rate: Input sample rate
            apply_noise_reduction: Whether to apply noise reduction
            apply_compression: Whether to apply dynamic range compression
            normalize_level: Whether to normalize audio level

        Returns:
            Processed audio optimized for telephony
        """
        try:
            start_time = time.time()

            # Step 1: Resample to telephony rate
            if input_sample_rate != self.target_sample_rate:
                audio_data = self.resample_audio(audio_data, input_sample_rate)

            # Step 2: Apply telephony bandpass filter
            audio_data = self.apply_telephony_filter(audio_data)

            # Step 3: Noise reduction (optional)
            if apply_noise_reduction:
                audio_data = self.noise_reduction(audio_data)

            # Step 4: Pre-emphasis for speech clarity
            audio_data = self.apply_preemphasis(audio_data)

            # Step 5: Dynamic range compression (optional)
            if apply_compression:
                audio_data = self.apply_dynamic_range_compression(audio_data)

            # Step 6: Normalize audio level (optional)
            if normalize_level:
                audio_data = self.normalize_audio_level(audio_data)

            # Step 7: Final limiting to prevent clipping
            audio_data = np.clip(audio_data, -0.95, 0.95)

            processing_time = time.time() - start_time
            self.logger.debug(f"Telephony processing completed in {processing_time:.3f}s")

            return audio_data.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Telephony processing failed: {e}")
            raise

    def save_for_asterisk(self,
                         audio_data: np.ndarray,
                         output_path: str,
                         format_type: str = "wav") -> str:
        """
        Save audio in format suitable for Asterisk playback

        Args:
            audio_data: Processed audio data
            output_path: Output file path
            format_type: Audio format (wav, raw, ulaw, alaw)

        Returns:
            Path to saved file
        """
        try:
            if format_type.lower() == "wav":
                # Save as 8kHz 16-bit WAV
                sf.write(
                    output_path,
                    audio_data,
                    self.target_sample_rate,
                    subtype='PCM_16'
                )

            elif format_type.lower() == "ulaw":
                # Convert to μ-law and save
                ulaw_data = self.pcm_to_ulaw(audio_data)
                with open(output_path, 'wb') as f:
                    f.write(ulaw_data.tobytes())

            elif format_type.lower() == "alaw":
                # Convert to A-law and save
                alaw_data = self.pcm_to_alaw(audio_data)
                with open(output_path, 'wb') as f:
                    f.write(alaw_data.tobytes())

            elif format_type.lower() == "raw":
                # Save as raw 16-bit PCM
                pcm_16bit = (audio_data * 32767).astype(np.int16)
                with open(output_path, 'wb') as f:
                    f.write(pcm_16bit.tobytes())

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            self.logger.debug(f"Audio saved for Asterisk: {output_path} ({format_type})")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save audio for Asterisk: {e}")
            raise


if __name__ == "__main__":
    # Test the telephony audio processor
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Telephony Audio Processor ===")

    # Create test audio (1 second sine wave at 1kHz)
    sample_rate = 22050
    duration = 1.0
    frequency = 1000

    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Add some noise
    noise = 0.05 * np.random.randn(len(test_audio)).astype(np.float32)
    test_audio += noise

    print(f"Test audio: {len(test_audio)} samples at {sample_rate} Hz")

    # Initialize processor
    processor = TelephonyAudioProcessor()

    # Analyze original audio
    original_metrics = processor.analyze_audio_quality(test_audio)
    print(f"Original audio metrics:")
    print(f"  RMS: {original_metrics.rms_level:.1f} dB")
    print(f"  Peak: {original_metrics.peak_level:.1f} dB")
    print(f"  SNR: {original_metrics.snr_estimate:.1f} dB")

    # Process for telephony
    start_time = time.time()
    processed_audio = processor.process_for_telephony(test_audio, sample_rate)
    processing_time = time.time() - start_time

    print(f"Processing time: {processing_time:.3f}s")

    # Analyze processed audio
    processed_metrics = processor.analyze_audio_quality(processed_audio)
    print(f"Processed audio metrics:")
    print(f"  RMS: {processed_metrics.rms_level:.1f} dB")
    print(f"  Peak: {processed_metrics.peak_level:.1f} dB")
    print(f"  SNR: {processed_metrics.snr_estimate:.1f} dB")
    print(f"  Sample rate: {processed_metrics.sample_rate} Hz")

    # Test codec conversions
    print("\nTesting codec conversions...")

    # Test μ-law conversion
    ulaw_data = processor.pcm_to_ulaw(processed_audio[:1000])
    recovered_ulaw = processor.ulaw_to_pcm(ulaw_data)
    ulaw_error = np.mean(np.abs(processed_audio[:1000] - recovered_ulaw))
    print(f"μ-law roundtrip error: {ulaw_error:.6f}")

    # Test A-law conversion
    alaw_data = processor.pcm_to_alaw(processed_audio[:1000])
    print(f"A-law encoding successful: {len(alaw_data)} bytes")

    # Save test files
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "test_telephony.wav"
        processor.save_for_asterisk(processed_audio, str(wav_path), "wav")
        print(f"Test WAV saved: {wav_path}")

    print("=== Audio Processor Test Complete ===")
