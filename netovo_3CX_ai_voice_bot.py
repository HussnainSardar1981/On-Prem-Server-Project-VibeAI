#!/usr/bin/env python3
import os
import time
import json
import logging
from collections import deque

import numpy as np
import requests
from dotenv import load_dotenv

import alsaaudio            # pyalsaaudio
import webrtcvad
import whisper

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

SR = int(os.getenv("SAMPLE_RATE", "8000"))           # 8000
FRAME_MS = 20                                         # fixed 20ms frames for VAD
FRAMES_PER_PERIOD = SR * FRAME_MS // 1000            # 160 samples
BYTES_PER_FRAME = FRAMES_PER_PERIOD * 2              # S16_LE mono
VAD_AGGR = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0..3

ALSA_IN_DEV  = os.getenv("ALSA_IN_DEV",  "hw:Loopback,1,0")
ALSA_OUT_DEV = os.getenv("ALSA_OUT_DEV", "hw:Loopback,0,0")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "orca2:7b")
TTS_MODEL     = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "30"))   # sec
TARGET_LATENCY   = int(os.getenv("TARGET_LATENCY", "300"))     # ms
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE         = os.getenv("LOG_FILE", "voice_bot.log")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=LOG_FILE,
)
console = logging.StreamHandler()
console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)
log = logging.getLogger("bridge")

# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────────────────────
def resample_audio(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Fast, decent resampling via polyphase."""
    if src_sr == dst_sr:
        return x
    # Lazily import to avoid heavy import on startup if not needed
    from scipy.signal import resample_poly
    # Normalize to float32 [-1,1] if int16
    if x.dtype != np.float32:
        x = (x.astype(np.float32) / 32768.0)
    gcd = np.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    y = resample_poly(x, up, down).astype(np.float32)
    return y

def int16_bytes_from_float32(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
log.info("Loading Whisper model: %s", WHISPER_MODEL)
wh_model = whisper.load_model(WHISPER_MODEL)

log.info("Loading TTS model: %s (this may take a moment)", TTS_MODEL)
from TTS.api import TTS   # delay import for faster startup
tts_model = TTS(model_name=TTS_MODEL, progress_bar=False, gpu=os.getenv("FORCE_GPU","true").lower()=="true")

# ──────────────────────────────────────────────────────────────────────────────
# ALSA devices
# ──────────────────────────────────────────────────────────────────────────────
def open_capture(dev: str):
    cap = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=dev)
    cap.setchannels(1)
    cap.setrate(SR)
    cap.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    cap.setperiodsize(FRAMES_PER_PERIOD)  # 160 samples (20ms @ 8k)
    return cap

def open_playback(dev: str):
    pb = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=dev)
    pb.setchannels(1)
    pb.setrate(SR)
    pb.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pb.setperiodsize(FRAMES_PER_PERIOD)
    return pb

cap = open_capture(ALSA_IN_DEV)
pb  = open_playback(ALSA_OUT_DEV)
log.info("ALSA opened. IN=%s, OUT=%s, frame=%d bytes (20ms)", ALSA_IN_DEV, ALSA_OUT_DEV, BYTES_PER_FRAME)

# ──────────────────────────────────────────────────────────────────────────────
# VAD
# ──────────────────────────────────────────────────────────────────────────────
vad = webrtcvad.Vad(VAD_AGGR)

def is_speech_20ms(buf: bytes) -> bool:
    """webrtcvad expects exactly 10/20/30ms; we enforce 20ms (320 bytes @ 8k S16)."""
    if len(buf) != BYTES_PER_FRAME:
        return False
    try:
        return vad.is_speech(buf, SR)
    except Exception:
        # If ALSA gave us something odd, treat as silence to stay safe
        return False

# ──────────────────────────────────────────────────────────────────────────────
# STT → LLM → TTS
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_bytes(pcm_bytes: bytes) -> str:
    """Transcribe 8k S16 mono bytes with Whisper (resample to 16k)."""
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    audio_16k = resample_audio(samples, SR, 16000)
    # whisper expects numpy float32 array at 16000 Hz
    result = wh_model.transcribe(audio_16k, language="en")
    return (result.get("text") or "").strip()

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = RESPONSE_TIMEOUT) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.6, "num_predict": 120}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json().get("response", "").strip()
    return out

def synth_and_play(text: str):
    """Coqui TTS → 8k PCM → ALSA playback in 20ms chunks."""
    if not text:
        return
    log.info("TTS -> '%s'", text[:100])
    wav = tts_model.tts(text=text)  # float32 [-1,1], native sr (usually 22050)
    wav = np.array(wav, dtype=np.float32).flatten()
    audio_8k = resample_audio(wav, 22050, SR)
    pcm = int16_bytes_from_float32(audio_8k)

    # Play in exact 20ms chunks so ALSA stays happy
    for i in range(0, len(pcm), BYTES_PER_FRAME):
        chunk = pcm[i:i+BYTES_PER_FRAME]
        if len(chunk) < BYTES_PER_FRAME:
            break
        pb.write(chunk)

# ──────────────────────────────────────────────────────────────────────────────
# Conversation loop
# ──────────────────────────────────────────────────────────────────────────────
GREETING = "Hello! I am the test bot. After the beep, please ask your question."
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant for a company phone line. "
    "Answer briefly (2-4 sentences), be polite, and ask a follow-up if needed. "
    "If unsure, say what you can do next."
)

def main():
    # Intro tone + greeting (optional)
    log.info("Sending greeting...")
    synth_and_play(GREETING)
    time.sleep(0.2)

    speech_buf = bytearray()
    silence_frames = 0
    max_silence_frames = 10  # 10 * 20ms = ~200ms gap ends an utterance

    log.info("Ready. Listening on ALSA… Speak after the beep.")

    # (optional) short beep to indicate listening
    pb.write(int16_bytes_from_float32(np.sin(2*np.pi*440*np.arange(FRAMES_PER_PERIOD)/SR).astype(np.float32)*0.2))

    while True:
        length, data = cap.read()
        if length <= 0:
            continue

        # Slice ALSA block into exact 20ms frames
        for i in range(0, len(data), BYTES_PER_FRAME):
            frame = data[i:i+BYTES_PER_FRAME]
            if len(frame) < BYTES_PER_FRAME:
                break

            if is_speech_20ms(frame):
                # speaking
                speech_buf.extend(frame)
                silence_frames = 0
            else:
                # silence
                if speech_buf:
                    silence_frames += 1
                    if silence_frames > max_silence_frames:
                        # Process utterance
                        try:
                            utter_bytes = bytes(speech_buf)
                            speech_buf.clear()
                            silence_frames = 0

                            t0 = time.time()
                            text = transcribe_bytes(utter_bytes)
                            t_stt = time.time() - t0
                            log.info("STT (%.3fs): %s", t_stt, text)

                            if not text:
                                continue

                            # Simple DTMF-like voice intents
                            lt = text.lower()
                            if "goodbye" in lt or "hang up" in lt or "end call" in lt:
                                synth_and_play("Goodbye! Have a great day.")
                                return
                            if "repeat" in lt:
                                synth_and_play("Sure — please say that again and I'll help.")
                                continue

                            # LLM
                            t1 = time.time()
                            prompt = f"{SYSTEM_INSTRUCTIONS}\n\nUser: {text}\nAssistant:"
                            reply = call_ollama(prompt)
                            t_llm = time.time() - t1
                            log.info("LLM  (%.3fs): %s", t_llm, reply)

                            # TTS
                            t2 = time.time()
                            synth_and_play(reply)
                            t_tts = time.time() - t2

                            total = (time.time() - t0)
                            log.info("Turn latency: %.0f ms (STT %.0f / LLM %.0f / TTS %.0f)",
                                     total*1000, t_stt*1000, t_llm*1000, t_tts*1000)

                        except Exception as e:
                            log.exception("Error in turn: %s", e)

    # (never reached)
    # cap, pb are closed by GC when process ends

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Exiting on Ctrl-C")
