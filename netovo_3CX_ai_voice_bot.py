#!/usr/bin/env python3
import os, time, logging, requests
import numpy as np
from dotenv import load_dotenv
import webrtcvad
import alsaaudio
from scipy.signal import resample_poly
import whisper
from TTS.api import TTS

load_dotenv()
LOG = logging.getLogger("bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# env
SR           = int(os.getenv("SAMPLE_RATE", "8000"))      # 8 kHz for G.711
FRAME        = int(os.getenv("CHUNK_SIZE", "160"))        # 20ms @ 8k
IN_DEV       = os.getenv("ALSA_IN_DEV",  "hw:Loopback,1,0")
OUT_DEV      = os.getenv("ALSA_OUT_DEV", "hw:Loopback,0,0")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "orca2:7b")
WHISPER_MODEL= os.getenv("WHISPER_MODEL", "base")
TTS_MODEL    = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

def open_cap(dev, rate, period):
    cap = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, device=dev)
    cap.setchannels(1); cap.setrate(rate); cap.setformat(alsaaudio.PCM_FORMAT_S16_LE); cap.setperiodsize(period)
    return cap

def open_play(dev, rate, period):
    pb = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, device=dev)
    pb.setchannels(1); pb.setrate(rate); pb.setformat(alsaaudio.PCM_FORMAT_S16_LE); pb.setperiodsize(period)
    return pb

def resample_i16(x_i16, src, dst):
    if src == dst: return x_i16.astype(np.float32)/32768.0
    x = x_i16.astype(np.float32)/32768.0
    g = np.gcd(src, dst)
    y = resample_poly(x, dst//g, src//g)
    return np.clip(y, -1.0, 1.0)

def write_float(pb, mono_f32):
    pb.write((np.clip(mono_f32, -1, 1)*32767).astype(np.int16).tobytes())

def beep(pb, freq=1000, ms=200):
    n = int(SR*ms/1000); t = np.arange(n)/SR
    write_float(pb, 0.3*np.sin(2*np.pi*freq*t))

def llm(prompt: str) -> str:
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature":0.4, "num_predict":120}
    }, timeout=30)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

if __name__ == "__main__":
    LOG.info("Opening ALSA cap=%s play=%s", IN_DEV, OUT_DEV)
    cap = open_cap(IN_DEV, SR, FRAME)
    pb  = open_play(OUT_DEV, SR, FRAME)
    vad = webrtcvad.Vad(2)

    LOG.info("Loading models: whisper=%s  tts=%s", WHISPER_MODEL, TTS_MODEL)
    stt = whisper.load_model(WHISPER_MODEL)
    tts = TTS(model_name=TTS_MODEL, gpu=True, progress_bar=False)

    LOG.info("Ready. Answer the call in Baresip and talk after the beep.")
    while True:
        # prompt caller
        text = "Hello! I am the test bot. After the beep, please ask your question."
        LOG.info("TTS greeting")
        wav = np.asarray(tts.tts(text=text), dtype=np.float32)  # ~22.05k
        write_float(pb, resample_i16((wav*32767).astype(np.int16), 22050, SR))
        time.sleep(0.1); beep(pb)

        # record one turn with simple VAD
        chunks=[]; heard=False; silence=0
        while True:
            n, data = cap.read()
            if n == 0: continue
            if vad.is_speech(data, SR):
                chunks.append(np.frombuffer(data, dtype=np.int16)); heard=True; silence=0
            else:
                if heard: silence += 1
                if heard and silence >= 25:  # ~0.5s of trailing silence
                    break

        if not chunks:
            wav = np.asarray(tts.tts(text="I did not hear anything. Let's try again."), dtype=np.float32)
            write_float(pb, resample_i16((wav*32767).astype(np.int16), 22050, SR))
            continue

        pcm8 = np.concatenate(chunks)
        up16 = resample_i16(pcm8, SR, 16000)
        LOG.info("Transcribing...")
        text = (stt.transcribe(up16, language="en", fp16=False).get("text") or "").strip()
        LOG.info("STT: %s", text)

        prompt = ("You are a concise, helpful assistant. Answer in one or two sentences.\n"
                  f"User: {text}\nAssistant:")
        try:
            answer = llm(prompt)
        except Exception as e:
            LOG.error("LLM error: %s", e); answer = "I'm having trouble reaching the language model right now."

        LOG.info("TTS answer")
        wav = np.asarray(tts.tts(text=answer), dtype=np.float32)
        write_float(pb, resample_i16((wav*32767).astype(np.int16), 22050, SR))
        time.sleep(0.2)
