#!/usr/bin/env python3
import os, sys, time, json, re, queue, threading, subprocess, tempfile, signal
from pathlib import Path

# third-party
import numpy as np
import requests
from dotenv import load_dotenv
import webrtcvad
import alsaaudio                 # sudo apt-get install python3-alsaaudio
from TTS.api import TTS          # pip install TTS==0.15.6
import whisper                   # pip install openai-whisper

load_dotenv()

# ---------- Config ----------
THREECX_SERVER   = os.getenv("THREECX_SERVER", "mtipbx.ny.3cx.us")
THREECX_PORT     = int(os.getenv("THREECX_PORT", "5060"))
THREECX_EXT      = os.getenv("THREECX_EXTENSION", "1600")
THREECX_AUTH     = os.getenv("THREECX_AUTH_ID", "qpZh2VS624")
THREECX_PASS     = os.getenv("THREECX_PASSWORD", "")

OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "orca2:7b")
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL", "base")
TTS_MODEL_ID     = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

SAMPLE_RATE      = int(os.getenv("SAMPLE_RATE", "8000"))          # 8 kHz for G.711
CHUNK_SAMPLES    = 160                                            # 20ms @ 8kHz (required for VAD)
VAD_MODE         = int(os.getenv("VAD_AGGRESSIVENESS", "2"))

LOOP_CAPTURE     = os.getenv("LOOP_CAPTURE", "hw:Loopback,0,0")   # where we read caller audio
LOOP_PLAYBACK    = os.getenv("LOOP_PLAYBACK", "hw:Loopback,1,0")  # where we write TTS audio

BARESIP_BIN      = os.getenv("BARESIP_BIN", "/usr/bin/baresip")
REG_DELAY        = int(os.getenv("BARESIP_REG_DELAY_SEC", "6"))

if not THREECX_PASS:
    print("ERROR: THREECX_PASSWORD is empty in .env")
    sys.exit(1)

# ---------- Helpers ----------
def resample_22050_to_8000(wav_22050: np.ndarray) -> np.ndarray:
    """Simple linear resample to 8k for G.711 path."""
    src = 22050
    dst = SAMPLE_RATE
    x = np.arange(len(wav_22050))
    xp = np.linspace(0, len(wav_22050)-1, int(len(wav_22050)*dst/src))
    out = np.interp(xp, x, wav_22050).astype(np.float32)
    return out

def float_to_int16(x: np.ndarray) -> bytes:
    y = np.clip(x, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    return y.tobytes()

def int16_to_float(buf: bytes) -> np.ndarray:
    return (np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32767.0)

# ---------- Baresip manager ----------
class BaresipProc:
    def __init__(self):
        self.proc = None
        self.workdir = Path(tempfile.mkdtemp(prefix="vibe-baresip-"))
        self.accounts = self.workdir / "accounts"
        self.config   = self.workdir / "config"
        self.stdout_lines = queue.Queue()

    def _write_files(self):
        # accounts (SIP URI with auth)
        acct = (
            f"<sip:{THREECX_EXT}@{THREECX_SERVER};transport=udp>;auth_user={THREECX_AUTH};auth_pass={THREECX_PASS};"
            f"outbound=;regint=300;pubint=0;answermode=auto"
        )
        self.accounts.write_text(acct + "\n")

        # minimal config to pin ALSA devices
        cfg = [
            "audio_player    alsa,{}".format(LOOP_PLAYBACK),
            "audio_source    alsa,{}".format(LOOP_CAPTURE),
            "ausrc_srate     8000",
            "auplay_srate    8000",
            "audio_channels  1",
            "module_path     /usr/lib/baresip/modules",
            "module          stdio.so",
        ]
        self.config.write_text("\n".join(cfg) + "\n")

    def start(self):
        self._write_files()
        env = os.environ.copy()
        cmd = [BARESIP_BIN, "-f", str(self.workdir)]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._pump_stdout, daemon=True).start()

    def _pump_stdout(self):
        for line in self.proc.stdout:
            self.stdout_lines.put(line.rstrip())

    def wait_registered(self, timeout=30) -> bool:
        start = time.time()
        pattern_ok = re.compile(r"registration success|200 OK", re.IGNORECASE)
        while time.time() - start < timeout:
            try:
                line = self.stdout_lines.get(timeout=0.5)
            except queue.Empty:
                continue
            # echo some logs
            if "REGISTER" in line or "notify" in line.lower():
                print(line)
            if pattern_ok.search(line):
                return True
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

# ---------- Audio IO ----------
class LoopbackIO:
    def __init__(self):
        # capture caller audio (non-blocking)
        self.cap = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NONBLOCK,
            device=LOOP_CAPTURE
        )
        self.cap.setchannels(1)
        self.cap.setrate(SAMPLE_RATE)
        self.cap.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.cap.setperiodsize(CHUNK_SAMPLES)

        # playback TTS to caller
        self.pb  = alsaaudio.PCM(
            type=alsaaudio.PCM_PLAYBACK,
            mode=alsaaudio.PCM_NORMAL,
            device=LOOP_PLAYBACK
        )
        self.pb.setchannels(1)
        self.pb.setrate(SAMPLE_RATE)
        self.pb.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.pb.setperiodsize(CHUNK_SAMPLES)

    def read_chunk(self) -> bytes | None:
        # returns 320 bytes (160 samples * 2) or None if no data yet
        length, data = self.cap.read()
        if length is None or length <= 0:
            return None
        # some drivers may give bigger bursts; trim to CHUNK_SAMPLES multiple
        n = (len(data) // 2)  # samples
        if n < CHUNK_SAMPLES:
            return None
        frames = (n // CHUNK_SAMPLES) * CHUNK_SAMPLES
        return data[:frames*2]

    def play_bytes(self, b: bytes):
        self.pb.write(b)

# ---------- AI Pipeline ----------
class AIVoice:
    def __init__(self):
        print("[INFO] Loading Whisper model:", WHISPER_MODEL_ID)
        self.whisper = whisper.load_model(WHISPER_MODEL_ID)

        print("[INFO] Loading TTS model:", TTS_MODEL_ID)
        # Coqui will auto-download; run on GPU if available
        self.tts = TTS(model_name=TTS_MODEL_ID, progress_bar=False)

        self.vad = webrtcvad.Vad(VAD_MODE)

    def greet(self, io: LoopbackIO, text="Hello! After the beep, please ask your question."):
        wav = self.tts.tts(text=text)                     # 22050 float32
        wav8 = resample_22050_to_8000(np.array(wav))
        buf = float_to_int16(wav8)
        # chunked playback so caller hears immediately
        step = CHUNK_SAMPLES*2
        for i in range(0, len(buf), step):
            io.play_bytes(buf[i:i+step])
            time.sleep(0.02)

    def transcribe(self, float_pcm: np.ndarray) -> str:
        # Whisper expects 16k; quick linear resample
        def resample_8k_to_16k(x):
            src, dst = 8000, 16000
            xp = np.linspace(0, len(x)-1, int(len(x)*dst/src))
            return np.interp(xp, np.arange(len(x)), x).astype(np.float32)

        audio16 = resample_8k_to_16k(float_pcm)
        result = self.whisper.transcribe(audio16, language="en")
        return result.get("text", "").strip()

    def llm(self, prompt: str) -> str:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.6, "num_predict": 120}}
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=25)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            print("[LLM] error:", e)
            return "Sorry, I could not generate a reply."

    def speak(self, io: LoopbackIO, text: str):
        wav = self.tts.tts(text=text)
        wav8 = resample_22050_to_8000(np.array(wav))
        buf = float_to_int16(wav8)
        step = CHUNK_SAMPLES*2
        for i in range(0, len(buf), step):
            io.play_bytes(buf[i:i+step])
            time.sleep(0.02)

# ---------- Main runner ----------
def main():
    print("[INFO] Starting baresip…")
    bs = BaresipProc()
    bs.start()
    if not bs.wait_registered(timeout=30 + REG_DELAY):
        bs.stop()
        print("ERROR: 3CX registration did not succeed. Check creds/firewall.")
        sys.exit(2)
    print("[INFO] Registered: OK")

    io = LoopbackIO()
    ai = AIVoice()

    # Non-blocking read loop + VAD buffering
    ai.greet(io)

    vad = ai.vad
    speech_buf = bytearray()
    silence_ms = 0
    last_print = time.time()

    print("[INFO] Voice loop ready. Speak from the remote side (call in). Ctrl+C to exit.")

    try:
        while True:
            data = io.read_chunk()
            if data is None:
                # nothing yet; keep the UI alive
                if time.time() - last_print > 1.5:
                    print("(waiting for RTP…)")
                    last_print = time.time()
                time.sleep(0.005)
                continue

            # Process every 20ms frame
            # webrtcvad requires exactly 10/20/30ms @ 8k in 16-bit mono
            for off in range(0, len(data), CHUNK_SAMPLES*2):
                frame = data[off:off+CHUNK_SAMPLES*2]
                if len(frame) < CHUNK_SAMPLES*2:
                    break
                is_speech = vad.is_speech(frame, SAMPLE_RATE)

                if is_speech:
                    speech_buf.extend(frame)
                    silence_ms = 0
                else:
                    if speech_buf:
                        silence_ms += 20
                        # end-of-utterance when 400ms of silence
                        if silence_ms >= 400:
                            # decode, STT, LLM, TTS
                            float_pcm = int16_to_float(bytes(speech_buf))
                            speech_buf.clear()
                            silence_ms = 0

                            print("[STT] transcribing…")
                            text = ai.transcribe(float_pcm)
                            print("User:", text)
                            if not text:
                                continue

                            # Simple commands
                            low = text.lower()
                            if "goodbye" in low or "hang up" in low:
                                ai.speak(io, "Goodbye!")
                                raise KeyboardInterrupt

                            prompt = f"You are a helpful assistant. Keep answers 1-3 sentences.\nUser: {text}\nAssistant:"
                            reply = ai.llm(prompt)
                            print("Bot :", reply)
                            ai.speak(io, reply)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        bs.stop()

if __name__ == "__main__":
    main()
