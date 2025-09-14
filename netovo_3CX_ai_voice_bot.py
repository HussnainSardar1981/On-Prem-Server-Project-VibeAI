#!/usr/bin/env python3
"""
Stable E2E smoke test using ALSA directly (no PortAudio):
- Registers to 3CX (pjsua2) with Null audio while idle
- On media active, pins PJSIP to ALSA Loopback device
- Captures caller audio from hw:Loopback,1,0
- Plays TTS back into call via hw:Loopback,1,0
- Single-turn conversation (greet -> STT -> LLM -> TTS -> hangup)

Env: uses your .env exactly (THREECX_*, OLLAMA_*, WHISPER_MODEL, TTS_MODEL, SAMPLE_RATE, CHUNK_SIZE).
Optional overrides:
  ALSA_IN_DEV=hw:Loopback,1,0
  ALSA_OUT_DEV=hw:Loopback,1,0
"""

import os, time, signal, logging, threading
from dotenv import load_dotenv
import numpy as np
from scipy.signal import resample_poly
import requests
import webrtcvad
import pjsua2 as pj

# ALSA direct bindings (no PortAudio)
import alsaaudio  # pip install pyalsaaudio

# --------- ENV ---------
load_dotenv()
SIP_DOMAIN   = os.getenv("THREECX_SERVER")
SIP_PORT     = int(os.getenv("THREECX_PORT", "5060"))
SIP_EXT      = os.getenv("THREECX_EXTENSION")
SIP_AUTH_ID  = os.getenv("THREECX_AUTH_ID")
SIP_PASSWORD = os.getenv("THREECX_PASSWORD")

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "orca2:7b")
WHISPER_MODEL= os.getenv("WHISPER_MODEL", "base")
TTS_MODEL    = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")

SAMPLE_RATE  = int(os.getenv("SAMPLE_RATE", "8000"))   # 8k for SIP
FRAME        = int(os.getenv("CHUNK_SIZE", "160"))     # 20ms @ 8k
ALSA_IN_DEV  = os.getenv("ALSA_IN_DEV",  "hw:Loopback,1,0")
ALSA_OUT_DEV = os.getenv("ALSA_OUT_DEV", "hw:Loopback,1,0")

# Keep native libs calm on servers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke-alsa")

# --------- ALSA helpers ---------
def open_alsa_capture(device: str, rate: int, period: int):
    cap = alsaaudio.PCM(type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL, device=device)
    cap.setchannels(1)
    cap.setrate(rate)
    cap.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    cap.setperiodsize(period)  # frames per read
    return cap

def open_alsa_playback(device: str, rate: int, period: int):
    pb = alsaaudio.PCM(type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL, device=device)
    pb.setchannels(1)
    pb.setrate(rate)
    pb.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pb.setperiodsize(period)
    return pb

def resample_i16(x_i16: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample int16 PCM using polyphase; returns int16."""
    if src == dst:
        return x_i16.astype(np.int16, copy=False)
    x = x_i16.astype(np.float32) / 32768.0
    g = np.gcd(src, dst)
    y = resample_poly(x, dst // g, src // g)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)

# --------- PJSUA2 wrappers ---------
class SmokeCall(pj.Call):
    def __init__(self, acc, app): super().__init__(acc); self.app = app
    def onCallState(self, prm):
        ci = self.getInfo()
        log.info("CALL %s (%s)", ci.stateText, ci.lastStatusCode)
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self.app.end_session()
    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for m in ci.media:
            if m.type == pj.PJMEDIA_TYPE_AUDIO and m.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                am  = pj.AudioMedia.typecastFromMedia(self.getMedia(m.index))
                adm = pj.Endpoint.instance().audDevManager()
                am.startTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().startTransmit(am)
                log.info("PJSIP media bridged to ALSA Loopback")
                self.app.start_session(self)

class SmokeAccount(pj.Account):
    def __init__(self, app): super().__init__(); self.app=app
    def onRegState(self, prm):
        info = self.getInfo()
        log.info("REGISTER active=%s code=%s txt=%s", info.regIsActive, prm.code, info.regStatusText)
    def onIncomingCall(self, prm):
        log.info("Incoming call")
        call = SmokeCall(self, self.app)
        op = pj.CallOpParam(); op.statusCode = 200
        call.answer(op)

# --------- App ---------
class App:
    def __init__(self):
        self.ep = pj.Endpoint()
        self.acc = None
        self.call = None
        self.cap = None     # ALSA capture
        self.pb  = None     # ALSA playback
        self.whisper = None
        self.tts     = None
        self.running = True

    def start(self):
        if not all([SIP_DOMAIN, SIP_EXT, SIP_AUTH_ID, SIP_PASSWORD]):
            raise RuntimeError("Missing THREECX_* env vars")

        # Start PJSIP with Null audio to avoid idle crashes
        self.ep.libCreate()
        lc = pj.LogConfig(); lc.level = 3; lc.msgLogging = 0
        ec = pj.EpConfig();  ec.logConfig = lc
        self.ep.libInit(ec)
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, pj.TransportConfig())
        self.ep.libStart()

        adm = self.ep.audDevManager()
        adm.setNullDev()  # keep off devices until media is active
        log.info("PJSIP started with Null audio device")

        # Register
        acfg = pj.AccountConfig()
        acfg.idUri = f"sip:{SIP_EXT}@{SIP_DOMAIN}"
        acfg.regConfig.registrarUri = f"sip:{SIP_DOMAIN}"
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "3CXPhoneSystem", SIP_AUTH_ID, 0, SIP_PASSWORD)
        )
        self.acc = SmokeAccount(self)
        self.acc.create(acfg)

        log.info("Ready. Call the DID/extension; it will auto-answer. Ctrl+C to quit.")

    def start_session(self, call: SmokeCall):
        self.call = call

        # Switch PJSIP to actual Loopback now that media is active
        adm = self.ep.audDevManager()
        cap_id, play_id = self._find_loopback_ids(adm)
        adm.setCaptureDev(cap_id); adm.setPlaybackDev(play_id)
        log.info("Pinned PJSIP devices cap=%s play=%s", cap_id, play_id)

        # Open ALSA endpoints we will use (mirror side)
        # Convention: PJSIP uses card 0; we sniff/send on card 1 side
        self.cap = open_alsa_capture(ALSA_IN_DEV,  SAMPLE_RATE, FRAME)
        self.pb  = open_alsa_playback(ALSA_OUT_DEV, SAMPLE_RATE, FRAME)
        log.info("Opened ALSA capture=%s playback=%s", ALSA_IN_DEV, ALSA_OUT_DEV)

        # Lazy-load models here to avoid heavy libs during registration
        import whisper as _wh
        log.info("Loading Whisper: %s", WHISPER_MODEL)
        self.whisper = _wh.load_model(WHISPER_MODEL)
        from TTS.api import TTS as _TTS
        log.info("Loading Coqui TTS: %s", TTS_MODEL)
        self.tts = _TTS(model_name=TTS_MODEL, gpu=True, progress_bar=False)

        threading.Thread(target=self._one_turn, daemon=True).start()

    def _find_loopback_ids(self, adm: pj.AudDevManager):
        cap = play = None
        for i in range(0, 64):
            try:
                di = adm.getDevInfo(i)
            except Exception:
                continue
            name = (di.name or "").lower()
            if "loopback" in name:
                if di.inputCount  > 0 and cap  is None: cap  = i
                if di.outputCount > 0 and play is None: play = i
        if cap is None or play is None:
            raise RuntimeError("PJSIP cannot find ALSA Loopback. Load snd-aloop.")
        return cap, play

    def _say(self, text: str):
        log.info("TTS: %s", text[:120])
        wav = np.asarray(self.tts.tts(text=text), dtype=np.float32)           # ~22.05kHz
        pcm8 = resample_i16((wav * 32767).astype(np.int16), 22050, SAMPLE_RATE)
        self.pb.write(pcm8.tobytes())

    def _beep(self, freq=1000, ms=220):
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.arange(n) / SAMPLE_RATE
        wave = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        self.pb.write(((wave * 32767).astype(np.int16)).tobytes())

    def _one_turn(self):
        # greet
        self._say("Hello! This is an AI test. After the beep, ask a short question.")
        time.sleep(0.25); self._beep()

        vad = webrtcvad.Vad(2)
        spoken = False; silence = 0; chunks = []

        # record a single utterance until 500 ms silence
        while self.call and self.call.isActive():
            nframes, data = self.cap.read()   # bytes for FRAME samples (S16LE)
            if nframes == 0: 
                continue
            frame = np.frombuffer(data, dtype=np.int16)
            if vad.is_speech(data, SAMPLE_RATE):
                spoken = True; silence = 0; chunks.append(frame)
            else:
                if spoken: silence += 1
                if spoken and silence >= 25:   # 25 * 20ms ≈ 0.5s
                    break

        if not chunks:
            self._say("I did not hear anything. Goodbye."); self.hangup(); return

        audio_8k  = np.concatenate(chunks)                            # int16
        audio_16k = resample_i16(audio_8k, SAMPLE_RATE, 16000)       # int16
        af        = audio_16k.astype(np.float32) / 32768.0

        t0 = time.time()
        res = self.whisper.transcribe(af, language="en", fp16=False)
        user = (res.get("text") or "").strip()
        log.info("STT %.2fs: %s", time.time()-t0, user)

        if not user:
            self._say("Sorry, I didn’t catch that. Goodbye."); self.hangup(); return

        # LLM
        prompt = ("You are a concise helpful assistant. Answer in 1–2 sentences.\n"
                  f"User: {user}\nAssistant:")
        try:
            t1 = time.time()
            r = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.4, "num_predict": 120}
            }, timeout=30)
            r.raise_for_status()
            reply = (r.json().get("response") or "").strip()
            log.info("LLM %.2fs", time.time()-t1)
        except Exception as e:
            log.error("Ollama error: %s", e)
            reply = "I’m having trouble reaching the model right now."

        self._say(reply)
        self.hangup()

    def hangup(self):
        try:
            if self.call and self.call.isActive():
                self.call.hangup(pj.HangupParam())
        except Exception:
            pass

    def end_session(self):
        try:
            if self.cap: self.cap.close()
            if self.pb:  self.pb.close()
        finally:
            self.cap = None; self.pb = None; self.call = None

    def stop(self):
        self.end_session()
        try: self.ep.hangupAllCalls()
        except Exception: pass
        self.ep.libDestroy()

# --------- main ---------
if __name__ == "__main__":
    app = App()
    app.start()

    def _stop(*_):
        app.stop(); raise SystemExit

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        time.sleep(1)
