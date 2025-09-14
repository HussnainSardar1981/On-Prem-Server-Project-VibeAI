#!/usr/bin/env python3
# Minimal, stable E2E: 3CX (pjsua2) + Whisper + Ollama + Coqui TTS
# Key stability tricks:
#   - Start pjsua2 with Null audio device (adm.setNullDev()) to avoid ALSA/PA races.
#   - Lazy-load Whisper/TTS only after media becomes active.

import os, time, signal, logging, threading, json
from dotenv import load_dotenv
import numpy as np
import requests
import pyaudio
from scipy.signal import resample_poly

import pjsua2 as pj

# ---- env ----
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

SAMPLE_RATE  = int(os.getenv("SAMPLE_RATE", "8000"))
FRAME        = int(os.getenv("CHUNK_SIZE", "160"))          # 20 ms @ 8 kHz
PA_IN_INDEX  = int(os.getenv("LOOPBACK_IN_INDEX",  "-1"))   # -1=auto
PA_OUT_INDEX = int(os.getenv("LOOPBACK_OUT_INDEX", "-1"))   # -1=auto

# tame thread storms from BLAS/torch if present
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke-stable")

# ---- audio helpers (PyAudio to ALSA Loopback) ----
def resample_i16(x_i16: np.ndarray, src: int, dst: int) -> np.ndarray:
    x = x_i16.astype(np.float32) / 32768.0
    g = np.gcd(src, dst); up = dst // g; down = src // g
    y = resample_poly(x, up, down)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)

class LoopbackIO:
    def __init__(self, rate=8000, frames=160, in_index=-1, out_index=-1):
        self.pa = pyaudio.PyAudio()
        if in_index < 0 or out_index < 0:
            in_index, out_index = self._auto_pick()
        self.in_index, self.out_index = in_index, out_index
        self.in_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                                      input=True, input_device_index=in_index,
                                      frames_per_buffer=frames, start=False)
        self.out_stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=rate,
                                       output=True, output_device_index=out_index,
                                       frames_per_buffer=frames, start=False)
        self.rate, self.frames = rate, frames

    def _auto_pick(self):
        inp = out = None
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if "loopback" in name:
                if info.get("maxInputChannels", 0) > 0 and inp is None: inp = i
                if info.get("maxOutputChannels",0) > 0 and out is None: out = i
        if inp is None or out is None:
            raise RuntimeError("No ALSA Loopback. Run: sudo modprobe snd-aloop")
        return inp, out

    def start(self):
        if not self.in_stream.is_active():  self.in_stream.start_stream()
        if not self.out_stream.is_active(): self.out_stream.start_stream()

    def stop_close(self):
        try:
            if self.in_stream.is_active():  self.in_stream.stop_stream()
            if self.out_stream.is_active(): self.out_stream.stop_stream()
        finally:
            self.in_stream.close(); self.out_stream.close(); self.pa.terminate()

    def read(self) -> np.ndarray:
        data = self.in_stream.read(self.frames, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def play(self, pcm_i16: np.ndarray):
        self.out_stream.write(pcm_i16.astype(np.int16).tobytes())

# ---- PJSUA2 call/account ----
class SmokeCall(pj.Call):
    def __init__(self, acc, app): super().__init__(acc); self.app = app

    def onCallState(self, prm):
        info = self.getInfo()
        log.info("CALL %s (%s)", info.stateText, info.lastStatusCode)
        if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self.app.end_session()

    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for m in ci.media:
            if m.type == pj.PJMEDIA_TYPE_AUDIO and m.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                am  = pj.AudioMedia.typecastFromMedia(self.getMedia(m.index))
                adm = pj.Endpoint.instance().audDevManager()
                am.startTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().startTransmit(am)
                log.info("Media bridged (pjsip <-> ALSA loopback)")
                self.app.start_session(self)

class SmokeAccount(pj.Account):
    def __init__(self, app): super().__init__(); self.app = app
    def onRegState(self, prm):
        info = self.getInfo()
        log.info("REGISTER active=%s code=%s txt=%s", info.regIsActive, prm.code, info.regStatusText)
    def onIncomingCall(self, prm):
        log.info("Incoming call")
        call = SmokeCall(self, self.app)
        op = pj.CallOpParam(); op.statusCode = 200
        call.answer(op)

# ---- App ----
class App:
    def __init__(self):
        self.ep = pj.Endpoint()
        self.acc = None
        self.call = None
        self.io   = None
        # models are lazy-loaded on first session
        self.whisper = None
        self.tts     = None

    def start(self):
        # sanity
        if not all([SIP_DOMAIN, SIP_EXT, SIP_AUTH_ID, SIP_PASSWORD]):
            raise RuntimeError("Missing THREECX_* env vars")

        # pjsip boot, with Null audio device to prevent segfaults while idle
        self.ep.libCreate()
        lc = pj.LogConfig(); lc.level = 3; lc.msgLogging = 0
        ec = pj.EpConfig();  ec.logConfig = lc
        self.ep.libInit(ec)
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, pj.TransportConfig())
        self.ep.libStart()

        adm = self.ep.audDevManager()
        adm.setNullDev()   # <---- stability: no real device until media is ACTIVE
        log.info("Started with Null audio device")

        # account/registration
        acfg = pj.AccountConfig()
        acfg.idUri = f"sip:{SIP_EXT}@{SIP_DOMAIN}"
        acfg.regConfig.registrarUri = f"sip:{SIP_DOMAIN}"
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "3CXPhoneSystem", SIP_AUTH_ID, 0, SIP_PASSWORD)
        )
        self.acc = SmokeAccount(self)
        self.acc.create(acfg)

        log.info("Ready. Call the DID/extension; it will auto-answer. Ctrl+C to quit.")

    # called from onCallMediaState
    def start_session(self, call: SmokeCall):
        self.call = call

        # Switch from Null to ALSA loopback devices now that media is active
        adm = self.ep.audDevManager()
        cap, play = self._pick_loopback_indices(adm)
        adm.setCaptureDev(cap); adm.setPlaybackDev(play)
        log.info("Pinned pjsip devices cap=%s play=%s", cap, play)

        # open PyAudio loopback and greet
        self.io = LoopbackIO(rate=SAMPLE_RATE, frames=FRAME,
                             in_index=PA_IN_INDEX, out_index=PA_OUT_INDEX)
        self.io.start()

        # Lazy-load models to keep registration path clean/stable
        self._ensure_models()

        threading.Thread(target=self._one_turn, daemon=True).start()

    def _ensure_models(self):
        if self.whisper is None:
            import whisper as _wh
            log.info("Loading Whisper: %s", WHISPER_MODEL)
            self.whisper = _wh.load_model(WHISPER_MODEL)
        if self.tts is None:
            from TTS.api import TTS as _TTS
            log.info("Loading Coqui TTS: %s", TTS_MODEL)
            self.tts = _TTS(model_name=TTS_MODEL, gpu=True, progress_bar=False)

    def _one_turn(self):
        # greet
        self._say("Hello! This is an AI test. After the beep, ask a short question.")
        time.sleep(0.25); self._beep()

        # record ~1 utterance until 500 ms silence
        import webrtcvad
        vad = webrtcvad.Vad(2)
        spoken = False; silence = 0; frames = []
        while self.call and self.call.isActive():
            f = self.io.read()  # int16 @ 8k, 20 ms
            if vad.is_speech(f.tobytes(), SAMPLE_RATE):
                spoken = True; silence = 0; frames.append(f)
            else:
                if spoken: silence += 1
                if spoken and silence >= 25:  # 25*20ms ≈ 500ms
                    break

        if not frames:
            self._say("I didn’t hear anything. Goodbye."); self.hangup(); return

        # STT (Whisper expects 16k float)
        audio_8k  = np.concatenate(frames)
        audio_16k = resample_i16(audio_8k, SAMPLE_RATE, 16000).astype(np.int16)
        af        = audio_16k.astype(np.float32) / 32768.0
        t0 = time.time()
        res = self.whisper.transcribe(af, language="en", fp16=False)
        user = (res.get("text") or "").strip()
        log.info("STT %.2fs: %s", time.time() - t0, user)

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
            log.info("LLM %.2fs", time.time() - t1)
        except Exception as e:
            log.error("Ollama error: %s", e)
            reply = "I’m having trouble reaching the model right now."

        # TTS -> play
        self._say(reply)
        self.hangup()

    def _pick_loopback_indices(self, adm: pj.AudDevManager):
        cap = play = None
        for i in range(0, 64):
            try:
                info = adm.getDevInfo(i)
            except Exception:
                continue
            name = (info.name or "").lower()
            if "loopback" in name:
                if info.inputCount  > 0 and cap  is None: cap  = i
                if info.outputCount > 0 and play is None: play = i
        if cap is None or play is None:
            raise RuntimeError("pjsua2 cannot find ALSA Loopback. Load `snd-aloop`.")
        return cap, play

    def _say(self, text: str):
        log.info("TTS: %s", text[:120])
        wav = np.asarray(self.tts.tts(text=text), dtype=np.float32)  # ~22.05k
        pcm8 = resample_i16((wav * 32767).astype(np.int16), 22050, SAMPLE_RATE)
        self.io.play(pcm8)

    def _beep(self, freq=1000, ms=220):
        n = int(SAMPLE_RATE * ms / 1000)
        t = np.arange(n) / SAMPLE_RATE
        wave = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        self.io.play((wave * 32767).astype(np.int16))

    def hangup(self):
        try:
            if self.call and self.call.isActive():
                self.call.hangup(pj.HangupParam())
        except Exception:
            pass

    def end_session(self):
        try:
            if self.io: self.io.stop_close()
        finally:
            self.io = None; self.call = None

    def stop(self):
        self.end_session()
        try: self.ep.hangupAllCalls()
        except Exception: pass
        self.ep.libDestroy()

# ---- main ----
if __name__ == "__main__":
    app = App()
    app.start()

    def _stop(*_):
        app.stop(); raise SystemExit

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        time.sleep(1)
