#!/usr/bin/env python3
import os, sys, time, signal, logging, threading, requests
from dataclasses import dataclass
import numpy as np
from scipy.signal import resample_poly
import pyaudio
import pjsua2 as pj
from dotenv import load_dotenv

# ---------- load .env ----------
load_dotenv()

# Back-compat env mapping (so your existing names drive the bot)
_COMPAT = {
    "SIP_DOMAIN":    "THREECX_SERVER",
    "SIP_EXT":       "THREECX_EXTENSION",
    "SIP_AUTH_USER": "THREECX_AUTH_ID",
    "SIP_AUTH_PASS": "THREECX_PASSWORD",
    "OUTBOUND_DIAL": "ECHO_EXTENSION",
    "WHISPER_MODEL": "WHISPER_MODEL",
    "OLLAMA_URL":    "OLLAMA_URL",
    "OLLAMA_MODEL":  "OLLAMA_MODEL",
    "COQUI_VOICE":   "TTS_MODEL",
    "SAMPLE_RATE":   "SAMPLE_RATE",
    "CHUNK_SIZE":    "CHUNK_SIZE",
}
for new, old in _COMPAT.items():
    v = os.getenv(old)
    if v and not os.getenv(new):
        os.environ[new] = v

# ---------- config ----------
@dataclass
class Cfg:
    SIP_DOMAIN: str = os.getenv("SIP_DOMAIN", "")
    SIP_EXT: str = os.getenv("SIP_EXT", "")
    SIP_AUTH_USER: str = os.getenv("SIP_AUTH_USER", "")
    SIP_AUTH_PASS: str = os.getenv("SIP_AUTH_PASS", "")
    OUTBOUND_DIAL: str | None = (os.getenv("OUTBOUND_DIAL") or "").strip() or None
    ENABLE_AI: bool = os.getenv("ENABLE_AI", "true").lower() in ("1","true","yes")

    SIP_SR: int = int(os.getenv("SAMPLE_RATE", "8000"))
    CH: int = 1
    CHUNK_MS: int = 20
    if os.getenv("CHUNK_SIZE"):
        _sz = int(os.getenv("CHUNK_SIZE"))
        CHUNK_MS = max(5, round(_sz * 1000 / SIP_SR))

    # AI
    FORCE_GPU: bool = os.getenv("FORCE_GPU", "true").lower() in ("1","true","yes")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL","base")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL","http://127.0.0.1:11434/api/generate")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL","orca2:7b")
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT","You are a concise helpful phone assistant.")
    COQUI_VOICE: str = os.getenv("COQUI_VOICE","tts_models/en/ljspeech/tacotron2-DDC")
    MAX_TURN_SEC: int = 8
    SILENCE_THRESH: float = 0.015
    MAX_CALL_SEC: int = 600

    # Logs
    LOG_LEVEL: str = os.getenv("LOG_LEVEL","INFO")
    LOG_FILE: str = os.getenv("LOG_FILE","voice_bot.log")

CFG = Cfg()

# ---------- logging ----------
level = getattr(logging, CFG.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CFG.LOG_FILE)
    ],
)
log = logging.getLogger("voicebot")

# ---------- lazy model loaders ----------
_whisper = None
_tts = None
def load_models():
    global _whisper, _tts
    if not CFG.ENABLE_AI:
        return
    if _whisper is None:
        import whisper, torch
        device = "cuda" if CFG.FORCE_GPU and torch.cuda.is_available() else "cpu"
        _whisper = whisper.load_model(CFG.WHISPER_MODEL).to(device)
        log.info(f"Whisper loaded on {device}")
    if _tts is None:
        from TTS.api import TTS
        _tts = TTS(CFG.COQUI_VOICE)
        log.info(f"Coqui TTS loaded: {CFG.COQUI_VOICE}")

def ask_ollama(prompt, system_prompt):
    payload = {"model": CFG.OLLAMA_MODEL, "prompt": prompt, "system": system_prompt, "stream": False}
    r = requests.post(CFG.OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    j = r.json()
    return j.get("response") or j.get("data","")

def resample_mono(x, sr_from, sr_to):
    if sr_from == sr_to: return x
    g = np.gcd(sr_from, sr_to)
    up, down = sr_to//g, sr_from//g
    return resample_poly(x, up, down)

# ---------- ALSA Loopback I/O ----------
class LoopbackIO:
    def __init__(self):
        import pyaudio
        self.pa = pyaudio.PyAudio()
        self.in_idx, self.out_idx = self._pick_indices()
        self.in_stream = self.out_stream = None
        self.chunk = int(CFG.SIP_SR * CFG.CHUNK_MS / 1000)

    def _pick_indices(self):
        in_idx = out_idx = None
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = info["name"].lower()
            if "loopback" in name:
                if info["maxInputChannels"] > 0 and in_idx is None:  in_idx = i
                if info["maxOutputChannels"] > 0 and out_idx is None: out_idx = i
        if in_idx is None or out_idx is None:
            raise RuntimeError("ALSA Loopback devices not found. `sudo modprobe snd-aloop` and retry.")
        return in_idx, out_idx

    def open(self):
        self.in_stream = self.pa.open(format=pyaudio.paInt16, channels=CFG.CH, rate=CFG.SIP_SR,
                                      input=True, frames_per_buffer=self.chunk, input_device_index=self.in_idx)
        self.out_stream = self.pa.open(format=pyaudio.paInt16, channels=CFG.CH, rate=CFG.SIP_SR,
                                       output=True, frames_per_buffer=self.chunk, output_device_index=self.out_idx)

    def read_chunk(self):
        data = self.in_stream.read(self.chunk, exception_on_overflow=False)
        return (np.frombuffer(data, dtype=np.int16).astype(np.float32))/32768.0

    def write_pcm(self, x_float):
        x = np.clip(x_float, -1.0, 1.0)
        self.out_stream.write((x*32767.0).astype(np.int16).tobytes())

    def close(self):
        try:
            if self.in_stream: self.in_stream.close()
            if self.out_stream: self.out_stream.close()
        finally:
            self.pa.terminate()

# ---------- pjsua2 classes ----------
class BotCall(pj.Call):
    def __init__(self, acc, io: LoopbackIO, on_active=None):
        super().__init__(acc, pj.PJSUA_INVALID_ID)
        self.io = io
        self.on_active = on_active

    def onCallState(self, prm):
        ci = self.getInfo()
        log.info(f"CALL state={ci.stateText} code={ci.lastStatusCode}")

    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                am = pj.AudioMedia.typecastFromMedia(self.getMedia(mi.index))
                adm = pj.Endpoint.instance().audDevManager()
                am.startTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().startTransmit(am)
                log.info("Media bridged to ALSA Loopback")
                if self.on_active: self.on_active(self)

class BotAccount(pj.Account):
    def __init__(self, io: LoopbackIO, on_new_call):
        super().__init__()
        self.io = io
        self.on_new_call = on_new_call

    def onRegState(self, prm):
        log.info(f"REGISTER active={self.getInfo().regIsActive} code={prm.code}")

    def onIncomingCall(self, prm):
        call = BotCall(self, self.io, on_active=self.on_new_call)
        log.info("Incoming call → answering")
        call.answer(pj.CallOpParam(True))

class VoiceBot:
    def __init__(self):
        self.ep = pj.Endpoint(); self.ep.libCreate()
        self.io = LoopbackIO()
        self.acc = None
        self.active_call = None
        self.stop = threading.Event()

    def start(self):
        ep_cfg = pj.EpConfig()
        log_cfg = pj.LogConfig(); log_cfg.level = 3; log_cfg.msgLogging = 0
        ep_cfg.logConfig = log_cfg
        self.ep.libInit(ep_cfg)

        tcfg = pj.TransportConfig(); tcfg.port = 0
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        self.ep.libStart()
        self.io.open()

        acfg = pj.AccountConfig()
        acfg.idUri = f"sip:{CFG.SIP_EXT}@{CFG.SIP_DOMAIN}"
        acfg.regConfig.registrarUri = f"sip:{CFG.SIP_DOMAIN}"
        acfg.regConfig.registerOnAdd = True
        acfg.regConfig.retryIntervalSec = 60
        acfg.regConfig.timeoutSec = 300
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest","3CXPhoneSystem",CFG.SIP_AUTH_USER,0,CFG.SIP_AUTH_PASS)
        )
        self.acc = BotAccount(self.io, on_new_call=self._on_call_active)
        self.acc.create(acfg)
        time.sleep(3)
        log.info(f"Registered: {self.acc.getInfo().regIsActive}")

        if CFG.OUTBOUND_DIAL:
            self.active_call = BotCall(self.acc, self.io, on_active=self._on_call_active)
            log.info(f"Dialing {CFG.OUTBOUND_DIAL}…")
            self.active_call.makeCall(f"sip:{CFG.OUTBOUND_DIAL}@{CFG.SIP_DOMAIN}", pj.CallOpParam(True))

    def _on_call_active(self, _call):
        self.active_call = _call
        if CFG.ENABLE_AI:
            threading.Thread(target=self._turn_loop, daemon=True).start()

    def _turn_loop(self):
        load_models()
        frames_needed = CFG.SIP_SR * CFG.MAX_TURN_SEC
        buf = np.zeros(0, dtype=np.float32)
        t0 = time.time()
        while not self.stop.is_set():
            x = self.io.read_chunk()
            buf = np.concatenate([buf, x])
            # simple end-of-turn: length or 1s of silence window
            if len(buf) >= frames_needed or (len(buf) > CFG.SIP_SR and np.sqrt(np.mean(buf[-CFG.SIP_SR:]**2)) < CFG.SILENCE_THRESH):
                if len(buf) == 0:
                    continue
                # STT
                import whisper
                up = resample_mono(buf, CFG.SIP_SR, 16000)
                result = _whisper.transcribe(up, fp16=False, language='en')
                user = (result.get("text") or "").strip()
                log.info(f"ASR: {user!r}")
                buf = np.zeros(0, dtype=np.float32)
                if not user:
                    continue
                # LLM
                reply = ask_ollama(user, CFG.SYSTEM_PROMPT).strip()
                log.info(f"LLM: {reply}")
                # TTS
                wav = _tts.tts(text=reply, speaker=0)
                if isinstance(wav, tuple):
                    wav, tts_sr = wav
                else:
                    tts_sr = getattr(_tts, "output_sample_rate", 22050)
                if isinstance(wav, list): wav = np.array(wav, dtype=np.float32)
                y = resample_mono(np.asarray(wav, dtype=np.float32), tts_sr, CFG.SIP_SR)
                y = y / (np.max(np.abs(y)) + 1e-6)
                self.io.write_pcm(y)
            if time.time() - t0 > CFG.MAX_CALL_SEC:
                log.info("Max call duration reached; ending loop")
                break

    def run_forever(self):
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop.set()
        try:
            self.io.close()
        finally:
            self.ep.hangupAllCalls()
            self.ep.libDestroy()
            log.info("Shutdown complete")

def main():
    required = [CFG.SIP_DOMAIN, CFG.SIP_EXT, CFG.SIP_AUTH_USER, CFG.SIP_AUTH_PASS]
    if not all(required):
        log.error("Missing SIP env vars. Check .env.")
        sys.exit(1)
    bot = VoiceBot()
    bot.start()
    bot.run_forever()

if __name__ == "__main__":
    main()
