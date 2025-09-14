#!/usr/bin/env python3
"""
Minimal E2E smoke test:
- Registers to 3CX (pjsua2)
- Bridges media to ALSA Loopback
- On inbound call: greet, capture one utterance, STT->LLM->TTS, play reply, hang up
"""

import os, time, signal, logging, threading, json
from dotenv import load_dotenv
import numpy as np
import requests
import webrtcvad
import pyaudio
from scipy.signal import resample_poly

import pjsua2 as pj
import whisper
from TTS.api import TTS

load_dotenv()

# ---------- ENV ----------
SIP_DOMAIN   = os.getenv("THREECX_SERVER")
SIP_PORT     = int(os.getenv("THREECX_PORT","5060"))
SIP_EXT      = os.getenv("THREECX_EXTENSION")
SIP_AUTH_ID  = os.getenv("THREECX_AUTH_ID")
SIP_PASSWORD = os.getenv("THREECX_PASSWORD")

OLLAMA_URL   = os.getenv("OLLAMA_URL","http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL","orca2:7b")
WHISPER_MODEL= os.getenv("WHISPER_MODEL","base")
TTS_MODEL    = os.getenv("TTS_MODEL","tts_models/en/ljspeech/tacotron2-DDC")

SAMPLE_RATE  = int(os.getenv("SAMPLE_RATE","8000"))  # SIP edge rate
FRAME        = int(os.getenv("CHUNK_SIZE","160"))    # 20ms @ 8k
VAD_MODE     = int(os.getenv("VAD_AGGRESSIVENESS","2"))

# Optional: force PortAudio loopback device indices if auto-pick fails
PA_IN_INDEX  = os.getenv("LOOPBACK_IN_INDEX")
PA_OUT_INDEX = os.getenv("LOOPBACK_OUT_INDEX")
PA_IN_INDEX  = int(PA_IN_INDEX) if PA_IN_INDEX else None
PA_OUT_INDEX = int(PA_OUT_INDEX) if PA_OUT_INDEX else None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("smoke")

# ---------- Audio helpers ----------
class LoopbackIO:
    def __init__(self, rate=8000, frames=160, in_index=None, out_index=None):
        self.pa = pyaudio.PyAudio()
        if in_index is None or out_index is None:
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
        ins, outs = [], []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            name = (info.get("name") or "").lower()
            if "loopback" in name:
                if info.get("maxInputChannels",0)  > 0: ins.append(i)
                if info.get("maxOutputChannels",0) > 0: outs.append(i)
        if not ins or not outs:
            raise RuntimeError("No PortAudio loopback devices. `sudo modprobe snd-aloop` then try again.")
        return ins[-1], outs[0]

    def start(self):
        if not self.in_stream.is_active():  self.in_stream.start_stream()
        if not self.out_stream.is_active(): self.out_stream.start_stream()

    def stop_close(self):
        try:
            if self.in_stream.is_active():  self.in_stream.stop_stream()
            if self.out_stream.is_active(): self.out_stream.stop_stream()
        finally:
            self.in_stream.close(); self.out_stream.close(); self.pa.terminate()

    def read(self):
        data = self.in_stream.read(self.frames, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def play(self, pcm_i16: np.ndarray):
        self.out_stream.write(pcm_i16.astype(np.int16).tobytes())

def resample_i16(x_i16: np.ndarray, src: int, dst: int) -> np.ndarray:
    x = x_i16.astype(np.float32) / 32768.0
    g = np.gcd(src, dst); up = dst//g; down = src//g
    y = resample_poly(x, up, down)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)

# ---------- PJSUA2 classes ----------
class SmokeCall(pj.Call):
    def __init__(self, acc, app): super().__init__(acc); self.app=app
    def onCallState(self, prm):
        ci = self.getInfo()
        log.info("CALL %s (%s)", ci.stateText, ci.lastStatusCode)
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            self.app.end_session()
    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for m in ci.media:
            if m.type == pj.PJMEDIA_TYPE_AUDIO and m.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                am = pj.AudioMedia.typecastFromMedia(self.getMedia(m.index))
                adm = pj.Endpoint.instance().audDevManager()
                am.startTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().startTransmit(am)
                log.info("Media bridged to ALSA Loopback")
                self.app.start_session(self)

class SmokeAccount(pj.Account):
    def __init__(self, app): super().__init__(); self.app=app
    def onRegState(self, prm):
        info = self.getInfo()
        log.info("REGISTER active=%s code=%s", info.regIsActive, prm.code)
    def onIncomingCall(self, prm):
        log.info("Incoming call")
        call = SmokeCall(self, self.app)
        op = pj.CallOpParam(); op.statusCode = 200
        call.answer(op)

# ---------- App ----------
class App:
    def __init__(self):
        # models
        log.info("Loading Whisper: %s", WHISPER_MODEL)
        self.whisper = whisper.load_model(WHISPER_MODEL)
        log.info("Loading Coqui TTS: %s", TTS_MODEL)
        self.tts = TTS(model_name=TTS_MODEL, gpu=True, progress_bar=False)

        self.ep = pj.Endpoint(); self.acc=None
        self.io=None; self.call=None
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.running=True

    def start(self):
        if not all([SIP_DOMAIN,SIP_EXT,SIP_AUTH_ID,SIP_PASSWORD]):
            raise RuntimeError("Missing SIP env (THREECX_*).")
        # pjsua2 init
        self.ep.libCreate()
        lc = pj.LogConfig(); lc.level=3; lc.msgLogging=0
        ec = pj.EpConfig();  ec.logConfig = lc
        self.ep.libInit(ec)
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, pj.TransportConfig())
        self.ep.libStart()

        # pin ALSA loopback by index
        adm = self.ep.audDevManager()
        cap, play = self._pick_loopback_indices(adm)
        adm.setCaptureDev(cap); adm.setPlaybackDev(play)
        log.info("Pinned pjsip devices cap=%s play=%s", cap, play)

        # account
        acfg = pj.AccountConfig()
        acfg.idUri = f"sip:{SIP_EXT}@{SIP_DOMAIN}"
        acfg.regConfig.registrarUri = f"sip:{SIP_DOMAIN}"
        acfg.sipConfig.authCreds.append(pj.AuthCredInfo("digest","3CXPhoneSystem",SIP_AUTH_ID,0,SIP_PASSWORD))
        self.acc = SmokeAccount(self)
        self.acc.create(acfg)
        time.sleep(3)

        log.info("Ready. Call the DID/extension now (it will auto-answer). Press Ctrl+C to quit.")

    def _pick_loopback_indices(self, adm: pj.AudDevManager):
        # portable: scan names, choose any "loopback"
        cap=play=None
        for i in range(0,64):
            try: info = adm.getDevInfo(i)
            except Exception: continue
            nm = (info.name or "").lower()
            if "loopback" in nm:
                if info.inputCount  > 0 and cap  is None: cap  = i
                if info.outputCount > 0 and play is None: play = i
        if cap is None or play is None:
            raise RuntimeError("pjsua2 couldn't find ALSA Loopback. `sudo modprobe snd-aloop` then retry.")
        return cap, play

    # media session callbacks
    def start_session(self, call: SmokeCall):
        self.call = call
        # Open PortAudio loopback (caller in / bot out)
        self.io = LoopbackIO(rate=SAMPLE_RATE, frames=FRAME, in_index=PA_IN_INDEX, out_index=PA_OUT_INDEX)
        self.io.start()
        # Do one full turn then hangup
        threading.Thread(target=self._one_turn, daemon=True).start()

    def _one_turn(self):
        # greet
        self._say("Hello! This is an AI test. Please ask a short question after the beep.")
        time.sleep(0.3)
        self._beep()

        # record until ~500ms silence after speech
        buf=[]; silence=0; spoken=False
        while self.call and self.call.isActive():
            frame = self.io.read()  # int16, 20ms
            speech = self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)
            if speech:
                spoken=True; silence=0; buf.append(frame)
            else:
                if spoken: silence += 1
                if spoken and silence >= 25:  # ~500ms
                    break

        if not buf:
            self._say("I didn’t hear anything. Hanging up now."); self.hangup(); return

        # STT
        audio = np.concatenate(buf)               # int16 @ 8k
        a16  = resample_i16(audio, SAMPLE_RATE, 16000).astype(np.int16)
        af   = a16.astype(np.float32)/32768.0
        t0=time.time()
        res = self.whisper.transcribe(af, language="en", fp16=False)
        text = (res.get("text") or "").strip()
        log.info("STT %.2fs: %s", time.time()-t0, text)

        if not text:
            self._say("Sorry, I didn’t catch that. Goodbye."); self.hangup(); return

        # LLM
        prompt = (
            "You are a concise helpful assistant. Answer in 1–2 sentences.\n"
            f"User: {text}\nAssistant:"
        )
        try:
            t1=time.time()
            r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                                                "options":{"temperature":0.4,"num_predict":120}}, timeout=25)
            r.raise_for_status()
            reply = (r.json().get("response") or "").strip()
            log.info("LLM %.2fs", time.time()-t1)
        except Exception as e:
            log.error("Ollama error: %s", e); reply = "I’m having trouble reaching the model."

        # TTS
        self._say(reply)
        self.hangup()

    def _say(self, text: str):
        log.info("TTS: %s", text[:120])
        wav = np.asarray(self.tts.tts(text=text), dtype=np.float32)  # 22.05k
        pcm8 = resample_i16((wav*32767).astype(np.int16), 22050, SAMPLE_RATE)
        self.io.play(pcm8)

    def _beep(self, freq=1000, ms=200):
        n = int(SAMPLE_RATE*ms/1000)
        t = np.arange(n)/SAMPLE_RATE
        beep = (0.4*np.sin(2*np.pi*freq*t)).astype(np.float32)
        self.io.play((beep*32767).astype(np.int16))

    def hangup(self):
        if self.call and self.call.isActive():
            self.call.hangup(pj.HangupParam())

    def end_session(self):
        try:
            if self.io: self.io.stop_close()
        finally:
            self.io=None; self.call=None

    def stop(self):
        self.end_session()
        try: self.ep.hangupAllCalls()
        except Exception: pass
        self.ep.libDestroy()

# ---------- main ----------
if __name__ == "__main__":
    app = App()
    app.start()
    def _stop(*_):
        app.stop(); raise SystemExit
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
    while True: time.sleep(1)
