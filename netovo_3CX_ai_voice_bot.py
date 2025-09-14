#!/usr/bin/env python3
"""
Milestone 1 voice bot:
- Loads 3CX config from .env (THREECX_* keys you provided)
- Registers and keeps the account alive
- Forces ALSA Loopback for capture/playback to avoid ALSA 'Unknown PCM' segfaults
- Auto-answers inbound calls and bridges audio (call <DID> → ext 1600)
- Optionally dials ECHO_EXTENSION (e.g. *777) once on startup for a quick test
- AI pipeline (Whisper/Ollama/Coqui) is OFF by default for M1 stability
"""

import os, sys, time, signal
from dotenv import load_dotenv
import pjsua2 as pj

# ---------- Load your .env ----------
load_dotenv()

# 3CX data (exactly your keys)
SIP_DOMAIN   = os.getenv("THREECX_SERVER", "").strip()
SIP_PORT     = int(os.getenv("THREECX_PORT", "5060"))
SIP_EXT      = os.getenv("THREECX_EXTENSION", "").strip()
SIP_AUTH_ID  = os.getenv("THREECX_AUTH_ID", "").strip()     # Authentication ID
SIP_PASSWORD = os.getenv("THREECX_PASSWORD", "").strip()

# Optional quick-dial targets (your keys)
ECHO_EXTENSION = os.getenv("ECHO_EXTENSION", "*777").strip()
TEST_EXTENSION = os.getenv("TEST_EXTENSION", "1680").strip()

# Audio knobs (your keys; CHUNK_SIZE is not used for M1 logic here)
SAMPLE_RATE   = int(os.getenv("SAMPLE_RATE", "8000"))
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1024"))

# ---- Toggle AI later if you want (kept False for M1 stability) ----
ENABLE_AI = False  # set to True after M1 if you want STT→LLM→TTS in this same process

# ---------- PJSUA2 helpers ----------
class M1Call(pj.Call):
    """Bridges call audio to ALSA Loopback."""
    def onCallState(self, prm):
        ci = self.getInfo()
        print(f"[CALL] state={ci.stateText} code={ci.lastStatusCode}")
        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            print("[CALL] Disconnected")

    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                am  = pj.AudioMedia.typecastFromMedia(self.getMedia(mi.index))
                adm = pj.Endpoint.instance().audDevManager()
                am.startTransmit(adm.getPlaybackDevMedia())
                adm.getCaptureDevMedia().startTransmit(am)
                print("[CALL] Media bridged to ALSA Loopback")

class M1Account(pj.Account):
    def __init__(self, on_incoming):
        super().__init__()
        self._on_incoming = on_incoming

    def onRegState(self, prm):
        print(f"[REG] active={self.getInfo().regIsActive} code={prm.code}")

    def onIncomingCall(self, prm):
        print("[SIP] Incoming call → answering")
        call = M1Call(self)
        call.answer(pj.CallOpParam(True))
        if self._on_incoming:
            self._on_incoming(call)

# ---------- App ----------
class M1Bot:
    def __init__(self):
        self.ep   = pj.Endpoint()
        self.acc  = None
        self.call = None
        self.running = True

    def _force_loopback(self):
        """Select ALSA Loopback explicitly to avoid ALSA segfaults."""
        adm  = self.ep.audDevManager()
        devs = adm.enumDev2()
        # Print once for debugging:
        # for d in devs: print(d.devId, d.name, d.inputCount, d.outputCount)
        cap  = next(d.devId for d in devs if "loopback" in d.name.lower() and d.inputCount  > 0)
        play = next(d.devId for d in devs if "loopback" in d.name.lower() and d.outputCount > 0)
        adm.setCaptureDev(cap)
        adm.setPlaybackDev(play)
        print(f"[AUDIO] Using ALSA Loopback cap={cap} play={play}")

    def start(self):
        if not (SIP_DOMAIN and SIP_EXT and SIP_AUTH_ID and SIP_PASSWORD):
            print("[FATAL] Missing SIP env vars; check your .env.")
            sys.exit(1)

        self.ep.libCreate()

        # modest logging (raise to 5 for deep SIP traces)
        log_cfg = pj.LogConfig(); log_cfg.level = 3; log_cfg.msgLogging = 0
        ep_cfg  = pj.EpConfig();  ep_cfg.logConfig = log_cfg
        self.ep.libInit(ep_cfg)

        # UDP transport (port=0 lets OS pick a free port)
        tcfg = pj.TransportConfig(); tcfg.port = 0
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        self.ep.libStart()

        # pin Loopback before creating account
        self._force_loopback()

        # Create/register account
        acfg = pj.AccountConfig()
        acfg.idUri  = f"sip:{SIP_EXT}@{SIP_DOMAIN}"
        acfg.regConfig.registrarUri   = f"sip:{SIP_DOMAIN}"
        acfg.regConfig.registerOnAdd  = True
        acfg.regConfig.retryIntervalSec = 60
        acfg.regConfig.timeoutSec       = 300
        # Realm must match 3CX challenge
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "3CXPhoneSystem", SIP_AUTH_ID, 0, SIP_PASSWORD)
        )

        self.acc = M1Account(on_incoming=self._on_media_active)
        self.acc.create(acfg)
        time.sleep(3)
        print("[SIP] Registered:", self.acc.getInfo().regIsActive)

        # Optional: place one echo call on startup for proof
        if ECHO_EXTENSION:
            self.call = M1Call(self.acc)
            dst = f"sip:{ECHO_EXTENSION}@{SIP_DOMAIN}"
            print(f"[SIP] Dialing {ECHO_EXTENSION} for echo test…")
            self.call.makeCall(dst, pj.CallOpParam(True))

    def _on_media_active(self, call):
        """Hook when inbound call becomes active. (For M1 we only bridge.)"""
        self.call = call
        if ENABLE_AI:
            # For M1 we keep AI off; if you enable it later, start your STT/LLM/TTS thread here.
            pass

    def run(self):
        def _stop(sig, frm):
            print("\n[SYS] Shutting down…")
            self.running = False
        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)

        while self.running:
            time.sleep(0.5)

        try:
            self.ep.hangupAllCalls()
        finally:
            self.ep.libDestroy()
            print("[SYS] Clean exit.")

if __name__ == "__main__":
    # Make sure ALSA loopback exists:
    #   sudo modprobe snd-aloop
    #   aplay -l / arecord -l should list "Loopback"
    bot = M1Bot()
    bot.start()
    bot.run()
