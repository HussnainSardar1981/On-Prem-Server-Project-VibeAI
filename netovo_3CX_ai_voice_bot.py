#!/usr/bin/env python3
"""
Milestone 1 voice bot:
- Reads your .env (THREECX_* keys)
- Registers to 3CX and keeps account alive
- Forces ALSA Loopback by enumerating devices by INDEX (portable across pjsua2 builds)
- Auto-answers inbound calls and bridges audio
- Optionally dials ECHO_EXTENSION (e.g., *777) once on startup
- AI pipeline is OFF for M1 stability
"""

import os, sys, time, signal
from dotenv import load_dotenv
import pjsua2 as pj

# ---------- Load your .env ----------
load_dotenv()

# 3CX data (your exact keys)
SIP_DOMAIN   = os.getenv("THREECX_SERVER", "").strip()
SIP_PORT     = int(os.getenv("THREECX_PORT", "5060"))
SIP_EXT      = os.getenv("THREECX_EXTENSION", "").strip()
SIP_AUTH_ID  = os.getenv("THREECX_AUTH_ID", "").strip()     # Authentication ID
SIP_PASSWORD = os.getenv("THREECX_PASSWORD", "").strip()

# Optional quick-dial targets
ECHO_EXTENSION = os.getenv("ECHO_EXTENSION", "*777").strip()
TEST_EXTENSION = os.getenv("TEST_EXTENSION", "1680").strip()

# Audio knobs (not used directly in M1 bridging, but kept for completeness)
SAMPLE_RATE   = int(os.getenv("SAMPLE_RATE", "8000"))
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1024"))

ENABLE_AI = False  # keep False for M1; turn True later if you add STT/LLM/TTS here

# ---------- PJSUA2 helpers ----------
class M1Call(pj.Call):
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
        """Portable: pick Loopback devices by index via getDevInfo(i)."""
        adm = self.ep.audDevManager()

        # We don't trust enum vector contents for IDs; use indices with getDevInfo(i).
        # Try a reasonable upper bound; if your build exposes a count method, you can swap this.
        indices = []
        try:
            # Newer bindings: enumerate via enumDev2()/enumDev(); fallback to probe indices 0..63
            devs = adm.enumDev2() if hasattr(adm, "enumDev2") else adm.enumDev()
            indices = list(range(len(devs)))
        except Exception:
            indices = list(range(64))  # probe first 64 device indices safely

        cap = play = None
        seen = []
        for i in indices:
            try:
                info = adm.getDevInfo(i)
            except Exception:
                continue
            name = (info.name or "").lower()
            seen.append((i, info.name, info.inputCount, info.outputCount))
            if "loopback" in name:
                if info.inputCount  > 0 and cap  is None: cap  = i
                if info.outputCount > 0 and play is None: play = i

        if cap is None or play is None:
            print("[AUDIO] Available devices:")
            for i, n, ic, oc in seen:
                print(f"   id={i:2d}  name={n}  in={ic} out={oc}")
            raise RuntimeError("ALSA Loopback not found. Ensure 'snd-aloop' is loaded and visible in arecord/aplay.")

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

        # UDP transport
        tcfg = pj.TransportConfig(); tcfg.port = 0
        self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
        self.ep.libStart()

        # Pin Loopback before account creation
        self._force_loopback()

        # Create/register account
        acfg = pj.AccountConfig()
        acfg.idUri  = f"sip:{SIP_EXT}@{SIP_DOMAIN}"
        acfg.regConfig.registrarUri   = f"sip:{SIP_DOMAIN}"
        acfg.regConfig.registerOnAdd  = True
        acfg.regConfig.retryIntervalSec = 60
        acfg.regConfig.timeoutSec       = 300
        acfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "3CXPhoneSystem", SIP_AUTH_ID, 0, SIP_PASSWORD)
        )

        self.acc = M1Account(on_incoming=self._on_media_active)
        self.acc.create(acfg)
        time.sleep(3)
        print("[SIP] Registered:", self.acc.getInfo().regIsActive)

        # Optional: place one echo call for proof
        if ECHO_EXTENSION:
            self.call = M1Call(self.acc)
            dst = f"sip:{ECHO_EXTENSION}@{SIP_DOMAIN}"
            print(f"[SIP] Dialing {ECHO_EXTENSION} for echo test…")
            self.call.makeCall(dst, pj.CallOpParam(True))

    def _on_media_active(self, call):
        self.call = call
        # If you later set ENABLE_AI=True, start your STT/LLM/TTS thread here.

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
    #   aplay -l ; arecord -l  (should list "Loopback")
    bot = M1Bot()
    bot.start()
    bot.run()
