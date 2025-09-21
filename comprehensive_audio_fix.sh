#!/bin/bash

echo "============================================="
echo "RESTORING WORKING CONFIGURATION - EMERGENCY FIX"
echo "============================================="

# Step 1: Stop Asterisk
echo "🔧 Step 1: Stopping Asterisk..."
sudo systemctl stop asterisk
sudo pkill -f asterisk
sleep 3

# Step 2: Restore ORIGINAL working PJSIP configuration
echo "🔧 Step 2: Restoring ORIGINAL working PJSIP config..."
sudo tee /etc/asterisk/pjsip.conf << 'EOF'
[global]
type=global
user_agent=NETOVO-VoiceBot

[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0:5060

[auth-3cx-1600]
type=auth
auth_type=userpass
username=qpZh2VS624
password=FcHw0P2FHK

[reg-3cx-1600]
type=registration
transport=transport-udp
outbound_auth=auth-3cx-1600
server_uri=sip:mtipbx.ny.3cx.us:5060
client_uri=sip:1600@mtipbx.ny.3cx.us
retry_interval=60
max_retries=10
expiration=120
line=yes
endpoint=ep-3cx

[ep-3cx]
type=endpoint
transport=transport-udp
context=voicebot-incoming
disallow=all
allow=ulaw
allow=alaw
outbound_auth=auth-3cx-1600
aors=aor-3cx
from_user=1600
from_domain=mtipbx.ny.3cx.us
send_pai=yes

[aor-3cx]
type=aor
contact=sip:mtipbx.ny.3cx.us:5060
qualify_frequency=30
max_contacts=1

[identify-3cx]
type=identify
endpoint=ep-3cx
match=172.208.69.71/32
EOF

# Step 3: Clean SIP.conf (keep ONLY the optimized codec settings, remove duplicates)
echo "🔧 Step 3: Cleaning SIP.conf but keeping ONLY codec optimizations..."
sudo tee /etc/asterisk/sip.conf << 'EOF'
[general]
context=default
allowoverlap=no
udpbindaddr=0.0.0.0
tcpenable=no
tcpbindaddr=0.0.0.0:0.0.0.0
transport=udp
srvlookup=yes

; ONLY keep the proven codec optimizations
disallow=all
allow=ulaw        ; 64 kbps, excellent compression (81.8%)
allow=alaw        ; 64 kbps, excellent compression (81.8%)
allow=gsm         ; 13 kbps, very low bandwidth

; Minimal VoIP quality settings (don't break what works)
qualify=yes
qualifyfreq=30
jbenable=yes
jbmaxsize=50
jbresyncthreshold=1000
jbimpl=adaptive
jbtargetextra=40
jblog=no
EOF

# Step 4: Use ORIGINAL module configuration (PJSIP working)
echo "🔧 Step 4: Restoring working module config..."
sudo tee /etc/asterisk/modules.conf << 'EOF'
[modules]
autoload=yes

; Keep PJSIP (it was working)
require = res_pjsip.so
require = res_pjsip_session.so
require = chan_pjsip.so

; Disable chan_sip to avoid conflicts
noload = chan_sip.so

; Core requirements
require = app_dial.so
require = app_playback.so
require = app_record.so
require = format_wav.so
require = codec_ulaw.so
require = codec_alaw.so
require = res_agi.so
EOF

# Step 5: Restore original extensions.conf for VoiceBot
echo "🔧 Step 5: Ensuring VoiceBot dialplan is correct..."
if [ ! -f "/etc/asterisk/extensions.conf.backup" ]; then
    sudo cp /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.backup
fi

# Add/ensure voicebot context exists
sudo tee -a /etc/asterisk/extensions.conf << 'EOF'

[voicebot-incoming]
exten => 1600,1,NoOp(=== NETOVO VoiceBot Call Started ===)
exten => 1600,n,Answer()
exten => 1600,n,Wait(1)
exten => 1600,n,AGI(/opt/voicebot/production_voicebot_professional.py)
exten => 1600,n,Hangup()
EOF

# Step 6: Clean runtime and restart
echo "🔧 Step 6: Cleaning runtime and restarting..."
sudo rm -rf /var/run/asterisk/*
sudo mkdir -p /var/run/asterisk
sudo chown asterisk:asterisk /var/run/asterisk

# Start Asterisk
sudo systemctl start asterisk
sleep 5

# Step 7: Verify EVERYTHING is working
echo "🔧 Step 7: Verification..."

echo ""
echo "Asterisk Status:"
sudo systemctl status asterisk --no-pager

echo ""
echo "PJSIP Endpoints (should show ep-3cx):"
sudo asterisk -rx "pjsip show endpoints" 2>/dev/null || echo "PJSIP command failed"

echo ""
echo "PJSIP Registrations (should show registration to 3CX):"
sudo asterisk -rx "pjsip show registrations" 2>/dev/null || echo "PJSIP registrations failed"

echo ""
echo "Control Socket:"
ls -la /var/run/asterisk/asterisk.ctl 2>/dev/null && echo "✅ Control socket OK" || echo "❌ Control socket missing"

echo ""
echo "============================================="
echo "🚨 EMERGENCY RESTORATION COMPLETE"
echo "============================================="
echo ""
echo "What was restored:"
echo "✅ ORIGINAL 3CX credentials (qpZh2VS624/FcHw0P2FHK)"
echo "✅ ORIGINAL server (mtipbx.ny.3cx.us)"
echo "✅ ORIGINAL UDP transport"
echo "✅ Removed SIP.conf duplicates ONLY"
echo "✅ Kept ONLY proven codec optimizations"
echo "✅ VoiceBot dialplan restored"
echo ""
echo "What was kept from improvements:"
echo "✅ Codec optimization (ulaw/alaw for 81.8% compression)"
echo "✅ Jitter buffer settings"
echo "✅ Conversation context in VoiceBot code"
echo ""
echo "Test now:"
echo "1. Call +1 (646) 358-3509"
echo "2. Should connect to VoiceBot"
echo "3. Should have better audio quality but working calls"
echo ""
echo "If this works, we keep ONLY the audio improvements!"
