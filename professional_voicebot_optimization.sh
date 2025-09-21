#!/bin/bash

echo "============================================="
echo "PROFESSIONAL VOICEBOT OPTIMIZATION - NETOVO"
echo "============================================="

# Step 1: Optimize for G.711 codec as per Vitaliy's specifications
echo "🔧 Step 1: Optimizing for G.711 codec (3CX priority)..."

# Update PJSIP configuration for G.711 priority
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
allow=g711
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

# Step 2: Install premium TTS engine for professional voice quality
echo "🔧 Step 2: Installing professional TTS engine..."

# Install Festival with better voices
sudo apt-get update -y
sudo apt-get install -y festival festvox-us-slt-hts festvox-us-clb-hts sox

# Install premium TTS engines
sudo apt-get install -y espeak-ng flite

# Configure Festival for professional voice
sudo mkdir -p /etc/festival
sudo tee /etc/festival/siteinit.scm << 'EOF'
; Professional voice configuration
(set! voice_default 'voice_us_slt_arctic_hts)
(Parameter.set 'Audio_Method 'Audio_Command)
(Parameter.set 'Audio_Command "sox -t wav - -r 8000 -c 1 -t wav $FILE")
EOF

# Step 3: Optimize STT for better accuracy and speed
echo "🔧 Step 3: Optimizing Speech Recognition..."

# Install better Whisper model (base instead of tiny)
sudo pip3 install --upgrade openai-whisper
whisper --download base

# Install additional STT engines
sudo apt-get install -y python3-speech-recognition python3-pyaudio

# Step 4: Create optimized audio processing pipeline
echo "🔧 Step 4: Creating optimized audio pipeline..."

sudo tee /opt/voicebot/audio_optimizer.sh << 'EOF'
#!/bin/bash

# Audio preprocessing for better STT accuracy
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Normalize audio for better recognition
sox "$INPUT_FILE" "$OUTPUT_FILE" \
    remix 1 \
    rate 16000 \
    norm -3 \
    compand 0.3,1 6:-70,-60,-20 -5 -90 0.2 \
    lowpass 3400 \
    highpass 300

echo "Audio optimized: $OUTPUT_FILE"
EOF

chmod +x /opt/voicebot/audio_optimizer.sh

# Step 5: Automatic temp file cleanup (aggressive)
echo "🔧 Step 5: Implementing aggressive temp file cleanup..."

# Create real-time cleanup service
sudo tee /etc/systemd/system/voicebot-realtime-cleanup.service << 'EOF'
[Unit]
Description=VoiceBot Real-time Cleanup Service
Wants=voicebot-realtime-cleanup.timer

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'find /tmp -name "*speak*" -o -name "*tts*" -o -name "*enterprise*" -o -name "*customer_input*" -mmin +1 -delete 2>/dev/null'

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/voicebot-realtime-cleanup.timer << 'EOF'
[Unit]
Description=Run VoiceBot cleanup every 10 seconds
Requires=voicebot-realtime-cleanup.service

[Timer]
OnCalendar=*:*:0,10,20,30,40,50
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable aggressive cleanup
sudo systemctl daemon-reload
sudo systemctl enable voicebot-realtime-cleanup.timer
sudo systemctl start voicebot-realtime-cleanup.timer

# Step 6: Audio quality optimization for VoIP
echo "🔧 Step 6: VoIP audio quality optimization..."

# Optimize ALSA for VoIP
sudo tee /etc/asound.conf << 'EOF'
pcm.!default {
    type plug
    slave {
        pcm "hw:0,0"
        rate 8000
        channels 1
        format S16_LE
    }
}

ctl.!default {
    type hw
    card 0
}
EOF

# Audio latency optimization
sudo tee -a /etc/security/limits.conf << 'EOF'
asterisk soft rtprio 99
asterisk hard rtprio 99
asterisk soft memlock unlimited
asterisk hard memlock unlimited
EOF

# Step 7: Network optimization for G.711
echo "🔧 Step 7: Network optimization for G.711..."

# G.711 specific optimizations
sudo sysctl -w net.core.rmem_max=8388608
sudo sysctl -w net.core.wmem_max=8388608
sudo sysctl -w net.core.rmem_default=87380
sudo sysctl -w net.core.wmem_default=16384
sudo sysctl -w net.ipv4.udp_rmem_min=4096
sudo sysctl -w net.ipv4.udp_wmem_min=4096

# RTP optimization for G.711
sudo tee /etc/asterisk/rtp.conf << 'EOF'
[general]
rtpstart=9000
rtpend=10999
rtpchecksums=no
dtmftimeout=3000
rtcpinterval=5000
strictrtp=yes
probation=4
icesupport=yes
EOF

echo "============================================="
echo "✅ PROFESSIONAL OPTIMIZATION COMPLETE"
echo "============================================="
echo ""
echo "Applied Optimizations:"
echo "✅ G.711 codec priority (as per Vitaliy's specs)"
echo "✅ Professional TTS engines (Festival + premium voices)"
echo "✅ Better STT model (base instead of tiny)"
echo "✅ Audio preprocessing pipeline"
echo "✅ Real-time temp file cleanup (every 10 seconds)"
echo "✅ VoIP-optimized audio settings"
echo "✅ Network optimization for G.711"
echo ""
echo "Expected Improvements:"
echo "🎯 Professional voice quality (no breaking)"
echo "🎯 Better speech recognition accuracy"
echo "🎯 Faster response times"
echo "🎯 No temp file accumulation"
echo "🎯 Optimized for 3CX G.711 codec"
echo ""
echo "Next: Apply VoiceBot code optimizations"
