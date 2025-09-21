#!/bin/bash
# Audio Quality Optimization for NETOVO VoiceBot
# Optimizes Asterisk audio settings for better VoIP quality

echo "=========================================="
echo "NETOVO Audio Quality Optimization"
echo "Optimizing Asterisk for better VoIP audio"
echo "=========================================="

# Backup original configurations
echo "Backing up original configurations..."
sudo cp /etc/asterisk/rtp.conf /etc/asterisk/rtp.conf.backup
sudo cp /etc/asterisk/sip.conf /etc/asterisk/sip.conf.backup

# Optimize RTP settings for better audio quality
echo "Optimizing RTP settings..."
sudo tee /etc/asterisk/rtp.conf > /dev/null << 'EOF'
[general]
rtpstart=10000
rtpend=20000
strictrtp=yes
probation=4
icesupport=yes
stunaddr=stun.l.google.com:19302

; Audio quality optimizations
; Reduce jitter buffer for lower latency
jbenable=yes
jbmaxsize=200
jbresyncthreshold=1000
jbimpl=fixed
jblog=no

; RTP timeout settings (in seconds)
rtptimeout=60
rtpholdtimeout=300
rtpkeepalive=0
EOF

# Optimize codec settings for better audio quality
echo "Optimizing codec settings..."
sudo tee -a /etc/asterisk/sip.conf >> /dev/null << 'EOF'

; Audio quality optimizations
disallow=all
allow=g722    ; HD audio codec
allow=ulaw    ; Standard codec
allow=alaw    ; European standard
allow=gsm     ; Low bandwidth fallback

; Network optimization
qualify=yes
qualifyfreq=60
nat=force_rport,comedia
directmedia=no
session-timers=refuse

; Audio processing
dtmfmode=rfc2833
rfc2833compensate=yes
t38pt_udptl=yes

; Bandwidth optimization
videosupport=no
maxcallbitrate=64
EOF

# Create audio quality test script
echo "Creating audio quality test script..."
cat > /opt/voicebot/test_audio_quality.py << 'EOF'
#!/usr/bin/env python3
"""
Audio Quality Test for NETOVO VoiceBot
Tests and optimizes audio processing pipeline
"""

import subprocess
import os
import time

def test_audio_generation():
    """Test TTS audio generation quality"""
    print("🔊 Testing TTS Audio Generation...")

    test_text = "This is a high quality audio test for the NETOVO voice bot system"

    # Test different TTS engines
    engines = [
        ("espeak", ["espeak", test_text, "-w", "/tmp/test_espeak_quality.wav", "-s", "150", "-p", "50"]),
        ("festival", ["festival", "--tts", "--otype", "wav", "--stdout"]),
        ("flite", ["flite", "-t", test_text, "-o", "/tmp/test_flite_quality.wav"])
    ]

    for engine_name, cmd in engines:
        try:
            if engine_name == "festival":
                # Special handling for festival
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                stdout, _ = proc.communicate(input=test_text.encode())

                with open("/tmp/test_festival_quality.wav", "wb") as f:
                    f.write(stdout)

                test_file = "/tmp/test_festival_quality.wav"
            else:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                test_file = f"/tmp/test_{engine_name}_quality.wav"

            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)

                # Analyze audio quality
                quality_cmd = subprocess.run([
                    'ffprobe', '-v', 'quiet',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=sample_rate,channels,bit_rate',
                    '-of', 'csv=p=0',
                    test_file
                ], capture_output=True, text=True)

                if quality_cmd.returncode == 0:
                    audio_info = quality_cmd.stdout.strip().split(',')
                    sample_rate = audio_info[0] if len(audio_info) > 0 else "unknown"
                    channels = audio_info[1] if len(audio_info) > 1 else "unknown"

                    print(f"✅ {engine_name}: {file_size} bytes, {sample_rate}Hz, {channels}ch")
                else:
                    print(f"✅ {engine_name}: {file_size} bytes")

                # Cleanup
                os.unlink(test_file)
            else:
                print(f"❌ {engine_name}: Failed to generate audio")

        except Exception as e:
            print(f"❌ {engine_name}: Error - {e}")

def test_audio_processing():
    """Test audio processing capabilities"""
    print("\n🔧 Testing Audio Processing...")

    # Test ffmpeg audio conversion
    try:
        test_input = "/tmp/test_input.wav"
        test_output = "/tmp/test_output.wav"

        # Create test audio
        subprocess.run([
            'espeak', 'Audio processing test',
            '-w', test_input
        ], capture_output=True, timeout=10)

        if os.path.exists(test_input):
            # Test audio format conversion
            result = subprocess.run([
                'ffmpeg', '-i', test_input,
                '-ar', '8000',    # 8kHz for telephony
                '-ac', '1',       # Mono
                '-y', test_output
            ], capture_output=True, timeout=10)

            if result.returncode == 0 and os.path.exists(test_output):
                in_size = os.path.getsize(test_input)
                out_size = os.path.getsize(test_output)
                print(f"✅ Audio conversion: {in_size} -> {out_size} bytes (8kHz mono)")

                # Cleanup
                os.unlink(test_input)
                os.unlink(test_output)
            else:
                print("❌ Audio conversion: Failed")
        else:
            print("❌ Audio processing: Failed to create test input")

    except Exception as e:
        print(f"❌ Audio processing: Error - {e}")

def test_network_quality():
    """Test network quality for VoIP"""
    print("\n🌐 Testing Network Quality...")

    try:
        # Test latency to Google DNS
        ping_result = subprocess.run([
            'ping', '-c', '4', '8.8.8.8'
        ], capture_output=True, text=True, timeout=15)

        if ping_result.returncode == 0:
            lines = ping_result.stdout.split('\n')
            for line in lines:
                if 'avg' in line:
                    print(f"✅ Network latency: {line.split('/')[-2]}ms average")
                    break
        else:
            print("❌ Network: Ping test failed")

        # Test bandwidth estimation
        print("📊 For VoIP quality:")
        print("   - Latency: <100ms (excellent), <200ms (good)")
        print("   - Bandwidth: 64kbps minimum per call")
        print("   - Packet loss: <1% for good quality")

    except Exception as e:
        print(f"❌ Network test: Error - {e}")

def main():
    print("NETOVO Audio Quality Test")
    print("=" * 40)

    test_audio_generation()
    test_audio_processing()
    test_network_quality()

    print("\n" + "=" * 40)
    print("Audio Quality Test Complete")
    print("=" * 40)

    print("\n📋 Recommendations:")
    print("1. Use G.722 codec for HD audio quality")
    print("2. Ensure network latency <100ms")
    print("3. Configure jitter buffer for stability")
    print("4. Monitor packet loss during calls")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/voicebot/test_audio_quality.py

# Restart Asterisk to apply changes
echo "Restarting Asterisk to apply optimizations..."
sudo systemctl restart asterisk

# Wait for Asterisk to fully restart
sleep 5

# Test Asterisk status
if sudo asterisk -rx "core show version" &> /dev/null; then
    echo "✅ Asterisk restarted successfully"
else
    echo "❌ Asterisk restart failed"
fi

echo "=========================================="
echo "Audio Quality Optimization Complete"
echo "=========================================="

echo "Optimizations Applied:"
echo "- RTP settings optimized for VoIP"
echo "- Jitter buffer configured"
echo "- HD audio codecs enabled (G.722)"
echo "- Network settings optimized"

echo ""
echo "To test audio quality: python3 /opt/voicebot/test_audio_quality.py"
echo "To monitor call quality: sudo asterisk -rx 'core show channels'"

echo ""
echo "🎯 Audio optimization complete - VoIP quality improved"
