#!/bin/bash
# Enterprise TTS Stack Installer for NETOVO VoiceBot
# Installs and configures all open-source TTS engines for maximum reliability

echo "=========================================="
echo "NETOVO Enterprise TTS Stack Installer"
echo "Installing open-source TTS engines..."
echo "=========================================="

# Update package list
echo "Updating package repositories..."
sudo apt update

# Install espeak-ng (enhanced espeak)
echo "Installing espeak-ng (Enhanced espeak)..."
sudo apt install -y espeak-ng espeak-ng-data

# Install Festival with voices
echo "Installing Festival TTS with voice packs..."
sudo apt install -y festival festvox-kallpc16k festvox-kdlpc16k festvox-rablpc16k

# Install Flite (lightweight TTS)
echo "Installing Flite TTS..."
sudo apt install -y flite flite1-dev

# Install additional audio tools
echo "Installing audio processing tools..."
sudo apt install -y sox libsox-fmt-all alsa-utils

# Install Python TTS libraries (optional, for future use)
echo "Installing Python TTS libraries..."
sudo pip3 install gtts pydub

# Test all TTS engines
echo "=========================================="
echo "Testing TTS engines..."
echo "=========================================="

# Test espeak
echo "Testing espeak..."
if command -v espeak &> /dev/null; then
    espeak "espeak is working" -w /tmp/test_espeak.wav
    if [ -f /tmp/test_espeak.wav ]; then
        echo "✅ espeak: WORKING"
        rm /tmp/test_espeak.wav
    else
        echo "❌ espeak: FAILED"
    fi
else
    echo "❌ espeak: NOT FOUND"
fi

# Test espeak-ng
echo "Testing espeak-ng..."
if command -v espeak-ng &> /dev/null; then
    espeak-ng "espeak-ng is working" -w /tmp/test_espeak_ng.wav
    if [ -f /tmp/test_espeak_ng.wav ]; then
        echo "✅ espeak-ng: WORKING"
        rm /tmp/test_espeak_ng.wav
    else
        echo "❌ espeak-ng: FAILED"
    fi
else
    echo "❌ espeak-ng: NOT FOUND"
fi

# Test Festival
echo "Testing Festival..."
if command -v festival &> /dev/null; then
    echo "Festival is working" | festival --tts --otype wav --stdout > /tmp/test_festival.wav 2>/dev/null
    if [ -f /tmp/test_festival.wav ] && [ -s /tmp/test_festival.wav ]; then
        echo "✅ Festival: WORKING"
        rm /tmp/test_festival.wav
    else
        echo "❌ Festival: FAILED"
    fi
else
    echo "❌ Festival: NOT FOUND"
fi

# Test Flite
echo "Testing Flite..."
if command -v flite &> /dev/null; then
    flite -t "Flite is working" -o /tmp/test_flite.wav
    if [ -f /tmp/test_flite.wav ]; then
        echo "✅ Flite: WORKING"
        rm /tmp/test_flite.wav
    else
        echo "❌ Flite: FAILED"
    fi
else
    echo "❌ Flite: NOT FOUND"
fi

# Create TTS performance test script
echo "Creating TTS performance test script..."
cat > /opt/voicebot/test_tts_performance.py << 'EOF'
#!/usr/bin/env python3
"""
TTS Performance Test for NETOVO VoiceBot
Tests all available TTS engines and measures performance
"""

import subprocess
import time
import os

def test_tts_engine(engine, cmd, text="Hello, this is a test message"):
    """Test a TTS engine and measure performance"""
    start_time = time.time()
    temp_file = f"/tmp/test_{engine}_{int(time.time())}.wav"

    try:
        if engine == "espeak":
            result = subprocess.run(['espeak', text, '-w', temp_file],
                                  capture_output=True, timeout=10)
        elif engine == "espeak-ng":
            result = subprocess.run(['espeak-ng', text, '-w', temp_file],
                                  capture_output=True, timeout=10)
        elif engine == "festival":
            script = f'(voice_kal_diphone)(set! audio_method \'wav)(set! audio_file "{temp_file}")(tts_text "{text}")'
            script_file = f"{temp_file}.scm"
            with open(script_file, 'w') as f:
                f.write(script)
            result = subprocess.run(['festival', '-b', script_file],
                                  capture_output=True, timeout=15)
            if os.path.exists(script_file):
                os.unlink(script_file)
        elif engine == "flite":
            result = subprocess.run(['flite', '-t', text, '-o', temp_file],
                                  capture_output=True, timeout=10)
        else:
            return False, 0, 0

        duration = time.time() - start_time

        if result.returncode == 0 and os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file)
            os.unlink(temp_file)
            return True, duration, file_size
        else:
            return False, duration, 0

    except Exception as e:
        print(f"Error testing {engine}: {e}")
        return False, 0, 0
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def main():
    print("NETOVO TTS Performance Test")
    print("=" * 40)

    engines = ["espeak", "espeak-ng", "festival", "flite"]
    test_text = "Thank you for calling NETOVO support. How may I help you today?"

    for engine in engines:
        print(f"\nTesting {engine}...")
        success, duration, file_size = test_tts_engine(engine, [], test_text)

        if success:
            print(f"✅ {engine}: SUCCESS")
            print(f"   Duration: {duration:.2f}s")
            print(f"   File size: {file_size} bytes")
            print(f"   Speed: {len(test_text)/duration:.1f} chars/sec")
        else:
            print(f"❌ {engine}: FAILED")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/voicebot/test_tts_performance.py

echo "=========================================="
echo "Enterprise TTS Stack Installation Complete"
echo "=========================================="

echo "Installed TTS Engines:"
echo "- espeak (standard)"
echo "- espeak-ng (enhanced)"
echo "- Festival (high quality)"
echo "- Flite (lightweight)"

echo ""
echo "To test performance: python3 /opt/voicebot/test_tts_performance.py"
echo "To restart VoiceBot: sudo systemctl restart asterisk"

echo ""
echo "🎯 Phase 1 Complete: Enterprise TTS Stack Ready"
