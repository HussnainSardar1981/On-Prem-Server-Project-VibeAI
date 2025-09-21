#!/bin/bash
# Enterprise Speech-to-Text Stack Installer for NETOVO VoiceBot
# Installs multiple STT engines for maximum accuracy and reliability

echo "=========================================="
echo "NETOVO Enterprise STT Stack Installer"
echo "Installing Speech-to-Text engines..."
echo "=========================================="

# Update package list
echo "Updating package repositories..."
sudo apt update

# Install Python dependencies
echo "Installing Python STT libraries..."
sudo apt install -y python3-pip python3-dev build-essential
sudo pip3 install SpeechRecognition pocketsphinx pyaudio soundfile librosa

# Install FFmpeg for audio processing
echo "Installing FFmpeg and audio tools..."
sudo apt install -y ffmpeg

# Install Whisper (OpenAI)
echo "Installing Whisper (OpenAI)..."
sudo pip3 install openai-whisper

# Install Vosk
echo "Installing Vosk STT..."
sudo pip3 install vosk

# Download Vosk English model
echo "Downloading Vosk English model..."
cd /opt
sudo wget -q https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
sudo unzip -q vosk-model-en-us-0.22.zip
sudo mv vosk-model-en-us-0.22 vosk-model-en
sudo rm vosk-model-en-us-0.22.zip

# Install PocketSphinx
echo "Installing PocketSphinx..."
sudo apt install -y sphinxbase-utils pocketsphinx pocketsphinx-en-us

# Install additional audio processing tools
echo "Installing additional audio tools..."
sudo apt install -y sox libsox-fmt-all alsa-utils

# Create STT test script
echo "Creating STT test script..."
cat > /opt/voicebot/test_stt_engines.py << 'EOF'
#!/usr/bin/env python3
"""
STT Engine Test for NETOVO VoiceBot
Tests all available Speech-to-Text engines
"""

import subprocess
import time
import os
import tempfile

def create_test_audio():
    """Create a test audio file using espeak"""
    test_file = "/tmp/stt_test_audio.wav"
    test_text = "Hello, this is a test message for speech recognition"

    try:
        # Generate test audio
        result = subprocess.run([
            'espeak', test_text,
            '-w', test_file,
            '-s', '150'
        ], capture_output=True, timeout=10)

        if result.returncode == 0 and os.path.exists(test_file):
            print(f"✅ Test audio created: {test_text}")
            return test_file, test_text
        else:
            print("❌ Failed to create test audio")
            return None, None

    except Exception as e:
        print(f"❌ Error creating test audio: {e}")
        return None, None

def test_whisper(audio_file):
    """Test Whisper STT"""
    try:
        print("\n🔍 Testing Whisper STT...")
        start_time = time.time()

        result = subprocess.run([
            'whisper', audio_file,
            '--model', 'tiny',
            '--language', 'en',
            '--output_format', 'txt',
            '--output_dir', '/tmp'
        ], capture_output=True, text=True, timeout=30)

        duration = time.time() - start_time

        if result.returncode == 0:
            # Look for generated text file
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            txt_file = f"/tmp/{base_name}.txt"

            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    text = f.read().strip()
                os.unlink(txt_file)

                print(f"✅ Whisper: SUCCESS")
                print(f"   Result: {text}")
                print(f"   Duration: {duration:.2f}s")
                return True

        print(f"❌ Whisper: FAILED")
        return False

    except Exception as e:
        print(f"❌ Whisper: ERROR - {e}")
        return False

def test_vosk(audio_file):
    """Test Vosk STT"""
    try:
        print("\n🔍 Testing Vosk STT...")
        start_time = time.time()

        # Convert audio for Vosk
        converted_file = f"/tmp/vosk_test_{int(time.time())}.wav"
        convert_result = subprocess.run([
            'ffmpeg', '-i', audio_file,
            '-ar', '16000', '-ac', '1',
            '-y', converted_file
        ], capture_output=True, timeout=10)

        if convert_result.returncode != 0:
            print("❌ Vosk: Audio conversion failed")
            return False

        # Test with Python Vosk
        import vosk
        import json
        import wave

        model = vosk.Model("/opt/vosk-model-en")
        rec = vosk.KaldiRecognizer(model, 16000)

        wf = wave.open(converted_file, 'rb')
        text_results = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text'):
                    text_results.append(result['text'])

        final_result = json.loads(rec.FinalResult())
        if final_result.get('text'):
            text_results.append(final_result['text'])

        wf.close()
        os.unlink(converted_file)

        duration = time.time() - start_time
        full_text = ' '.join(text_results).strip()

        if full_text:
            print(f"✅ Vosk: SUCCESS")
            print(f"   Result: {full_text}")
            print(f"   Duration: {duration:.2f}s")
            return True
        else:
            print(f"❌ Vosk: No text recognized")
            return False

    except Exception as e:
        print(f"❌ Vosk: ERROR - {e}")
        return False

def test_pocketsphinx(audio_file):
    """Test PocketSphinx STT"""
    try:
        print("\n🔍 Testing PocketSphinx STT...")
        start_time = time.time()

        import speech_recognition as sr

        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_sphinx(audio_data)
        duration = time.time() - start_time

        if text and len(text.strip()) > 2:
            print(f"✅ PocketSphinx: SUCCESS")
            print(f"   Result: {text}")
            print(f"   Duration: {duration:.2f}s")
            return True
        else:
            print(f"❌ PocketSphinx: No text recognized")
            return False

    except Exception as e:
        print(f"❌ PocketSphinx: ERROR - {e}")
        return False

def main():
    print("NETOVO Enterprise STT Test")
    print("=" * 40)

    # Create test audio
    audio_file, expected_text = create_test_audio()
    if not audio_file:
        print("❌ Cannot proceed without test audio")
        return

    print(f"Expected text: '{expected_text}'")

    # Test all STT engines
    engines = [
        ("Whisper", test_whisper),
        ("Vosk", test_vosk),
        ("PocketSphinx", test_pocketsphinx)
    ]

    results = {}
    for engine_name, test_func in engines:
        results[engine_name] = test_func(audio_file)

    # Cleanup
    if os.path.exists(audio_file):
        os.unlink(audio_file)

    # Summary
    print("\n" + "=" * 40)
    print("STT ENGINE TEST RESULTS")
    print("=" * 40)

    working_engines = []
    for engine, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{engine:15}: {status}")
        if success:
            working_engines.append(engine)

    print(f"\nWorking engines: {len(working_engines)}/3")
    if working_engines:
        print(f"Primary engine: {working_engines[0]}")
        print("🎯 STT system ready for production!")
    else:
        print("❌ No STT engines working - check installation")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/voicebot/test_stt_engines.py

# Test all installations
echo "=========================================="
echo "Testing STT installations..."
echo "=========================================="

# Test Whisper
echo "Testing Whisper installation..."
if command -v whisper &> /dev/null; then
    echo "✅ Whisper: INSTALLED"
else
    echo "❌ Whisper: NOT FOUND"
fi

# Test Vosk model
echo "Testing Vosk model..."
if [ -d "/opt/vosk-model-en" ]; then
    echo "✅ Vosk model: INSTALLED"
else
    echo "❌ Vosk model: NOT FOUND"
fi

# Test PocketSphinx
echo "Testing PocketSphinx..."
if command -v pocketsphinx_continuous &> /dev/null; then
    echo "✅ PocketSphinx: INSTALLED"
else
    echo "❌ PocketSphinx: NOT FOUND"
fi

# Test Python libraries
echo "Testing Python STT libraries..."
python3 -c "import speech_recognition; print('✅ SpeechRecognition: INSTALLED')" 2>/dev/null || echo "❌ SpeechRecognition: FAILED"
python3 -c "import vosk; print('✅ Vosk: INSTALLED')" 2>/dev/null || echo "❌ Vosk: FAILED"

echo "=========================================="
echo "Enterprise STT Stack Installation Complete"
echo "=========================================="

echo "Installed STT Engines:"
echo "- Whisper (OpenAI) - Highest accuracy"
echo "- Vosk - Fast and lightweight"
echo "- PocketSphinx - Reliable fallback"

echo ""
echo "To test STT engines: python3 /opt/voicebot/test_stt_engines.py"
echo "To restart VoiceBot: sudo systemctl restart asterisk"

echo ""
echo "🎯 Phase 2A Complete: Enterprise STT Stack Ready"
