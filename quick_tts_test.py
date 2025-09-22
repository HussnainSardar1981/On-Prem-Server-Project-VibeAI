#!/usr/bin/env python3
"""
Quick TTS Test - Verify neural TTS is working
"""

import torch
import time
import tempfile
import os

def test_tts():
    """Test the fixed TTS model"""
    print("🎙️  Testing Fixed Neural TTS...")

    try:
        from TTS.api import TTS

        # Check GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_name}")

        # Load the reliable model
        print("   Loading tacotron2-DDC model...")
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

        # Test synthesis
        test_text = "Hello, this is the neural TTS engine working on H100 GPU"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            start_time = time.time()

            tts.tts_to_file(
                text=test_text,
                file_path=temp_file.name
            )

            synthesis_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(temp_file.name)

            print(f"   ✅ TTS Synthesis Time: {synthesis_time:.0f}ms")
            print(f"   ✅ Audio File Size: {file_size} bytes")

            # Cleanup
            os.unlink(temp_file.name)

            if synthesis_time < 2000:  # Under 2 seconds is good
                print("   🚀 TTS Performance: EXCELLENT")
                return True
            else:
                print("   ⚠️  TTS Performance: Acceptable but could be optimized")
                return True

    except Exception as e:
        print(f"   ❌ TTS Test Failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Neural TTS Test")
    print("=" * 30)

    if test_tts():
        print("\n✅ TTS IS WORKING! Ready for production deployment")
    else:
        print("\n❌ TTS needs fixing before deployment")
