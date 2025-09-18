#!/usr/bin/env python3
"""
Debug script to test AI components without AGI
"""

import sys
import os
import logging
import tempfile

# Add your voicebot path
sys.path.append('/opt/voicebot')

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('DEBUG_TEST')

def test_ai_models():
    """Test AI model loading"""
    try:
        from production_asterisk_voicebot import AIProcessor, ProductionConfig
        
        logger.info("Testing AI model initialization...")
        
        config = ProductionConfig()
        ai_processor = AIProcessor(config)
        
        # Test model loading
        success = ai_processor.initialize_models()
        if success:
            logger.info("✓ AI models loaded successfully")
        else:
            logger.error("✗ AI model loading failed")
            return False
            
        # Test TTS
        logger.info("Testing TTS...")
        audio_file = ai_processor.text_to_speech("Hello, this is a test message.")
        if audio_file and os.path.exists(audio_file):
            logger.info(f"✓ TTS working, created: {audio_file}")
            os.unlink(audio_file)  # Clean up
        else:
            logger.error("✗ TTS failed")
            return False
            
        # Test STT with dummy audio
        logger.info("Testing STT...")
        # Create a dummy audio file for testing
        import numpy as np
        import soundfile as sf
        
        dummy_audio = np.random.randn(16000)  # 1 second of noise
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_audio.name, dummy_audio, 16000)
        
        text = ai_processor.speech_to_text(temp_audio.name)
        os.unlink(temp_audio.name)
        
        logger.info(f"✓ STT result: {text}")
        
        # Test LLM
        logger.info("Testing LLM...")
        response = ai_processor.generate_response("Hello, can you help me?")
        logger.info(f"✓ LLM response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_permissions():
    """Test file system permissions"""
    try:
        # Test log file creation
        log_path = '/var/log/asterisk/netovo_voicebot.log'
        with open(log_path, 'a') as f:
            f.write(f"Test write at {time.time()}\n")
        logger.info("✓ Log file writable")
        
        # Test temp file creation
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav')
        logger.info(f"✓ Temp file creation OK: {temp_file.name}")
        temp_file.close()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ File permission test failed: {e}")
        return False

def main():
    logger.info("Starting AI VoiceBot debug tests...")
    
    # Test 1: File permissions
    logger.info("\n=== Testing File Permissions ===")
    if not test_file_permissions():
        return False
    
    # Test 2: AI models
    logger.info("\n=== Testing AI Models ===")
    if not test_ai_models():
        return False
        
    logger.info("\n✓ All tests passed! VoiceBot should work in AGI mode.")
    return True

if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)
