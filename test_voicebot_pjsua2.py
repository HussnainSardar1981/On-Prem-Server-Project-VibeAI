#!/usr/bin/env python3
"""
Test script for voicebot_pjsua2.py
Validates environment and components before running the main voice bot
"""

import os
import sys
import logging
import subprocess
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports"""
    logger.info("Testing Python imports...")
    
    required_modules = [
        'numpy', 'torch', 'librosa', 'soundfile', 'pyaudio', 
        'webrtcvad', 'whisper', 'TTS', 'requests', 'pjsua2'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed imports: {failed_imports}")
        return False
    
    logger.info("All imports successful")
    return True

def test_alsa_loopback():
    """Test ALSA Loopback devices"""
    logger.info("Testing ALSA Loopback devices...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        loopback_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if 'loopback' in info['name'].lower():
                loopback_devices.append((i, info))
        
        if len(loopback_devices) < 2:
            logger.error("Insufficient ALSA Loopback devices found")
            logger.error("Run: sudo modprobe snd-aloop")
            return False
        
        logger.info("Found ALSA Loopback devices:")
        for dev_id, info in loopback_devices:
            logger.info(f"  Device {dev_id}: {info['name']} - Inputs: {info['maxInputChannels']}, Outputs: {info['maxOutputChannels']}")
        
        p.terminate()
        return True
        
    except Exception as e:
        logger.error(f"ALSA Loopback test failed: {e}")
        return False

def test_ollama():
    """Test Ollama connection"""
    logger.info("Testing Ollama connection...")
    
    try:
        import requests
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"Ollama connected - {len(models)} models available")
            
            # Check for orca2:7b
            orca_models = [m for m in models if 'orca2' in m.get('name', '')]
            if orca_models:
                logger.info(f"✓ Found ORCA2 models: {[m['name'] for m in orca_models]}")
            else:
                logger.warning("ORCA2 model not found - run: ollama pull orca2:7b")
            
            return True
        else:
            logger.error(f"Ollama connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        logger.error("Start Ollama: ollama serve")
        return False

def test_environment():
    """Test environment configuration"""
    logger.info("Testing environment configuration...")
    
    required_vars = [
        'THREECX_SERVER', 'THREECX_EXTENSION', 'THREECX_AUTH_ID', 'THREECX_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        logger.error("Create .env file with required configuration")
        return False
    
    logger.info("Environment configuration OK")
    return True

def test_audio_format():
    """Test audio format support"""
    logger.info("Testing audio format support...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Test by trying to open streams (simpler approach)
        logger.info("Testing 8kHz mono S16LE format by opening streams...")
        
        input_success = False
        output_success = False
        
        # Test input stream
        try:
            input_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=8000,
                input=True,
                frames_per_buffer=160
            )
            input_stream.close()
            logger.info("✓ Input stream test passed")
            input_success = True
        except Exception as e:
            logger.warning(f"Input stream test failed: {e}")
        
        # Test output stream
        try:
            output_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=8000,
                output=True,
                frames_per_buffer=160
            )
            output_stream.close()
            logger.info("✓ Output stream test passed")
            output_success = True
        except Exception as e:
            logger.warning(f"Output stream test failed: {e}")
        
        p.terminate()
        
        # Both input and output must work for the test to pass
        if input_success and output_success:
            logger.info("✓ 8kHz mono S16LE format supported")
            return True
        else:
            logger.error("✗ Audio format test failed - not all streams could be opened")
            return False
        
    except Exception as e:
        logger.error(f"Audio format test failed: {e}")
        return False

def test_pjsua2():
    """Test pjsua2 functionality"""
    logger.info("Testing pjsua2 functionality...")
    
    try:
        import pjsua2 as pj
        
        # Test basic pjsua2 operations
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        
        # Test configuration - use correct API
        try:
            # Try newer pjsua2 API with EpConfig
            ep_cfg = pj.EpConfig()
            endpoint.libInit(ep_cfg)
            endpoint.libDestroy()
            
        except AttributeError:
            try:
                # Fallback: try with individual configs
                ua_cfg = pj.UAConfig()
                log_cfg = pj.LogConfig()
                media_cfg = pj.MediaConfig()
                endpoint.libInit(ua_cfg, log_cfg, media_cfg)
                endpoint.libDestroy()
                
            except AttributeError:
                # Final fallback for older pjsua2 versions
                try:
                    endpoint.libInit()
                    endpoint.libDestroy()
                except Exception as e2:
                    logger.error(f"pjsua2 libInit failed: {e2}")
                    return False
        
        logger.info("✓ pjsua2 basic functionality OK")
        return True
        
    except Exception as e:
        logger.error(f"pjsua2 test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=== Voice Bot Environment Test ===")
    
    tests = [
        ("Python Imports", test_imports),
        ("ALSA Loopback", test_alsa_loopback),
        ("Ollama Connection", test_ollama),
        ("Environment Config", test_environment),
        ("Audio Format", test_audio_format),
        ("pjsua2 Functionality", test_pjsua2),
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    passed_tests = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            passed_tests.append(test_name)
            logger.info(f"✓ {test_name} PASSED")
        else:
            failed_tests.append(test_name)
            logger.error(f"✗ {test_name} FAILED")
    
    # Show detailed results
    logger.info(f"\n=== Test Results: {passed}/{total} PASSED ===")
    
    if passed_tests:
        logger.info("\n✅ PASSED TESTS:")
        for test in passed_tests:
            logger.info(f"  ✓ {test}")
    
    if failed_tests:
        logger.error("\n❌ FAILED TESTS:")
        for test in failed_tests:
            logger.error(f"  ✗ {test}")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Voice bot should work correctly.")
        return True
    else:
        logger.error(f"\n⚠️  {len(failed_tests)} test(s) failed. Please fix the issues before running the voice bot.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
