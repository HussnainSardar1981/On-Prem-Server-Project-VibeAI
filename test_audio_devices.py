#!/usr/bin/env python3
"""
Test script to debug ALSA Loopback device issues
"""

import logging
import pyaudio
import pjsua2 as pj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pyaudio_devices():
    """Test PyAudio device enumeration"""
    logger.info("=== PyAudio Device Test ===")
    
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()
    
    logger.info(f"PyAudio found {device_count} devices")
    
    loopback_devices = []
    for i in range(device_count):
        try:
            info = pa.get_device_info_by_index(i)
            if 'loopback' in info['name'].lower():
                loopback_devices.append((i, info))
                logger.info(f"Loopback Device {i}: {info['name']}")
                logger.info(f"  - Inputs: {info['maxInputChannels']}")
                logger.info(f"  - Outputs: {info['maxOutputChannels']}")
                logger.info(f"  - Default Rate: {info['defaultSampleRate']}")
        except Exception as e:
            logger.error(f"Error getting info for device {i}: {e}")
    
    pa.terminate()
    return loopback_devices

def test_pjsua2_devices():
    """Test pjsua2 device enumeration"""
    logger.info("=== pjsua2 Device Test ===")
    
    try:
        # Initialize pjsua2
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        
        # Basic initialization with proper configuration
        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = 4  # INFO level
        ep_cfg.logConfig.consoleLevel = 4
        endpoint.libInit(ep_cfg)
        endpoint.libStart()
        
        # Get audio device manager
        aud_dev_manager = pj.Endpoint.instance().audDevManager()
        dev_count = aud_dev_manager.getDevCount()
        
        logger.info(f"pjsua2 found {dev_count} devices")
        
        loopback_devices = []
        for i in range(dev_count):
            try:
                dev_info = aud_dev_manager.getDevInfo(i)
                if 'loopback' in dev_info.name.lower():
                    loopback_devices.append((i, dev_info))
                    logger.info(f"pjsua2 Loopback Device {i}: {dev_info.name}")
                    logger.info(f"  - Input channels: {dev_info.inputCount}")
                    logger.info(f"  - Output channels: {dev_info.outputCount}")
            except Exception as e:
                logger.error(f"Error getting pjsua2 info for device {i}: {e}")
        
        # Cleanup
        endpoint.libDestroy()
        return loopback_devices
        
    except Exception as e:
        logger.error(f"pjsua2 test failed: {e}")
        return []

def test_alsa_direct():
    """Test ALSA devices directly"""
    logger.info("=== Direct ALSA Test ===")
    
    import subprocess
    import os
    
    try:
        # Check if loopback module is loaded
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        if 'snd_aloop' in result.stdout:
            logger.info("✓ snd_aloop module is loaded")
        else:
            logger.error("✗ snd_aloop module is NOT loaded")
            logger.info("Load it with: sudo modprobe snd-aloop")
        
        # Check PCM devices
        if os.path.exists('/proc/asound/pcm'):
            with open('/proc/asound/pcm', 'r') as f:
                pcm_info = f.read()
                logger.info("PCM devices:")
                for line in pcm_info.strip().split('\n'):
                    if 'Loopback' in line:
                        logger.info(f"  {line}")
        
        # Check card info
        if os.path.exists('/proc/asound/cards'):
            with open('/proc/asound/cards', 'r') as f:
                cards_info = f.read()
                logger.info("Sound cards:")
                for line in cards_info.strip().split('\n'):
                    if 'Loopback' in line or line.strip().startswith('0'):
                        logger.info(f"  {line}")
        
    except Exception as e:
        logger.error(f"Direct ALSA test failed: {e}")

def main():
    """Run all tests"""
    logger.info("Starting ALSA Loopback device tests...")
    
    # Test direct ALSA
    test_alsa_direct()
    
    # Test PyAudio
    pyaudio_loopback = test_pyaudio_devices()
    
    # Test pjsua2
    pjsua2_loopback = test_pjsua2_devices()
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"PyAudio found {len(pyaudio_loopback)} loopback devices")
    logger.info(f"pjsua2 found {len(pjsua2_loopback)} loopback devices")
    
    if len(pyaudio_loopback) >= 1 and len(pjsua2_loopback) >= 1:
        logger.info("✓ Both PyAudio and pjsua2 can see loopback devices")
        logger.info("The issue is likely in the device selection logic")
    elif len(pyaudio_loopback) >= 1:
        logger.info("✓ PyAudio can see loopback devices")
        logger.info("✗ pjsua2 cannot see loopback devices - this is the problem")
    elif len(pjsua2_loopback) >= 1:
        logger.info("✗ PyAudio cannot see loopback devices")
        logger.info("✓ pjsua2 can see loopback devices")
    else:
        logger.info("✗ Neither PyAudio nor pjsua2 can see loopback devices")
        logger.info("Check ALSA Loopback module loading and configuration")

if __name__ == "__main__":
    main()
