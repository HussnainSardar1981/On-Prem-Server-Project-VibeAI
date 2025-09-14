#!/usr/bin/env python3
"""
Diagnostic script for pjsua2 API compatibility
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def diagnose_pjsua2():
    """Diagnose pjsua2 installation and API"""
    logger.info("=== pjsua2 Diagnostic ===")
    
    try:
        import pjsua2 as pj
        logger.info("✓ pjsua2 imported successfully")
        
        # Check version
        try:
            version = pj.version()
            logger.info(f"pjsua2 version: {version}")
        except:
            logger.info("pjsua2 version: unknown")
        
        # Check available classes
        logger.info("Available pjsua2 classes:")
        classes = [attr for attr in dir(pj) if attr[0].isupper()]
        for cls in sorted(classes):
            logger.info(f"  - {cls}")
        
        # Test basic functionality
        logger.info("\nTesting basic functionality...")
        
        try:
            endpoint = pj.Endpoint()
            logger.info("✓ Endpoint created")
            
            endpoint.libCreate()
            logger.info("✓ libCreate() successful")
            
            # Test different initialization methods
            logger.info("\nTesting initialization methods...")
            
            # Method 1: With EpConfig (newer API)
            try:
                ep_cfg = pj.EpConfig()
                endpoint.libInit(ep_cfg)
                logger.info("✓ libInit(EpConfig) - SUCCESS")
                endpoint.libDestroy()
                return True
            except Exception as e:
                logger.info(f"✗ libInit(EpConfig) failed: {e}")
            
            # Method 2: With all configs
            try:
                endpoint = pj.Endpoint()
                endpoint.libCreate()
                ua_cfg = pj.UAConfig()
                log_cfg = pj.LogConfig()
                media_cfg = pj.MediaConfig()
                ep_cfg = pj.EpConfig()
                ep_cfg.uaConfig = ua_cfg
                ep_cfg.logConfig = log_cfg
                ep_cfg.mediaConfig = media_cfg
                endpoint.libInit(ep_cfg)
                logger.info("✓ libInit(EpConfig with all configs) - SUCCESS")
                endpoint.libDestroy()
                return True
            except Exception as e:
                logger.info(f"✗ libInit(EpConfig with all configs) failed: {e}")
            
            # Method 3: No parameters (older API)
            try:
                endpoint = pj.Endpoint()
                endpoint.libCreate()
                endpoint.libInit()
                logger.info("✓ libInit() with no parameters - SUCCESS")
                endpoint.libDestroy()
                return True
            except Exception as e:
                logger.info(f"✗ libInit() with no parameters failed: {e}")
            
            # Method 4: Check if configs exist
            logger.info("\nChecking configuration classes...")
            configs = ['EpConfig', 'UAConfig', 'LogConfig', 'MediaConfig', 'AccountConfig', 'TransportConfig']
            for config in configs:
                if hasattr(pj, config):
                    logger.info(f"✓ {config} available")
                else:
                    logger.info(f"✗ {config} not available")
            
            return False
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import pjsua2: {e}")
        logger.error("Install pjsua2: pip install pjsua2")
        return False

def diagnose_pyaudio():
    """Diagnose PyAudio installation and API"""
    logger.info("\n=== PyAudio Diagnostic ===")
    
    try:
        import pyaudio
        logger.info("✓ PyAudio imported successfully")
        
        # Check version
        try:
            version = pyaudio.__version__
            logger.info(f"PyAudio version: {version}")
        except:
            logger.info("PyAudio version: unknown")
        
        # Test API methods
        p = pyaudio.PyAudio()
        
        logger.info("Testing PyAudio API methods...")
        
        # Test is_format_supported
        try:
            result = p.is_format_supported(
                rate=8000,
                input_device=None,
                output_device=None,
                input_channels=1,
                output_channels=1,
                format=pyaudio.paInt16
            )
            logger.info("✓ is_format_supported() with all parameters - SUCCESS")
        except TypeError as e:
            logger.info(f"✗ is_format_supported() with all parameters failed: {e}")
            
            # Try without format parameter
            try:
                result = p.is_format_supported(
                    rate=8000,
                    input_device=None,
                    output_device=None,
                    input_channels=1,
                    output_channels=1
                )
                logger.info("✓ is_format_supported() without format parameter - SUCCESS")
            except Exception as e2:
                logger.info(f"✗ is_format_supported() without format parameter failed: {e2}")
        
        # Test stream opening
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=8000,
                input=True,
                frames_per_buffer=160
            )
            stream.close()
            logger.info("✓ Stream opening - SUCCESS")
        except Exception as e:
            logger.info(f"✗ Stream opening failed: {e}")
        
        p.terminate()
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import PyAudio: {e}")
        return False

def main():
    """Run diagnostics"""
    logger.info("Running pjsua2 and PyAudio diagnostics...")
    
    pjsua2_ok = diagnose_pjsua2()
    pyaudio_ok = diagnose_pyaudio()
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"pjsua2: {'✓ OK' if pjsua2_ok else '✗ Issues found'}")
    logger.info(f"PyAudio: {'✓ OK' if pyaudio_ok else '✗ Issues found'}")
    
    if pjsua2_ok and pyaudio_ok:
        logger.info("All diagnostics passed!")
        return True
    else:
        logger.info("Some issues found. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
