#!/usr/bin/env python3
"""
Simple pjsua2 initialization test
"""

import logging
import pjsua2 as pj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pjsua2_init():
    """Test different pjsua2 initialization methods"""
    logger.info("Testing pjsua2 initialization methods...")
    
    # Method 1: Try EpConfig
    try:
        logger.info("Method 1: Trying EpConfig...")
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        
        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = 4
        ep_cfg.logConfig.consoleLevel = 4
        endpoint.libInit(ep_cfg)
        endpoint.libStart()
        
        logger.info("Method 1 SUCCESS: EpConfig worked")
        
        # Test device enumeration
        aud_dev_manager = pj.Endpoint.instance().audDevManager()
        dev_count = aud_dev_manager.getDevCount()
        logger.info("Found {} audio devices".format(dev_count))
        
        endpoint.libDestroy()
        return True
        
    except Exception as e:
        logger.error("Method 1 FAILED: {}".format(e))
    
    # Method 2: Try separate configs
    try:
        logger.info("Method 2: Trying separate configs...")
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        
        ua_cfg = pj.UAConfig()
        log_cfg = pj.LogConfig()
        log_cfg.level = 4
        log_cfg.consoleLevel = 4
        media_cfg = pj.MediaConfig()
        endpoint.libInit(ua_cfg, log_cfg, media_cfg)
        endpoint.libStart()
        
        logger.info("Method 2 SUCCESS: Separate configs worked")
        
        # Test device enumeration
        aud_dev_manager = pj.Endpoint.instance().audDevManager()
        dev_count = aud_dev_manager.getDevCount()
        logger.info("Found {} audio devices".format(dev_count))
        
        endpoint.libDestroy()
        return True
        
    except Exception as e:
        logger.error("Method 2 FAILED: {}".format(e))
    
    # Method 3: Try minimal EpConfig
    try:
        logger.info("Method 3: Trying minimal EpConfig...")
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        
        ep_cfg = pj.EpConfig()
        endpoint.libInit(ep_cfg)
        endpoint.libStart()
        
        logger.info("Method 3 SUCCESS: Minimal EpConfig worked")
        
        # Test device enumeration
        aud_dev_manager = pj.Endpoint.instance().audDevManager()
        dev_count = aud_dev_manager.getDevCount()
        logger.info("Found {} audio devices".format(dev_count))
        
        endpoint.libDestroy()
        return True
        
    except Exception as e:
        logger.error("Method 3 FAILED: {}".format(e))
    
    # Method 4: Try no config (oldest API)
    try:
        logger.info("Method 4: Trying no config...")
        endpoint = pj.Endpoint()
        endpoint.libCreate()
        endpoint.libInit()
        endpoint.libStart()
        
        logger.info("Method 4 SUCCESS: No config worked")
        
        # Test device enumeration
        aud_dev_manager = pj.Endpoint.instance().audDevManager()
        dev_count = aud_dev_manager.getDevCount()
        logger.info("Found {} audio devices".format(dev_count))
        
        endpoint.libDestroy()
        return True
        
    except Exception as e:
        logger.error("Method 4 FAILED: {}".format(e))
    
    logger.error("All pjsua2 initialization methods failed!")
    return False

if __name__ == "__main__":
    success = test_pjsua2_init()
    if success:
        print("pjsua2 initialization test PASSED")
    else:
        print("pjsua2 initialization test FAILED")
