#!/usr/bin/env python3
"""
Simplified 3CX Voice Bot - pjsua2 Implementation (Non-async)
"""

import time
import signal
import sys
import logging
import os
import pjsua2 as pj

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for environment variables"""
    
    def __init__(self):
        self.threecx_server = os.getenv('THREECX_SERVER', 'mtipbx.ny.3cx.us')
        self.threecx_port = int(os.getenv('THREECX_PORT', '5060'))
        self.threecx_extension = os.getenv('THREECX_EXTENSION', '1600')
        self.threecx_password = os.getenv('THREECX_PASSWORD', 'FcHw0P2FHK')
        self.threecx_auth_id = os.getenv('THREECX_AUTH_ID', 'qpZh2VS624')
        
        logger.info(f"Configuration loaded - Server: {self.threecx_server}:{self.threecx_port}")
        logger.info(f"Extension: {self.threecx_extension}, Auth ID: {self.threecx_auth_id}")

class PJSIPClient(pj.Account):
    """pjsua2-based SIP client for 3CX integration"""
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config
        self.registered = False
        
    def onRegState(self, prm):
        """Handle registration state changes"""
        info = self.getInfo()
        if info.regIsActive:
            self.registered = True
            logger.info(f"✅ Successfully registered with 3CX: {info.regIsActive}")
        else:
            self.registered = False
            logger.warning(f"❌ Registration failed: {info.regStatusText}")
    
    def onIncomingCall(self, prm):
        """Handle incoming calls"""
        logger.info(f"📞 Incoming call from: {prm.callId}")
        # For now, just answer the call
        call = pj.Call()
        call.answer(200)
        logger.info(f"✅ Call answered: {prm.callId}")

class ThreeCXVoiceBot:
    """Simplified 3CX Voice Bot with pjsua2"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.ep = None
        self.sip_client = None
        self.is_running = False
        
        logger.info("🚀 Initializing 3CX Voice Bot...")
    
    def initialize_sip(self):
        """Initialize SIP components"""
        try:
            logger.info("📡 Initializing pjsua2...")
            
            # Initialize pjsua2
            self.ep = pj.Endpoint()
            self.ep.libCreate()
            self.ep.libInit(pj.EpConfig())
            
            # Create UDP transport
            tcfg = pj.TransportConfig()
            tcfg.port = 0  # Let system choose port
            self.ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, tcfg)
            self.ep.libStart()
            
            logger.info("✅ pjsua2 initialized successfully")
            
            # Create SIP account
            self.create_sip_account()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ SIP initialization failed: {e}")
            return False
    
    def create_sip_account(self):
        """Create and configure SIP account"""
        try:
            logger.info("�� Creating SIP account...")
            
            acfg = pj.AccountConfig()
            acfg.idUri = f"sip:{self.config.threecx_extension}@{self.config.threecx_server}"
            acfg.regConfig.registrarUri = f"sip:{self.config.threecx_server}"
            acfg.regConfig.registerOnAdd = True
            
            # Add authentication credentials
            acfg.sipConfig.authCreds.append(
                pj.AuthCredInfo("digest", "3CXPhoneSystem", 
                               self.config.threecx_auth_id, 0, 
                               self.config.threecx_password)
            )
            
            # Create account
            self.sip_client = PJSIPClient(self.config)
            self.sip_client.create(acfg)
            
            logger.info("✅ SIP account created successfully")
            
        except Exception as e:
            logger.error(f"❌ SIP account creation failed: {e}")
            raise
    
    def wait_for_registration(self, timeout=10):
        """Wait for SIP registration"""
        logger.info("⏳ Waiting for SIP registration...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.sip_client and self.sip_client.registered:
                logger.info("✅ SIP registration confirmed!")
                return True
            time.sleep(0.5)
        
        logger.error("❌ SIP registration timeout")
        return False
    
    def run(self):
        """Main bot execution"""
        try:
            logger.info("�� Starting 3CX Voice Bot...")
            logger.info("=" * 50)
            
            # Initialize SIP
            if not self.initialize_sip():
                logger.error("❌ Failed to initialize SIP")
                return False
            
            # Wait for registration
            if not self.wait_for_registration():
                logger.error("❌ SIP registration failed")
                return False
            
            # Keep running
            logger.info("🎯 Bot is ready! Waiting for incoming calls...")
            logger.info("Press Ctrl+C to stop")
            
            self.is_running = True
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("�� Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("�� Shutting down voice bot...")
        self.is_running = False
        
        if self.ep:
            try:
                self.ep.hangupAllCalls()
                self.ep.libDestroy()
                logger.info("✅ Clean shutdown completed")
            except Exception as e:
                logger.error(f"❌ Shutdown error: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    logger.info("3CX Production Voice Bot - pjsua2 Simplified Version")
    logger.info("=" * 50)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run bot
    bot = ThreeCXVoiceBot()
    bot.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("�� Application terminated by user")
    except Exception as e:
        logger.error(f"❌ Application crashed: {e}")
        import traceback
        traceback.print_exc()
