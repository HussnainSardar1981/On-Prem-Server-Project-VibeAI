#!/usr/bin/env python3
"""
Simple AGI Recording Test Script
Tests if basic AGI recording commands work
"""

import os
import sys
import logging
import tempfile
import time

# Check AGI environment
if sys.stdin.isatty():
    print("ERROR: Must be called from Asterisk AGI", file=sys.stderr)
    sys.exit(0)

try:
    from asterisk.agi import AGI
except ImportError:
    print("ERROR: pyst2 not found", file=sys.stderr)
    sys.exit(0)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('/var/log/asterisk/agi_test.log'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('AGI_Test')

def test_agi_commands():
    """Test basic AGI recording commands"""
    try:
        agi = AGI()
        logger.info("AGI initialized successfully")

        # Answer call
        agi.answer()
        logger.info("Call answered")

        # Test 1: Say Alpha (we know this works)
        logger.info("Test 1: Say Alpha")
        agi.say_alpha("HELLO")

        # Test 2: Stream File (play beep)
        logger.info("Test 2: Stream File")
        try:
            result = agi.stream_file('beep', '')
            logger.info(f"Stream file result: {result}")
        except Exception as e:
            logger.error(f"Stream file failed: {e}")

        # Test 3: Record File (the problematic one)
        logger.info("Test 3: Record File")
        try:
            record_name = f"/tmp/agi_test_{int(time.time())}"

            # Simple record command
            result = agi.record_file(
                record_name,
                format='wav',
                escape_digits='#',
                timeout=5000,  # 5 seconds
                offset=0,
                beep=1,
                silence=2
            )

            logger.info(f"Record file result: {result}")

            # Check if file was created
            wav_file = record_name + '.wav'
            if os.path.exists(wav_file):
                size = os.path.getsize(wav_file)
                logger.info(f"Recording created: {size} bytes")
                # Clean up
                os.unlink(wav_file)
            else:
                logger.error("No recording file created")

        except Exception as e:
            logger.error(f"Record file failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Test 4: Alternative record method
        logger.info("Test 4: Alternative Record Method")
        try:
            agi.say_alpha("RECORDING")

            # Use execute instead of record_file
            record_name2 = f"/tmp/agi_test2_{int(time.time())}"
            result = agi.execute('Record', f"{record_name2}.wav,5,#")
            logger.info(f"Execute Record result: {result}")

            # Check file
            if os.path.exists(f"{record_name2}.wav"):
                size = os.path.getsize(f"{record_name2}.wav")
                logger.info(f"Execute recording created: {size} bytes")
                os.unlink(f"{record_name2}.wav")

        except Exception as e:
            logger.error(f"Execute Record failed: {e}")

        # Final message
        agi.say_alpha("TEST COMPLETE")
        agi.hangup()

        logger.info("All tests completed")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    logger.info("Starting AGI recording test")
    success = test_agi_commands()
    logger.info(f"Test result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0)

if __name__ == "__main__":
    main()
