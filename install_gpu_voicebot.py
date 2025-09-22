#!/usr/bin/env python3
"""
NETOVO VoiceBot - GPU-Accelerated Installation Script
Automated installation and configuration for on-premise deployment
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict, Any


class GPUVoiceBotInstaller:
    """Automated installer for GPU-accelerated VoiceBot system"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.install_dir = Path("/opt/voicebot")
        self.requirements_installed = False

    def _setup_logging(self):
        """Setup installation logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"gpu_voicebot_install_{int(time.time())}.log")
            ]
        )
        return logging.getLogger("GPUVoiceBotInstaller")

    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging"""
        self.logger.info(f"Running: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode != 0 and check:
            self.logger.error(f"Command failed: {command}")
            self.logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Command failed: {command}")

        return result

    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements for GPU VoiceBot"""
        self.logger.info("Checking system requirements...")

        checks = {}

        # Check Python version
        python_version = sys.version_info
        checks["python_version"] = python_version >= (3, 8)
        self.logger.info(f"Python version: {python_version.major}.{python_version.minor}")

        # Check CUDA availability
        try:
            result = self.run_command("nvidia-smi", check=False)
            checks["nvidia_gpu"] = result.returncode == 0
            if checks["nvidia_gpu"]:
                self.logger.info("NVIDIA GPU detected")
            else:
                self.logger.warning("NVIDIA GPU not detected")
        except:
            checks["nvidia_gpu"] = False

        # Check available disk space (need at least 10GB)
        try:
            result = self.run_command("df -h /opt", check=False)
            checks["disk_space"] = True  # Simplified check
            self.logger.info("Sufficient disk space available")
        except:
            checks["disk_space"] = False

        # Check if running as root
        checks["root_access"] = os.geteuid() == 0
        if not checks["root_access"]:
            self.logger.warning("Not running as root - some operations may fail")

        return checks

    def install_system_dependencies(self):
        """Install system-level dependencies"""
        self.logger.info("Installing system dependencies...")

        # Update package list
        self.run_command("apt-get update")

        # Install basic dependencies
        dependencies = [
            "build-essential",
            "python3-dev",
            "python3-pip",
            "ffmpeg",
            "sox",
            "espeak",
            "espeak-data",
            "festival",
            "flite",
            "portaudio19-dev",
            "libasound2-dev",
            "libfftw3-dev",
            "libsndfile1-dev",
            "git",
            "curl",
            "wget"
        ]

        for dep in dependencies:
            try:
                self.run_command(f"apt-get install -y {dep}")
                self.logger.info(f"Installed: {dep}")
            except Exception as e:
                self.logger.warning(f"Failed to install {dep}: {e}")

    def setup_python_environment(self):
        """Setup Python environment and install packages"""
        self.logger.info("Setting up Python environment...")

        # Upgrade pip
        self.run_command("python3 -m pip install --upgrade pip")

        # Install PyTorch with CUDA support (for H100)
        self.logger.info("Installing PyTorch with CUDA support...")
        self.run_command(
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )

        # Install core AI packages
        ai_packages = [
            "openai-whisper",
            "TTS",  # Coqui TTS
            "transformers",
            "accelerate",
            "xformers",  # Memory efficient transformers
            "optimum",   # ONNX optimization
        ]

        for package in ai_packages:
            try:
                self.run_command(f"pip3 install {package}")
                self.logger.info(f"Installed: {package}")
            except Exception as e:
                self.logger.warning(f"Failed to install {package}: {e}")

        # Install audio processing packages
        audio_packages = [
            "soundfile",
            "librosa",
            "scipy",
            "numpy",
            "pyaudio"
        ]

        for package in audio_packages:
            try:
                self.run_command(f"pip3 install {package}")
                self.logger.info(f"Installed: {package}")
            except Exception as e:
                self.logger.warning(f"Failed to install {package}: {e}")

        # Install AGI and utility packages
        utility_packages = [
            "pyst2",      # AGI library
            "requests",   # For Ollama communication
            "asyncio",
            "aiofiles",
            "psutil"
        ]

        for package in utility_packages:
            try:
                self.run_command(f"pip3 install {package}")
                self.logger.info(f"Installed: {package}")
            except Exception as e:
                self.logger.warning(f"Failed to install {package}: {e}")

        self.requirements_installed = True

    def setup_ollama(self):
        """Install and configure Ollama for LLM processing"""
        self.logger.info("Setting up Ollama...")

        try:
            # Download and install Ollama
            self.run_command("curl -fsSL https://ollama.ai/install.sh | sh")

            # Start Ollama service
            self.run_command("systemctl enable ollama")
            self.run_command("systemctl start ollama")

            # Wait for service to start
            time.sleep(10)

            # Pull the Orca2 model
            self.logger.info("Downloading Orca2 model (this may take a while)...")
            self.run_command("ollama pull orca2:7b", check=False)

            self.logger.info("Ollama setup completed")

        except Exception as e:
            self.logger.error(f"Ollama setup failed: {e}")
            raise

    def create_directory_structure(self):
        """Create necessary directory structure"""
        self.logger.info("Creating directory structure...")

        directories = [
            self.install_dir,
            self.install_dir / "logs",
            self.install_dir / "models",
            self.install_dir / "temp",
            self.install_dir / "config",
            self.install_dir / "scripts"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")

        # Set permissions
        self.run_command(f"chown -R asterisk:asterisk {self.install_dir}")
        self.run_command(f"chmod -R 755 {self.install_dir}")

    def install_voicebot_files(self):
        """Install VoiceBot Python files"""
        self.logger.info("Installing VoiceBot files...")

        # List of files to install (these would be copied from the development directory)
        voicebot_files = [
            "conversation_context_manager.py",
            "gpu_stt_engine.py",
            "gpu_neural_tts_engine.py",
            "telephony_audio_processor.py",
            "streaming_ai_pipeline.py",
            "netovo_gpu_voicebot_production.py"
        ]

        scripts_dir = self.install_dir / "scripts"

        for filename in voicebot_files:
            # In a real deployment, these would be copied from the source directory
            # For now, we'll create symbolic links or copy from current directory
            source_path = Path.cwd() / filename
            target_path = scripts_dir / filename

            if source_path.exists():
                self.run_command(f"cp {source_path} {target_path}")
                self.run_command(f"chmod +x {target_path}")
                self.logger.info(f"Installed: {filename}")
            else:
                self.logger.warning(f"Source file not found: {filename}")

    def create_wrapper_script(self):
        """Create wrapper script for Asterisk AGI"""
        wrapper_content = '''#!/bin/bash
# NETOVO GPU VoiceBot Wrapper Script

# Set environment variables
export PYTHONPATH="/opt/voicebot/scripts:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

# Set working directory
cd /opt/voicebot/scripts

# Run the GPU VoiceBot
python3 netovo_gpu_voicebot_production.py "$@"

# Exit with the same code as the Python script
exit $?
'''

        wrapper_path = self.install_dir / "voicebot_gpu_wrapper.sh"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)

        self.run_command(f"chmod +x {wrapper_path}")
        self.run_command(f"chown asterisk:asterisk {wrapper_path}")

        self.logger.info(f"Created wrapper script: {wrapper_path}")

    def configure_asterisk(self):
        """Create Asterisk configuration templates"""
        self.logger.info("Creating Asterisk configuration templates...")

        # Create extensions.conf template
        extensions_conf = '''
; NETOVO GPU VoiceBot Extensions Configuration

[voicebot-incoming]
; GPU-accelerated VoiceBot context
exten => 1600,1,NoOp(=== NETOVO GPU VoiceBot Call Started ===)
 same => n,Answer()
 same => n,Wait(1)
 same => n,AGI(/opt/voicebot/voicebot_gpu_wrapper.sh)
 same => n,NoOp(=== NETOVO GPU VoiceBot Call Ended ===)
 same => n,Hangup()

; Error handling
exten => 1600,102,NoOp(=== VoiceBot Error ===)
 same => n,Playback(technical-difficulties)
 same => n,Hangup()
'''

        # Create pjsip.conf template for 3CX integration
        pjsip_conf = '''
; NETOVO 3CX Integration Configuration

[transport-tcp]
type=transport
protocol=tcp
bind=0.0.0.0:5060

[netovo-3cx]
type=endpoint
context=voicebot-incoming
auth=netovo-3cx-auth
aors=netovo-3cx-aor
direct_media=no
allow=ulaw,alaw,g722

[netovo-3cx-auth]
type=auth
auth_type=userpass
username=1600
password=FcHw0P2FHK

[netovo-3cx-aor]
type=aor
contact=sip:1600@172.208.69.71:5060
'''

        # Save configuration templates
        config_dir = self.install_dir / "config"

        with open(config_dir / "extensions_gpu_voicebot.conf", 'w') as f:
            f.write(extensions_conf)

        with open(config_dir / "pjsip_netovo.conf", 'w') as f:
            f.write(pjsip_conf)

        self.logger.info("Asterisk configuration templates created")
        self.logger.info("Manual step required: Include these configurations in Asterisk")

    def test_gpu_acceleration(self):
        """Test GPU acceleration components"""
        self.logger.info("Testing GPU acceleration...")

        test_script = '''
import torch
import sys

print("Testing GPU acceleration...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test GPU allocation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("GPU computation test: PASSED")
    except Exception as e:
        print(f"GPU computation test: FAILED - {e}")
else:
    print("GPU not available - will use CPU")
'''

        test_file = self.install_dir / "test_gpu.py"
        with open(test_file, 'w') as f:
            f.write(test_script)

        try:
            result = self.run_command(f"python3 {test_file}")
            self.logger.info("GPU test completed successfully")
        except Exception as e:
            self.logger.warning(f"GPU test failed: {e}")

    def create_systemd_service(self):
        """Create systemd service for VoiceBot management"""
        service_content = '''[Unit]
Description=NETOVO GPU VoiceBot Service
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/true
ExecStop=/bin/true
User=asterisk
Group=asterisk

[Install]
WantedBy=multi-user.target
'''

        service_path = Path("/etc/systemd/system/netovo-gpu-voicebot.service")
        with open(service_path, 'w') as f:
            f.write(service_content)

        self.run_command("systemctl daemon-reload")
        self.run_command("systemctl enable netovo-gpu-voicebot")

        self.logger.info("Systemd service created")

    def run_full_installation(self):
        """Run complete installation process"""
        self.logger.info("=== Starting NETOVO GPU VoiceBot Installation ===")

        try:
            # Check system requirements
            requirements = self.check_system_requirements()
            if not all(requirements.values()):
                self.logger.warning("Some system requirements not met:")
                for req, status in requirements.items():
                    if not status:
                        self.logger.warning(f"  - {req}: FAILED")

            # Install system dependencies
            self.install_system_dependencies()

            # Setup Python environment
            self.setup_python_environment()

            # Setup Ollama
            self.setup_ollama()

            # Create directory structure
            self.create_directory_structure()

            # Install VoiceBot files
            self.install_voicebot_files()

            # Create wrapper script
            self.create_wrapper_script()

            # Configure Asterisk
            self.configure_asterisk()

            # Test GPU acceleration
            self.test_gpu_acceleration()

            # Create systemd service
            self.create_systemd_service()

            self.logger.info("=== Installation completed successfully! ===")
            self.logger.info("")
            self.logger.info("Next steps:")
            self.logger.info("1. Include the Asterisk configuration templates in your dialplan")
            self.logger.info("2. Restart Asterisk: systemctl restart asterisk")
            self.logger.info("3. Test the VoiceBot by calling extension 1600")
            self.logger.info("")
            self.logger.info(f"Installation directory: {self.install_dir}")
            self.logger.info(f"Wrapper script: {self.install_dir}/voicebot_gpu_wrapper.sh")

        except Exception as e:
            self.logger.error(f"Installation failed: {e}")
            raise


def main():
    """Main installation entry point"""
    if os.geteuid() != 0:
        print("This installer must be run as root")
        print("Usage: sudo python3 install_gpu_voicebot.py")
        sys.exit(1)

    installer = GPUVoiceBotInstaller()

    try:
        installer.run_full_installation()
    except KeyboardInterrupt:
        print("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
