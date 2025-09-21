#!/bin/bash
# Advanced Audio Diagnostics for NETOVO VoiceBot
# Monitors TTS, STT, and VoIP audio quality in real-time

echo "=========================================="
echo "NETOVO Advanced Audio Diagnostics"
echo "Monitoring TTS, STT, and VoIP Quality"
echo "=========================================="

# Create comprehensive audio monitoring script
cat > /opt/voicebot/monitor_audio_quality.py << 'EOF'
#!/usr/bin/env python3
"""
Advanced Audio Quality Monitor for NETOVO VoiceBot
Real-time monitoring of TTS, STT, and VoIP audio streams
"""

import subprocess
import time
import os
import psutil
import threading
from datetime import datetime

class AudioQualityMonitor:
    def __init__(self):
        self.monitoring = True
        self.log_file = f"/var/log/asterisk/audio_quality_{int(time.time())}.log"

    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"{timestamp} - {message}"
        print(log_message)

        try:
            with open(self.log_file, 'a') as f:
                f.write(log_message + "\n")
        except:
            pass

    def monitor_system_resources(self):
        """Monitor system resources affecting audio"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Disk I/O
                disk_io = psutil.disk_io_counters()

                # Network I/O
                network_io = psutil.net_io_counters()

                self.log(f"SYSTEM - CPU: {cpu_percent}%, Memory: {memory_percent}%, "
                        f"Disk R/W: {disk_io.read_bytes}/{disk_io.write_bytes}, "
                        f"Network R/W: {network_io.bytes_recv}/{network_io.bytes_sent}")

                # Alert on high resource usage
                if cpu_percent > 80:
                    self.log("WARNING - High CPU usage may cause audio breaking")
                if memory_percent > 85:
                    self.log("WARNING - High memory usage may affect audio quality")

                time.sleep(5)

            except Exception as e:
                self.log(f"ERROR monitoring system resources: {e}")
                time.sleep(5)

    def monitor_asterisk_channels(self):
        """Monitor active Asterisk channels and audio quality"""
        while self.monitoring:
            try:
                # Get channel information
                result = subprocess.run([
                    'sudo', 'asterisk', '-rx', 'core show channels'
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    channels_info = result.stdout
                    active_channels = len([line for line in channels_info.split('\n')
                                         if 'PJSIP' in line and 'Up' in line])

                    self.log(f"ASTERISK - Active channels: {active_channels}")

                    if active_channels > 0:
                        # Get detailed channel info
                        detail_result = subprocess.run([
                            'sudo', 'asterisk', '-rx', 'core show channels verbose'
                        ], capture_output=True, text=True, timeout=10)

                        if detail_result.returncode == 0:
                            self.log(f"ASTERISK DETAIL - {detail_result.stdout[:200]}...")

                time.sleep(3)

            except Exception as e:
                self.log(f"ERROR monitoring Asterisk: {e}")
                time.sleep(5)

    def monitor_network_quality(self):
        """Monitor network quality for VoIP"""
        while self.monitoring:
            try:
                # Ping test for latency
                ping_result = subprocess.run([
                    'ping', '-c', '1', '8.8.8.8'
                ], capture_output=True, text=True, timeout=5)

                if ping_result.returncode == 0:
                    # Extract latency
                    for line in ping_result.stdout.split('\n'):
                        if 'time=' in line:
                            latency = line.split('time=')[1].split(' ')[0]
                            self.log(f"NETWORK - Latency: {latency}ms")

                            # Alert on high latency
                            try:
                                if float(latency) > 150:
                                    self.log("WARNING - High network latency may cause audio breaking")
                            except:
                                pass
                            break

                # Check network interface statistics
                net_stats = psutil.net_io_counters(pernic=True)
                for interface, stats in net_stats.items():
                    if interface.startswith(('eth', 'ens', 'enp')):
                        self.log(f"NETWORK - {interface}: "
                                f"RX: {stats.bytes_recv} bytes, TX: {stats.bytes_sent} bytes, "
                                f"Errors: RX {stats.errin}, TX {stats.errout}, "
                                f"Drops: RX {stats.dropin}, TX {stats.dropout}")

                time.sleep(10)

            except Exception as e:
                self.log(f"ERROR monitoring network: {e}")
                time.sleep(10)

    def test_tts_quality(self):
        """Test TTS audio generation quality"""
        try:
            self.log("TTS QUALITY TEST - Starting...")

            test_phrases = [
                "Hello, this is a short test",
                "This is a medium length test phrase for audio quality assessment",
                "This is a very long test phrase that will help us determine if there are any issues with extended speech generation that might cause audio breaking or quality degradation during longer responses"
            ]

            for i, phrase in enumerate(test_phrases):
                self.log(f"TTS TEST {i+1} - Testing: '{phrase[:30]}...'")

                # Test espeak
                start_time = time.time()
                temp_file = f"/tmp/tts_test_{i}_{int(time.time())}.wav"

                result = subprocess.run([
                    'espeak', phrase,
                    '-w', temp_file,
                    '-s', '150', '-p', '50'
                ], capture_output=True, timeout=15)

                duration = time.time() - start_time

                if result.returncode == 0 and os.path.exists(temp_file):
                    file_size = os.path.getsize(temp_file)

                    # Analyze audio file
                    audio_info = subprocess.run([
                        'ffprobe', '-v', 'quiet',
                        '-show_entries', 'format=duration,bit_rate:stream=sample_rate,channels',
                        '-of', 'csv=p=0',
                        temp_file
                    ], capture_output=True, text=True)

                    if audio_info.returncode == 0:
                        self.log(f"TTS TEST {i+1} - SUCCESS: {file_size} bytes, {duration:.2f}s generation")
                        self.log(f"TTS TEST {i+1} - Audio info: {audio_info.stdout.strip()}")
                    else:
                        self.log(f"TTS TEST {i+1} - Audio analysis failed")

                    # Cleanup
                    os.unlink(temp_file)
                else:
                    self.log(f"TTS TEST {i+1} - FAILED: {result.stderr}")

                time.sleep(2)

        except Exception as e:
            self.log(f"ERROR in TTS quality test: {e}")

    def monitor_temp_files(self):
        """Monitor /tmp for TTS/STT file creation and cleanup"""
        while self.monitoring:
            try:
                # Count temp files
                temp_files = os.listdir('/tmp')
                tts_files = [f for f in temp_files if 'speak' in f or 'tts' in f or 'enterprise' in f]
                stt_files = [f for f in temp_files if 'customer' in f or 'stt' in f or 'whisper' in f]

                self.log(f"TEMP FILES - TTS: {len(tts_files)}, STT: {len(stt_files)}")

                if len(tts_files) > 10:
                    self.log("WARNING - Many TTS temp files, possible cleanup issue")
                if len(stt_files) > 10:
                    self.log("WARNING - Many STT temp files, possible cleanup issue")

                time.sleep(15)

            except Exception as e:
                self.log(f"ERROR monitoring temp files: {e}")
                time.sleep(15)

    def start_monitoring(self):
        """Start all monitoring threads"""
        self.log("AUDIO QUALITY MONITOR - Starting comprehensive monitoring...")

        # Start monitoring threads
        threads = [
            threading.Thread(target=self.monitor_system_resources, daemon=True),
            threading.Thread(target=self.monitor_asterisk_channels, daemon=True),
            threading.Thread(target=self.monitor_network_quality, daemon=True),
            threading.Thread(target=self.monitor_temp_files, daemon=True),
        ]

        for thread in threads:
            thread.start()

        # Run TTS quality test
        self.test_tts_quality()

        # Keep monitoring
        try:
            while True:
                self.log("MONITOR - All systems monitoring... (Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            self.log("MONITOR - Stopping monitoring...")
            self.monitoring = False

if __name__ == "__main__":
    monitor = AudioQualityMonitor()
    monitor.start_monitoring()
EOF

chmod +x /opt/voicebot/monitor_audio_quality.py

# Create real-time audio stream analyzer
cat > /opt/voicebot/analyze_audio_stream.py << 'EOF'
#!/usr/bin/env python3
"""
Real-time Audio Stream Analyzer for VoIP Quality Issues
Captures and analyzes audio during live calls
"""

import subprocess
import time
import os
from datetime import datetime

def capture_rtp_stream():
    """Capture RTP audio stream for analysis"""
    print("🎤 Starting RTP stream capture...")
    print("📞 Make a call now and speak to capture audio...")

    # Use tcpdump to capture RTP packets
    capture_file = f"/tmp/rtp_capture_{int(time.time())}.pcap"

    try:
        # Capture RTP traffic
        print(f"📡 Capturing RTP traffic to {capture_file}")
        subprocess.run([
            'sudo', 'tcpdump',
            '-i', 'any',
            '-w', capture_file,
            'udp and portrange 10000-20000',
            '-c', '1000'  # Capture 1000 packets
        ], timeout=60)

        if os.path.exists(capture_file):
            file_size = os.path.getsize(capture_file)
            print(f"✅ Captured {file_size} bytes of RTP traffic")

            # Analyze the capture
            print("🔍 Analyzing RTP stream...")
            analysis = subprocess.run([
                'tshark', '-r', capture_file,
                '-q', '-z', 'rtp,streams'
            ], capture_output=True, text=True)

            if analysis.returncode == 0:
                print("📊 RTP Stream Analysis:")
                print(analysis.stdout)
            else:
                print("⚠️ RTP analysis failed - tshark not available")

        else:
            print("❌ No RTP traffic captured")

    except subprocess.TimeoutExpired:
        print("⏱️ Capture timeout - analysis complete")
    except Exception as e:
        print(f"❌ Capture error: {e}")

def test_audio_codecs():
    """Test different audio codecs for quality"""
    print("\n🔊 Testing Audio Codecs...")

    test_text = "Testing audio codec quality for VoIP optimization"
    codecs = [
        ('G.722', ['-ar', '16000', '-ac', '1']),
        ('ulaw', ['-ar', '8000', '-ac', '1', '-acodec', 'pcm_mulaw']),
        ('alaw', ['-ar', '8000', '-ac', '1', '-acodec', 'pcm_alaw']),
    ]

    for codec_name, ffmpeg_args in codecs:
        try:
            print(f"\n🎵 Testing {codec_name} codec...")

            # Generate test audio
            raw_file = f"/tmp/test_raw_{int(time.time())}.wav"
            encoded_file = f"/tmp/test_{codec_name.lower()}_{int(time.time())}.wav"

            # Create raw audio
            subprocess.run([
                'espeak', test_text,
                '-w', raw_file
            ], check=True)

            # Encode with codec
            cmd = ['ffmpeg', '-i', raw_file] + ffmpeg_args + ['-y', encoded_file]
            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0 and os.path.exists(encoded_file):
                raw_size = os.path.getsize(raw_file)
                encoded_size = os.path.getsize(encoded_file)
                compression = (1 - encoded_size/raw_size) * 100

                print(f"✅ {codec_name}: {raw_size} → {encoded_size} bytes ({compression:.1f}% compression)")

                # Cleanup
                os.unlink(raw_file)
                os.unlink(encoded_file)
            else:
                print(f"❌ {codec_name}: Encoding failed")

        except Exception as e:
            print(f"❌ {codec_name}: Error - {e}")

def monitor_live_call():
    """Monitor audio quality during a live call"""
    print("\n📞 Live Call Audio Monitor")
    print("Make a call now and this will monitor audio quality...")

    start_time = time.time()

    while time.time() - start_time < 120:  # Monitor for 2 minutes
        try:
            # Check Asterisk channels
            channels_result = subprocess.run([
                'sudo', 'asterisk', '-rx', 'core show channels'
            ], capture_output=True, text=True)

            if 'Up' in channels_result.stdout:
                print(f"🟢 {datetime.now().strftime('%H:%M:%S')} - Active call detected")

                # Check RTP quality
                rtp_result = subprocess.run([
                    'sudo', 'asterisk', '-rx', 'rtp show stats'
                ], capture_output=True, text=True)

                if rtp_result.returncode == 0:
                    print(f"📊 RTP Stats: {rtp_result.stdout[:100]}...")

            else:
                print(f"⚪ {datetime.now().strftime('%H:%M:%S')} - No active calls")

            time.sleep(3)

        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            time.sleep(3)

def main():
    print("NETOVO Audio Stream Analyzer")
    print("=" * 40)

    choice = input("""
Choose analysis type:
1. Capture RTP stream (requires active call)
2. Test audio codecs
3. Monitor live call quality
4. All tests

Enter choice (1-4): """).strip()

    if choice == '1':
        capture_rtp_stream()
    elif choice == '2':
        test_audio_codecs()
    elif choice == '3':
        monitor_live_call()
    elif choice == '4':
        test_audio_codecs()
        print("\n" + "="*40)
        monitor_live_call()
        print("\n" + "="*40)
        capture_rtp_stream()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/voicebot/analyze_audio_stream.py

# Create bandwidth optimization script
cat > /opt/voicebot/optimize_bandwidth.sh << 'EOF'
#!/bin/bash
# Advanced Bandwidth Optimization for VoIP Quality

echo "🔧 Advanced VoIP Bandwidth Optimization"

# Optimize Linux network stack for VoIP
echo "Optimizing network stack..."

# Increase network buffer sizes
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.udp_rmem_min=8192
sudo sysctl -w net.ipv4.udp_wmem_min=8192

# Optimize for low latency
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
sudo sysctl -w net.core.default_qdisc=fq

# Make changes persistent
sudo tee -a /etc/sysctl.conf >> /dev/null << 'SYSCTL_EOF'

# VoIP Optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.udp_rmem_min=8192
net.ipv4.udp_wmem_min=8192
net.ipv4.tcp_congestion_control=bbr
net.core.default_qdisc=fq
SYSCTL_EOF

# Optimize Asterisk for lower bandwidth
sudo tee /etc/asterisk/codecs.conf > /dev/null << 'CODEC_EOF'
[general]
; Optimize codec selection for bandwidth
;
[g722]
generic_plc = true
; G.722 provides HD audio but uses more bandwidth

[g711]
; G.711 (ulaw/alaw) standard quality, good bandwidth efficiency
generic_plc = true

[gsm]
; GSM very low bandwidth, lower quality
generic_plc = true
CODEC_EOF

# Update SIP configuration for bandwidth optimization
sudo tee -a /etc/asterisk/sip.conf >> /dev/null << 'SIP_EOF'

; Additional bandwidth optimizations
progressinband=no
useragent=NETOVO-Asterisk
compactheaders=yes
videosupport=no
maxcallbitrate=48
; Prefer low-bandwidth codecs
disallow=all
allow=ulaw    ; 64 kbps
allow=alaw    ; 64 kbps
allow=gsm     ; 13 kbps (very low bandwidth)
allow=g722    ; 64 kbps (HD audio)
SIP_EOF

echo "✅ Bandwidth optimization complete"
echo "🔄 Restart Asterisk to apply changes: sudo systemctl restart asterisk"
EOF

chmod +x /opt/voicebot/optimize_bandwidth.sh

echo "=========================================="
echo "Audio Diagnostics Tools Created"
echo "=========================================="

echo "📊 Available Tools:"
echo "1. monitor_audio_quality.py - Comprehensive real-time monitoring"
echo "2. analyze_audio_stream.py - RTP stream analysis"
echo "3. optimize_bandwidth.sh - Advanced bandwidth optimization"

echo ""
echo "🚀 Usage:"
echo "Real-time monitoring: python3 /opt/voicebot/monitor_audio_quality.py"
echo "Stream analysis: python3 /opt/voicebot/analyze_audio_stream.py"
echo "Bandwidth optimization: sudo /opt/voicebot/optimize_bandwidth.sh"

echo ""
echo "🎯 To diagnose voice breaking:"
echo "1. Start monitoring BEFORE making a call"
echo "2. Make a test call while monitoring runs"
echo "3. Check logs for bottlenecks and quality issues"
