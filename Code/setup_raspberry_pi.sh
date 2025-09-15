#!/bin/bash
# ASL Alphabet Recognition System - Raspberry Pi 4 Setup Script
# This script sets up the ASL recognition system on Raspberry Pi 4

echo "=========================================="
echo "ASL ALPHABET RECOGNITION SYSTEM"
echo "Raspberry Pi 4 Setup Script"
echo "=========================================="

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  Warning: This script is designed for Raspberry Pi"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python 3 and pip if not already installed
echo "🐍 Installing Python 3 and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y libhdf5-dev libhdf5-serial-dev libhdf5-103
sudo apt install -y libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install -y libatlas-base-dev libjasper-dev
sudo apt install -y libqt4-test
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libv4l-dev libxvidcore-dev libx264-dev
sudo apt install -y libgtk-3-dev libtbb2 libtbb-dev libdc1394-22-dev

# Install espeak for text-to-speech
echo "🔊 Installing text-to-speech engine..."
sudo apt install -y espeak espeak-data libespeak1 libespeak-dev

# Create virtual environment
echo "🌐 Creating Python virtual environment..."
python3 -m venv asl_env
source asl_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python packages optimized for Pi
echo "📚 Installing Python packages..."
pip install --no-cache-dir opencv-python==4.5.5.64
pip install --no-cache-dir mediapipe==0.8.11
pip install --no-cache-dir pyttsx3==2.90
pip install --no-cache-dir numpy==1.21.6

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /home/pi/asl_recognition
mkdir -p /home/pi/asl_recognition/logs
mkdir -p /home/pi/asl_recognition/data
mkdir -p /home/pi/asl_recognition/exports

# Copy files to Pi directory
echo "📋 Copying application files..."
cp asl_raspberry_pi.py /home/pi/asl_recognition/
cp pi_config.json /home/pi/asl_recognition/asl_config.json
cp requirements_asl.txt /home/pi/asl_recognition/

# Set permissions
echo "🔐 Setting permissions..."
chmod +x /home/pi/asl_recognition/asl_raspberry_pi.py
chown -R pi:pi /home/pi/asl_recognition

# Create desktop shortcut
echo "🖥️  Creating desktop shortcut..."
cat > /home/pi/Desktop/ASL_Recognition.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ASL Recognition
Comment=ASL Alphabet Recognition System
Exec=cd /home/pi/asl_recognition && source asl_env/bin/activate && python asl_raspberry_pi.py
Icon=applications-education
Terminal=true
Categories=Education;Accessibility;
EOF

chmod +x /home/pi/Desktop/ASL_Recognition.desktop

# Create startup script
echo "🚀 Creating startup script..."
cat > /home/pi/asl_recognition/start_asl.sh << 'EOF'
#!/bin/bash
cd /home/pi/asl_recognition
source asl_env/bin/activate
python asl_raspberry_pi.py
EOF

chmod +x /home/pi/asl_recognition/start_asl.sh

# Test camera
echo "📷 Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('✅ Camera is working')
    else:
        print('⚠️  Camera opened but failed to read')
    cap.release()
else:
    print('❌ Camera not accessible')
"

# Create systemd service (optional)
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/asl-recognition.service > /dev/null << EOF
[Unit]
Description=ASL Alphabet Recognition System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/asl_recognition
ExecStart=/home/pi/asl_recognition/start_asl.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable service (optional)
echo "🔧 Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable asl-recognition.service

echo ""
echo "=========================================="
echo "🎉 SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "✅ ASL Alphabet Recognition System is ready for Raspberry Pi 4"
echo ""
echo "📋 Quick Start:"
echo "   1. Desktop: Double-click 'ASL Recognition' icon"
echo "   2. Terminal: cd /home/pi/asl_recognition && ./start_asl.sh"
echo "   3. Service: sudo systemctl start asl-recognition"
echo ""
echo "🔧 Configuration:"
echo "   - Edit: /home/pi/asl_recognition/asl_config.json"
echo "   - Logs: /home/pi/asl_recognition/logs/"
echo "   - Data: /home/pi/asl_recognition/data/"
echo ""
echo "📚 Troubleshooting:"
echo "   - Check camera: ls /dev/video*"
echo "   - Check logs: tail -f /home/pi/asl_recognition/logs/asl_recognition.log"
echo "   - Restart service: sudo systemctl restart asl-recognition"
echo ""
echo "🎯 Performance Tips:"
echo "   - Use good lighting"
echo "   - Close other applications"
echo "   - Consider overclocking Pi (optional)"
echo ""
echo "=========================================="
