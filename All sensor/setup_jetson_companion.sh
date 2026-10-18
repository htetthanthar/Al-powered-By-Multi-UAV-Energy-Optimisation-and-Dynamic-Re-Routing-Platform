#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "  UAV COMPANION COMPUTER PROVISIONING (NVIDIA JETSON / UBUNTU 22.04 LTS)  "
echo "=========================================================================="

# 1. System Update and Core Build Tools
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libssl-dev \
    screen \
    net-tools

# 2. Install ROS 2 Humble Core
if [ ! -d "/opt/ros/humble" ]; then
    echo "[*] Installing ROS 2 Humble Base..."
    sudo apt-get install -y software-properties-common curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y ros-humble-ros-base python3-colcon-common-extensions
fi

# 3. Build Micro-XRCE-DDS Agent for PX4 Fast-RTPS Bridge
if [ ! -d "/usr/local/bin/MicroXRCEAgent" ]; then
    echo "[*] Compiling Micro-XRCE-DDS Agent..."
    cd /tmp
    git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
    cd Micro-XRCE-DDS-Agent && mkdir build && cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    sudo ldconfig
fi

# 4. Install Python Dependencies & ONNX Runtime with TensorRT Acceleration
echo "[*] Installing Python Machine Learning Runtime Dependencies..."
pip3 install --upgrade pip
pip3 install \
    numpy \
    pandas \
    onnx \
    onnxruntime-gpu \
    pymavlink \
    httpx \
    shapely \
    duckdb

# 5. Configure Systemd Service for Auto-Boot UAV Control
SERVICE_FILE="/etc/systemd/system/uav-trajectory-daemon.service"
echo "[*] Installing Systemd Autonomous Dispatch Daemon..."

sudo bash -c "cat <<EOF > ${SERVICE_FILE}
[Unit]
Description=Autonomous UAV Dynamic Trajectory Daemon
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/ros2_uav_node.py --onnx_model_path $(pwd)/checkpoints/onnx/energy_surrogate.onnx
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable uav-trajectory-daemon.service

echo "=========================================================================="
echo "[+] COMPANION PROVISIONING COMPLETE. REBOOT TO ENGAGE FLIGHT DAEMON."
echo "=========================================================================="