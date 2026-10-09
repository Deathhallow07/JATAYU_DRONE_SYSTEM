# Install — ROS 2 Lyrical (Ubuntu 26.04, Nav2 1.5)

Straight-through install for the **dev / simulation machine**: x86_64,
Ubuntu 26.04 LTS, ROS 2 Lyrical, Nav2 1.5. This is the configuration the repo is
written against — `nav2_params.yaml` and the `/cmd_vel` subscribers both target
Nav2 1.5, so **nothing needs patching on this path**.

Deploying to a Jetson on Ubuntu 22.04 instead? Use
[`INSTALL_HUMBLE.md`](INSTALL_HUMBLE.md) — the Nav2 config is *not* portable
between the two.

Tags: **[COMMON]** always · **[SIM]** simulation only · **[HW]** real lidar/FCU only.
Run the stages in order.

```bash
export ROS_DISTRO=lyrical
```

---

## 0. Prerequisites [COMMON]

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget lsb-release gnupg build-essential python3-pip

# ROS 2 needs a UTF-8 locale
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

## 1. ROS 2 Lyrical [COMMON]

```bash
sudo add-apt-repository -y universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-lyrical-desktop ros-dev-tools

echo "source /opt/ros/lyrical/setup.bash" >> ~/.bashrc
source /opt/ros/lyrical/setup.bash

sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update
```

## 2. Nav2 1.5 + RViz [COMMON]

```bash
sudo apt install -y \
  ros-lyrical-navigation2 \
  ros-lyrical-nav2-bringup \
  ros-lyrical-nav2-mppi-controller \
  ros-lyrical-nav2-rviz-plugins \
  ros-lyrical-rviz2 \
  ros-lyrical-tf2-ros \
  ros-lyrical-tf2-geometry-msgs
```

`navigation2` pulls controller / planner / behaviors / bt-navigator /
waypoint-follower / lifecycle-manager / costmap-2d / navfn. The explicit
`nav2-mppi-controller` line is insurance — MPPI is the controller this stack uses.

## 3. MAVROS + GeographicLib datasets [COMMON]

```bash
sudo apt install -y ros-lyrical-mavros ros-lyrical-mavros-extras ros-lyrical-mavros-msgs

# REQUIRED, one time — MAVROS will not start cleanly without the geoid datasets
sudo /opt/ros/lyrical/lib/mavros/install_geographiclib_datasets.sh
```

## 4. Gazebo ↔ ROS bridge [SIM]

```bash
sudo apt install -y ros-lyrical-ros-gz
```

The `ros-gz` metapackage pulls the Gazebo release paired with your ROS distro
and provides `ros_gz_bridge`, which this stack uses to bridge `/clock` and
`/lidar/scan`.

## 5. Gazebo Sim [SIM]

```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
sudo apt install -y gz-harmonic
```

## 6. ardupilot_gazebo plugin [SIM]

```bash
sudo apt install -y libgz-sim8-dev rapidjson-dev

cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo
cd ardupilot_gazebo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

cat >> ~/.bashrc <<'EOF'
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
EOF
source ~/.bashrc
```

## 7. ArduPilot SITL + MAVProxy [SIM]

```bash
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

./waf configure --board sitl
./waf copter

echo 'export PATH=$PATH:$HOME/ardupilot/Tools/autotest' >> ~/.bashrc
source ~/.bashrc
```

The prereqs script installs MAVProxy and pymavlink. If they are missing:

```bash
pip3 install --user MAVProxy pymavlink
```

## 8. LightWare SF45/B driver [HW]

```bash
mkdir -p ~/obstacle_avoidance_drone/src
cd ~/obstacle_avoidance_drone/src
git clone https://github.com/LightWare-Optoelectronics/lightwarelidar2

sudo usermod -aG dialout $USER    # log out and back in for this to take effect
```

Optional: LightWare Studio (`.deb` from lightware.co.za) for sensor configuration.

## 9. This project [COMMON]

```bash
cd ~/obstacle_avoidance_drone/src
git clone https://github.com/botvik155/obstacle_avoidance_drone.git .
```

Expected layout:

```
~/obstacle_avoidance_drone/
  src/
    lightwarelidar2/          # stage 8
    obstacle_avoidance/       # simulation package
    obstacle_avoidance_hw/    # hardware package
```

The live nodes need only rclpy / geometry_msgs / tf2, all of which come with
ROS. No numpy, OpenCV, or torch.

## 10. Build [COMMON]

```bash
cd ~/obstacle_avoidance_drone
source /opt/ros/lyrical/setup.bash

rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select lightwarelidar2

echo "source ~/obstacle_avoidance_drone/install/setup.bash" >> ~/.bashrc
source ~/obstacle_avoidance_drone/install/setup.bash
```

> **setuptools caveat.** Ubuntu 26.04 ships setuptools ≥ 80, which dropped the
> commands colcon uses to build **ament_python** packages — you get
> `error: option --editable not recognized`. The launch files run our Python
> nodes straight from source, so building them is optional. To build them
> anyway: `pip3 install "setuptools<80"` then `colcon build`.

## 11. Verify

```bash
ros2 pkg prefix nav2_mppi_controller && echo "nav2 OK"
ros2 pkg prefix mavros             && echo "mavros OK"
ros2 pkg prefix ros_gz_bridge      && echo "ros_gz OK"
ros2 pkg prefix lightwarelidar2    && echo "sf45b driver OK"
gz sim --version
which sim_vehicle.py
```

## 12. Run

### Simulation — one command

```bash
./start_sim.sh          # bring up everything and attach to the tmux session
./start_sim.sh stop
./start_sim.sh status
```

`start_sim.sh` defaults to `ROS_SETUP=/opt/ros/lyrical/setup.bash`; override the
variable if your path differs. It launches Gazebo, ArduPilot SITL, the gz→ROS
bridges, MAVROS, the TF bridge, Nav2, and RViz across six tmux panes, and gets
the drone airborne. Then send goals:

```bash
python3 lawnmower.py                  # the full survey pattern
python3 send_nav_goal.py 0 45         # single goal: x=east, y=north
python3 send_nav_goal.py --cancel
```

### Simulation — manual, one terminal per stage

```bash
gz sim -v4 -r runway.sdf
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
ros2 launch src/obstacle_avoidance/launch/bringup.launch.py
# arm → GUIDED → takeoff, then:
ros2 launch src/obstacle_avoidance/launch/nav2_mppi.launch.py
rviz2 -d src/obstacle_avoidance/config/nav2_drone.rviz
```

### Hardware — real SF45/B and flight controller

```bash
ros2 launch src/obstacle_avoidance_hw/launch/bringup_hw.launch.py \
    fcu_url:=/dev/ttyACM1:921600 lidar_port:=/dev/ttyACM0

# after position fix + takeoff:
ros2 launch src/obstacle_avoidance_hw/launch/nav2_hw.launch.py

# RViz on the ground station, same ROS_DOMAIN_ID:
rviz2 -d src/obstacle_avoidance_hw/config/nav2_drone.rviz
```

> ⚠️ **Hardware needs a position estimate.** Without GPS, optical-flow, VIO, or
> mocap, `/mavros/local_position/pose` never publishes — so `map→base_link` is
> absent, Nav2 cannot run, and ArduPilot GUIDED velocity control will not work
> either. GPS is useless indoors; use optical-flow + rangefinder or VIO there.

> **Give the velocity sender its own MAVLink port.** Two processes on one UDP
> socket steal each other's packets. MAVROS holds 14550 for pose and telemetry;
> the sim sender defaults to `udpin:0.0.0.0:14555` (which `start_sim.sh` gives
> SITL via `--out`) and the hardware sender to `udpin:0.0.0.0:14551`. For a real
> FCU, forward the link to a second UDP port with MAVProxy or mavlink-router and
> pass `mavlink_conn:=…` to `nav2_hw.launch.py`.

---

## Known-good versions

Dev machine as installed:

| Component | Version |
|---|---|
| Ubuntu | 26.04 LTS |
| ROS 2 | Lyrical |
| Python | 3.14.4 |
| Gazebo Sim | 10.4.0 |
| nav2 (mppi) | 1.5.1 |
| mavros / extras | 2.14.0 |
| ros_gz_bridge | 3.0.9 |
| rviz2 | 15.2.5 |
| lightwarelidar2 | source (main) |
