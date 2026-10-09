# Obstacle-Avoidance Drone Stack — Full Install From Scratch

End-to-end setup for the Nav2 (MPPI) + lidar obstacle-avoidance stack, covering
both the **simulation** path (Gazebo + ArduPilot SITL) and the **hardware** path
(LightWare SF45/B lidar + real flight controller).

> **Prefer a single-distro guide?** This page installs both platforms from one
> set of stages with `$ROS_DISTRO` switching. If you only care about one
> machine, these are straight-through with nothing to substitute:
>
> | Guide | Platform |
> |---|---|
> | [`INSTALL_LYRICAL.md`](INSTALL_LYRICAL.md) | Ubuntu 26.04 · ROS 2 Lyrical · Nav2 1.5 — dev / simulation box, runs the repo as-is |
> | [`INSTALL_HUMBLE.md`](INSTALL_HUMBLE.md) | Ubuntu 22.04 · ROS 2 Humble · Nav2 1.1 — Jetson Orin NX OBC, **includes the Nav2 1.1 port patches** |

Legend:  [COMMON] needed for everything · [SIM] simulation only · [HW] hardware only

Run the stages in order. Lines starting with `sudo` need admin rights.

--------------------------------------------------------------------------------
## Platforms
--------------------------------------------------------------------------------
- **Dev / simulation machine:** x86_64, Ubuntu 26.04 LTS, **ROS 2 Lyrical**
  (Nav2 1.5) — run all stages. This is what `start_sim.sh` / `start_sim_oa.sh`
  default to (`ROS_SETUP=/opt/ros/lyrical/setup.bash`).
- **Onboard computer (OBC):** **NVIDIA Jetson Orin NX 16 GB**, JetPack 6.x
  (Ubuntu 22.04, **arm64/aarch64**) — **ROS 2 Humble**, because Lyrical has no
  22.04 build. **No Gazebo, no ArduPilot SITL needed** — run only stages
  **0, 1, 2, 3, 8, 9, 10** and **skip 4–7**.

> ⚠️ **The two platforms are on different ROS releases, and the Nav2 config is
> not portable between them.** The `nav2_params.yaml` files in this repo target
> **Nav2 1.5**; on Humble's Nav2 1.1 they fail to load. See
> [Nav2 1.5 vs 1.1 parameters](#nav2-15-vs-11-parameters) before deploying to
> the Jetson.

Substitute your distro throughout: the stages below write `$ROS_DISTRO`, so
export it once and the apt lines resolve correctly on either platform.

```bash
export ROS_DISTRO=lyrical     # dev / simulation machine (Ubuntu 26.04)
# export ROS_DISTRO=humble    # Jetson Orin NX (JetPack 6 / Ubuntu 22.04)
```

All apt commands below use `$(dpkg --print-architecture)`, so they resolve to
`arm64` on the Jetson automatically — the same lines work on both platforms.

### Jetson Orin NX quick notes
- JetPack 6 is Ubuntu 22.04 → ROS 2 **Humble** installs from apt on arm64 as normal.
  Humble is the *only* option here; Lyrical (used on the dev machine) needs 24.04+.
- The stack needs **no CUDA / torch** — the live nodes are pure rclpy. Nothing here
  depends on the Jetson GPU.
- Headless OBC: install **`ros-$ROS_DISTRO-ros-base`** instead of `ros-$ROS_DISTRO-desktop`
  (stage 1) to save space; run RViz on your ground station, not the Jetson.
- Set max performance before flying:
  ```bash
  sudo nvpmodel -m 0      # MAXN power mode
  sudo jetson_clocks      # lock clocks high (helps MPPI/costmap real-time)
  ```
- Real flight controller wiring (instead of SITL): connect the FCU by USB
  (`/dev/ttyACM*`) or the Jetson UART (`/dev/ttyTHS1`) and pass it as `fcu_url`
  (see stage 12). Add yourself to `dialout` (stage 8) for both the lidar and FCU.

--------------------------------------------------------------------------------
## 0. Prerequisites [COMMON]
--------------------------------------------------------------------------------
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget lsb-release gnupg build-essential python3-pip
```

Set the locale (ROS 2 needs UTF-8):
```bash
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

--------------------------------------------------------------------------------
## 1. ROS 2 (Lyrical on the dev box, Humble on the Jetson) [COMMON]
--------------------------------------------------------------------------------
```bash
sudo add-apt-repository -y universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-$ROS_DISTRO-desktop ros-dev-tools
# Jetson / headless OBC: use the lean base instead (run RViz on the ground station):
#   sudo apt install -y ros-$ROS_DISTRO-ros-base ros-dev-tools
```

Source ROS in every shell:
```bash
echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
source /opt/ros/$ROS_DISTRO/setup.bash
```

Build tooling (colcon + rosdep):
```bash
sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update
```

--------------------------------------------------------------------------------
## 2. Nav2 + RViz [COMMON]
--------------------------------------------------------------------------------
```bash
sudo apt install -y \
  ros-$ROS_DISTRO-navigation2 \
  ros-$ROS_DISTRO-nav2-bringup \
  ros-$ROS_DISTRO-nav2-mppi-controller \
  ros-$ROS_DISTRO-nav2-rviz-plugins \
  ros-$ROS_DISTRO-rviz2 \
  ros-$ROS_DISTRO-tf2-ros \
  ros-$ROS_DISTRO-tf2-geometry-msgs
```
(`navigation2` pulls controller/planner/behaviors/bt-navigator/waypoint-follower/
lifecycle-manager/costmap-2d/navfn; the mppi-controller line is explicit insurance.)

--------------------------------------------------------------------------------
## 3. MAVROS (+ GeographicLib datasets) [COMMON]
--------------------------------------------------------------------------------
```bash
sudo apt install -y ros-$ROS_DISTRO-mavros ros-$ROS_DISTRO-mavros-extras ros-$ROS_DISTRO-mavros-msgs

# REQUIRED one-time: geoid/gravity/magnetic datasets (MAVROS won't start cleanly without them)
sudo /opt/ros/$ROS_DISTRO/lib/mavros/install_geographiclib_datasets.sh
```

--------------------------------------------------------------------------------
## 4. Gazebo↔ROS bridge [SIM]
--------------------------------------------------------------------------------
Gazebo Sim pairing (Harmonic on Humble, Ionic on Lyrical):
```bash
sudo apt install -y ros-$ROS_DISTRO-ros-gz     # metapackage: pulls the right Gazebo for your distro
```
This provides `ros_gz_bridge` (used to bridge `/clock` and `/lidar/scan`).

--------------------------------------------------------------------------------
## 5. Gazebo Sim 8 "Harmonic" [SIM]
--------------------------------------------------------------------------------
```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
sudo apt install -y gz-harmonic
```

--------------------------------------------------------------------------------
## 6. ardupilot_gazebo plugin [SIM]
--------------------------------------------------------------------------------
```bash
sudo apt install -y libgz-sim8-dev rapidjson-dev

cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo
cd ardupilot_gazebo
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

# Environment (append to ~/.bashrc)
cat >> ~/.bashrc <<'EOF'
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
EOF
source ~/.bashrc
```

--------------------------------------------------------------------------------
## 7. ArduPilot SITL + MAVProxy [SIM]
--------------------------------------------------------------------------------
```bash
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile

./waf configure --board sitl
./waf copter

# add autotest (sim_vehicle.py) to PATH
echo 'export PATH=$PATH:$HOME/ardupilot/Tools/autotest' >> ~/.bashrc
source ~/.bashrc
```
MAVProxy/pymavlink are installed by the prereqs script. If missing:
```bash
pip3 install --user MAVProxy pymavlink
```

--------------------------------------------------------------------------------
## 8. LightWare SF45/B driver [HW]
--------------------------------------------------------------------------------
Cloned into the workspace src (built in stage 10):
```bash
mkdir -p ~/obstacle_avoidance_drone/src
cd ~/obstacle_avoidance_drone/src
git clone https://github.com/LightWare-Optoelectronics/lightwarelidar2
```
Serial access for the USB sensor (log out/in afterwards):
```bash
sudo usermod -aG dialout $USER
```
(Optional) LightWare Studio for sensor config — download the .deb from lightware.co.za.

--------------------------------------------------------------------------------
## 9. This project's packages [COMMON]
--------------------------------------------------------------------------------
Place `obstacle_avoidance` (sim) and `obstacle_avoidance_hw` (hardware) under
`~/obstacle_avoidance_drone/src/`. If you're cloning your own repo:
```bash
cd ~/obstacle_avoidance_drone/src
# git clone <your-repo-url> .        # or copy the two package folders here
```
Layout expected:
```
~/obstacle_avoidance_drone/
  src/
    lightwarelidar2/          # from stage 8
    obstacle_avoidance/       # sim package
    obstacle_avoidance_hw/    # hardware package
```

Python deps for the live nodes are just rclpy/geometry_msgs/tf2 (already from ROS).
No numpy/opencv/torch needed for the current stack.

--------------------------------------------------------------------------------
## 10. Build the workspace [COMMON]
--------------------------------------------------------------------------------
```bash
cd ~/obstacle_avoidance_drone
source /opt/ros/$ROS_DISTRO/setup.bash

# resolve any declared package deps
rosdep install --from-paths src --ignore-src -r -y

# lightwarelidar2 is C++ and builds cleanly
colcon build --packages-select lightwarelidar2
```

⚠️ **setuptools caveat:** Ubuntu/pip may ship setuptools >= 80, which removed the
commands colcon uses to build **ament_python** packages (you'll see
`error: option --editable not recognized`). The launch files run our Python nodes
directly from source, so you can skip building them. If you *do* want to
`colcon build` `obstacle_avoidance*`:
```bash
pip3 install "setuptools<80"
colcon build
```

Source the overlay in every shell:
```bash
echo "source ~/obstacle_avoidance_drone/install/setup.bash" >> ~/.bashrc
source ~/obstacle_avoidance_drone/install/setup.bash
```

--------------------------------------------------------------------------------
## 11. Verify
--------------------------------------------------------------------------------
```bash
# ROS + Nav2 MPPI
ros2 pkg prefix nav2_mppi_controller && echo "nav2 OK"
# MAVROS
ros2 pkg prefix mavros && echo "mavros OK"
# Sim bridge
ros2 pkg prefix ros_gz_bridge && echo "ros_gz OK"
# Gazebo
gz sim --version
# ArduPilot SITL launcher
which sim_vehicle.py
# Lidar driver
ros2 pkg prefix lightwarelidar2 && echo "sf45b driver OK"
```

--------------------------------------------------------------------------------
## 12. Run
--------------------------------------------------------------------------------
### Simulation path
```bash
# terminal 1: Gazebo world
gz sim -v4 -r runway.sdf
# terminal 2: ArduPilot SITL (gazebo model)
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
# terminal 3: infra (clock+scan bridges, mavros, tf)
ros2 launch src/obstacle_avoidance/launch/bringup.launch.py
# arm -> GUIDED -> takeoff, then terminal 4: Nav2
ros2 launch src/obstacle_avoidance/launch/nav2_mppi.launch.py
# terminal 5: RViz + Nav2 Goal tool
rviz2 -d src/obstacle_avoidance/config/nav2_drone.rviz
```

### Hardware path (real SF45/B — Jetson OBC + real FCU)
```bash
# infra (real lidar + mavros + tf, wall-time, NO gz/clock).
# Real FCU: pass fcu_url for the serial link. Examples:
#   USB autopilot:   fcu_url:=/dev/ttyACM1:921600
#   Jetson UART:     fcu_url:=/dev/ttyTHS1:921600
ros2 launch src/obstacle_avoidance_hw/launch/bringup_hw.launch.py \
    fcu_url:=/dev/ttyACM1:921600 lidar_port:=/dev/ttyACM0

# after a position fix + takeoff (GPS/flow/VIO required):
ros2 launch src/obstacle_avoidance_hw/launch/nav2_hw.launch.py

# RViz — run on the GROUND STATION (same ROS_DOMAIN_ID), not the Jetson:
rviz2 -d src/obstacle_avoidance_hw/config/nav2_drone.rviz
```
On the bench with SITL instead of a real FCU, omit `fcu_url` (defaults to
`udp://127.0.0.1:14550@`) and run `sim_vehicle.py -v ArduCopter --console --map`.

⚠️ **Hardware needs a position estimate.** Without GPS (or optical-flow/VIO/mocap),
`/mavros/local_position/pose` never publishes, so `map->base_link` is absent and
Nav2 cannot run — and ArduPilot GUIDED velocity control won't work either. GPS is
useless indoors: use optical-flow + rangefinder or VIO there.

--------------------------------------------------------------------------------
## Version reference (known-good, as installed)
--------------------------------------------------------------------------------
Dev / simulation machine, as currently installed:

| Component            | Version           |
|----------------------|-------------------|
| Ubuntu               | 26.04 LTS         |
| ROS 2                | Lyrical           |
| Python               | 3.14.4            |
| Gazebo Sim           | 10.4.0            |
| nav2 (mppi)          | 1.5.1             |
| mavros / extras      | 2.14.0            |
| ros_gz_bridge        | 3.0.9             |
| rviz2                | 15.2.5            |
| lightwarelidar2      | source (main)     |

The Jetson OBC runs the Humble equivalents (Nav2 1.1.x), which is why the
parameter differences below exist.

--------------------------------------------------------------------------------
## Nav2 1.5 vs 1.1 parameters
--------------------------------------------------------------------------------
`src/obstacle_avoidance/config/nav2_params.yaml` and
`src/obstacle_avoidance_hw/config/nav2_params.yaml` are written for **Nav2 1.5**.
Three things changed from the 1.1 era, and each fails loudly rather than
silently on the wrong version:

| Setting | Nav2 1.5 (in this repo) | Nav2 1.1 (Humble) |
|---|---|---|
| Planner plugin | `nav2_navfn_planner::NavfnPlanner` | `nav2_navfn_planner/NavfnPlanner` |
| MPPI motion model | `motion_model: "omni"` **plus** an `omni:` block with `plugin: "mppi::OmniMotionModel"` | `motion_model: "Omni"` (plain string, no block) |
| `/cmd_vel` type | `TwistStamped` | `Twist` |

The `/cmd_vel` type is a code difference, not a config one: Nav2 switched in 1.4,
so `cmd_vel_to_mavros.py` and `cmdvel_to_send_ned.py` subscribe to
`TwistStamped`. On Humble those subscribers **receive nothing at all** — no
error, the drone simply never moves. Change them back to `Twist` (and drop the
`.twist` accessor) for a Humble OBC.

`nav2_hw.launch.py` also sets `service_timeout: 120.0` on the lifecycle manager,
harmless on 1.1 but only necessary on 1.5.

> `src/config/nav2_params.yaml` is a **stale unused copy** still in the 1.1 form.
> No launch file reads it — both load their own package's `config/`. Do not edit
> it expecting an effect.
