# Install — ROS 2 Humble (Ubuntu 22.04, Nav2 1.1)

Straight-through install for **Ubuntu 22.04**, ROS 2 Humble, Nav2 1.1 — the
onboard-computer path. Primary target is an **NVIDIA Jetson Orin NX 16 GB** on
JetPack 6.x (Ubuntu 22.04, arm64); the same steps work on x86_64 22.04 if you
want the simulator here too.

Humble is the only option on 22.04 — ROS 2 Lyrical needs 24.04 or newer. On the
dev machine, use [`INSTALL_LYRICAL.md`](INSTALL_LYRICAL.md) instead.

> ⚠️ **Read [§13 Porting to Nav2 1.1](#13-porting-the-repo-to-nav2-11) before you
> fly.** This repo is written against Nav2 1.5. Three things must change for
> Humble, and one of them fails **silently** — the drone simply never moves.
> Do the port as part of the install, not after the first confusing test flight.

Tags: **[COMMON]** always · **[SIM]** simulation only · **[HW]** real lidar/FCU only.
A headless Jetson OBC needs stages **0–3 and 8–13** only; skip 4–7.

```bash
export ROS_DISTRO=humble
```

Every apt line uses `$(dpkg --print-architecture)`, so it resolves to `arm64` on
the Jetson automatically.

---

## 0. Prerequisites [COMMON]

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget lsb-release gnupg build-essential python3-pip

sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### Jetson Orin NX notes

- JetPack 6 is Ubuntu 22.04, so Humble installs from apt on arm64 normally.
- **No CUDA or torch is needed.** Every live node is pure rclpy; nothing in this
  stack touches the Jetson GPU.
- Lock in max performance before flying — MPPI and the costmaps are real-time
  sensitive:
  ```bash
  sudo nvpmodel -m 0      # MAXN power mode
  sudo jetson_clocks      # pin clocks high
  ```
- Wire the flight controller by USB (`/dev/ttyACM*`) or the Jetson UART
  (`/dev/ttyTHS1`), and pass it as `fcu_url` in stage 12.

## 1. ROS 2 Humble [COMMON]

```bash
sudo add-apt-repository -y universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update

# Headless OBC — RViz runs on the ground station, not the Jetson:
sudo apt install -y ros-humble-ros-base ros-dev-tools
# Desktop / x86 dev box that also runs the simulator:
#   sudo apt install -y ros-humble-desktop ros-dev-tools

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash

sudo apt install -y python3-colcon-common-extensions python3-rosdep
sudo rosdep init 2>/dev/null || true
rosdep update
```

## 2. Nav2 1.1 + RViz [COMMON]

```bash
sudo apt install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-nav2-mppi-controller \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs

# Ground station / desktop only:
sudo apt install -y ros-humble-rviz2 ros-humble-nav2-rviz-plugins
```

Confirm the Nav2 version you actually got — the port in §13 is written against
1.1.x:

```bash
apt-cache policy ros-humble-navigation2 | head -2
```

## 3. MAVROS + GeographicLib datasets [COMMON]

```bash
sudo apt install -y ros-humble-mavros ros-humble-mavros-extras ros-humble-mavros-msgs

# REQUIRED, one time — MAVROS will not start cleanly without the geoid datasets
sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
```

## 4. Gazebo ↔ ROS bridge [SIM]

Humble's default apt pairing is **Gazebo Fortress**:

```bash
sudo apt install -y ros-humble-ros-gz
```

`ardupilot_gazebo` (stage 6) targets **Gazebo Harmonic**. If you want Harmonic
under Humble, add the OSRF repo from stage 5 first and install the vendored
bridge instead:

```bash
sudo apt install -y ros-humble-ros-gzharmonic
```

Pick one. Mixing the Fortress and Harmonic bridges in the same workspace gives
confusing library-load failures. Either way you get `ros_gz_bridge`, which this
stack uses for `/clock` and `/lidar/scan`.

> The Jetson does not need this stage at all — skip to stage 8.

## 5. Gazebo Sim Harmonic [SIM]

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

## 8. LightWare SF45/B driver [HW]

```bash
mkdir -p ~/obstacle_avoidance_drone/src
cd ~/obstacle_avoidance_drone/src
git clone https://github.com/LightWare-Optoelectronics/lightwarelidar2

sudo usermod -aG dialout $USER    # covers both the lidar and a USB FCU
```

Log out and back in for the group change to take effect.

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
    obstacle_avoidance_hw/    # hardware package — this is the one the OBC runs
```

## 10. Build [COMMON]

```bash
cd ~/obstacle_avoidance_drone
source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select lightwarelidar2

echo "source ~/obstacle_avoidance_drone/install/setup.bash" >> ~/.bashrc
source ~/obstacle_avoidance_drone/install/setup.bash
```

Humble ships an older setuptools, so the `--editable` breakage that affects the
Lyrical box usually does not appear here. If it does:
`pip3 install "setuptools<80"`. The launch files run the Python nodes from
source regardless, so building them is optional.

On a Jetson, `colcon build` on all cores can exhaust RAM. If the C++ driver
build gets OOM-killed, throttle it:

```bash
MAKEFLAGS="-j2" colcon build --packages-select lightwarelidar2 \
  --executor sequential
```

## 11. Verify

```bash
ros2 pkg prefix nav2_mppi_controller && echo "nav2 OK"
ros2 pkg prefix mavros             && echo "mavros OK"
ros2 pkg prefix lightwarelidar2    && echo "sf45b driver OK"
# simulation machines only:
ros2 pkg prefix ros_gz_bridge      && echo "ros_gz OK"
gz sim --version
which sim_vehicle.py
```

## 12. Run

### Hardware path — Jetson OBC + real FCU

```bash
# infra: real lidar + MAVROS + TF, wall-time clock, no gz bridge
ros2 launch src/obstacle_avoidance_hw/launch/bringup_hw.launch.py \
    fcu_url:=/dev/ttyACM1:921600 lidar_port:=/dev/ttyACM0
#   Jetson UART instead:  fcu_url:=/dev/ttyTHS1:921600

# after a position fix and takeoff:
ros2 launch src/obstacle_avoidance_hw/launch/nav2_hw.launch.py

# RViz on the GROUND STATION, same ROS_DOMAIN_ID — not on the Jetson:
rviz2 -d src/obstacle_avoidance_hw/config/nav2_drone.rviz
```

On the bench with SITL instead of a real FCU, omit `fcu_url` (it defaults to
`udp://127.0.0.1:14550@`) and run
`sim_vehicle.py -v ArduCopter --console --map`.

### Simulation path — x86 22.04 only

```bash
gz sim -v4 -r runway.sdf
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
ros2 launch src/obstacle_avoidance/launch/bringup.launch.py
ros2 launch src/obstacle_avoidance/launch/nav2_mppi.launch.py
rviz2 -d src/obstacle_avoidance/config/nav2_drone.rviz
```

`start_sim.sh` defaults to the Lyrical setup path — override it:

```bash
ROS_SETUP=/opt/ros/humble/setup.bash ./start_sim.sh
```

> ⚠️ **Hardware needs a position estimate.** Without GPS, optical-flow, VIO, or
> mocap, `/mavros/local_position/pose` never publishes — so `map→base_link` is
> absent, Nav2 cannot run, and ArduPilot GUIDED velocity control will not work
> either. GPS is useless indoors; use optical-flow + rangefinder or VIO there.

> **Give the velocity sender its own MAVLink port.** MAVROS holds 14550 for pose
> and telemetry; two processes on one UDP socket steal each other's packets.
> `cmdvel_to_send_ned.py` defaults to `udpin:0.0.0.0:14551`. For a real FCU,
> forward the serial link to a second UDP port with MAVProxy or mavlink-router
> and pass `mavlink_conn:=…` to `nav2_hw.launch.py`.

---

## 13. Porting the repo to Nav2 1.1

`src/obstacle_avoidance_hw/config/nav2_params.yaml` and the `/cmd_vel`
subscriber both target **Nav2 1.5**. Three changes are needed on Humble.

### 13a. Planner plugin name — fails loudly

Nav2 1.5 uses `::` separators for the planner plugin; 1.1 uses `/`.

```yaml
planner_server:
  ros__parameters:
    GridBased:
-     plugin: "nav2_navfn_planner::NavfnPlanner"
+     plugin: "nav2_navfn_planner/NavfnPlanner"
```

Symptom if missed: the planner server fails to load the plugin and never reaches
the `active` lifecycle state.

### 13b. MPPI motion model — fails loudly

In 1.5 the motion model is a pluginlib plugin, and `motion_model` names a
parameter namespace that must carry a `plugin` key. In 1.1 it is a plain string
enum, capitalised, with no block.

```yaml
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
-     motion_model: "omni"
-     omni:
-       plugin: "mppi::OmniMotionModel"
+     motion_model: "Omni"
```

Also check the progress checker while you are in this file — 1.1 takes a single
string, 1.3+ takes a list:

```yaml
-   progress_checker_plugins: ["progress_checker"]
+   progress_checker_plugin: "progress_checker"
```

Costmap layer plugins (`nav2_costmap_2d::ObstacleLayer`,
`nav2_costmap_2d::InflationLayer`) and the controller-side
`nav2_controller::SimpleProgressChecker` / `SimpleGoalChecker` keep their `::`
form on Humble — leave those alone.

### 13c. `/cmd_vel` message type — **fails silently**

This is the one that wastes an afternoon. Nav2 switched `/cmd_vel` from `Twist`
to `TwistStamped` in 1.4, so the nodes in this repo subscribe to `TwistStamped`.
On Humble, Nav2 publishes plain `Twist` — a `TwistStamped` subscriber receives
**nothing at all**. No error, no warning; Nav2 plans happily, RViz looks
correct, and the drone never moves.

In `src/obstacle_avoidance_hw/obstacle_avoidance_hw/cmdvel_to_send_ned.py`:

```python
-from geometry_msgs.msg import TwistStamped
+from geometry_msgs.msg import Twist

-self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_cb, 10)
+self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_cb, 10)

-def cmd_vel_cb(self, msg: TwistStamped):
-    self.vx       =  msg.twist.linear.x
-    self.vy       = -msg.twist.linear.y
-    self.vz       = -msg.twist.linear.z
-    self.yaw_rate = -msg.twist.angular.z
+def cmd_vel_cb(self, msg: Twist):
+    self.vx       =  msg.linear.x
+    self.vy       = -msg.linear.y
+    self.vz       = -msg.linear.z
+    self.yaw_rate = -msg.angular.z
```

Apply the identical change to
`src/obstacle_avoidance/obstacle_avoidance/cmd_vel_to_mavros.py` if you also run
the simulation package on Humble. The sign flips stay as they are — they convert
ROS ENU (CCW-positive yaw) to NED (CW-positive), which is unrelated to the
message type.

**Check it before flying:**

```bash
ros2 topic info /cmd_vel -v      # confirm the type, and that the sub is connected
ros2 topic hz /cmd_vel           # must tick while a goal is active
```

If `ros2 topic hz` shows traffic but the drone is still stationary, the
subscriber type is still mismatched.

### 13d. Harmless leftovers

`nav2_hw.launch.py` sets `service_timeout: 120.0` on the lifecycle manager. That
is only *necessary* on 1.5, but it is accepted and harmless on 1.1 — leave it.

> `src/config/nav2_params.yaml` is a **stale, unused copy** that is still in the
> 1.1 form. No launch file reads it — each package loads its own `config/`.
> Do not edit it expecting an effect, and do not mistake it for a ready-made
> Humble config.

### Summary

| Setting | Nav2 1.5 (repo default) | Nav2 1.1 (Humble) | Failure mode |
|---|---|---|---|
| Planner plugin | `nav2_navfn_planner::NavfnPlanner` | `nav2_navfn_planner/NavfnPlanner` | Loud — plugin load error |
| MPPI motion model | `motion_model: "omni"` + `omni:` block | `motion_model: "Omni"` | Loud — parameter error |
| Progress checker | `progress_checker_plugins: [...]` | `progress_checker_plugin: "..."` | Loud — parameter error |
| `/cmd_vel` type | `TwistStamped` | `Twist` | **Silent — drone never moves** |
