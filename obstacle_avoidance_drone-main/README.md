# obstacle_avoidance_drone

Mapless **Nav2 (MPPI) + 2D-lidar** obstacle avoidance for an ArduPilot multirotor,
with two parallel setups:

| Package | Use | Lidar | Clock |
|---|---|---|---|
| `obstacle_avoidance`    | **Simulation** (Gazebo + ArduPilot SITL) | Gazebo lidar → `/lidar/scan` | sim time (`/clock`) |
| `obstacle_avoidance_hw` | **Hardware / OBC** (real LightWare SF45/B) | `lightwarelidar2` → `/lidar/scan` | wall time |

Nav2 runs mapless: `map→base_link` comes from `/mavros/local_position/pose`
(flattened to 2D), both costmaps are rolling windows off `/lidar/scan`, and MPPI's
`/cmd_vel` is sent to ArduPilot as **BODY_NED velocity via pymavlink**
(`SET_POSITION_TARGET_LOCAL_NED`, node `cmdvel_to_send_ned`) — **not** through mavros.

> The velocity sender uses its **own** MAVLink endpoint, separate from mavros
> (14550, used for pose/telemetry) — two processes on one UDP socket steal each
> other's packets. Hardware (`cmdvel_to_send_ned.py`) defaults to
> `udpin:0.0.0.0:14551`; simulation (`cmd_vel_to_mavros.py`) defaults to
> `udpin:0.0.0.0:14555`, which `start_sim.sh` gives SITL via `--out`. For a real
> FCU, use MAVProxy / mavlink-router to forward the link to a second UDP port.
> Override with `mavlink_conn:=...` on `nav2_hw.launch.py`.
>
> Both senders stay **silent until Nav2 actually publishes `/cmd_vel`**, and go
> silent again ~1 s after the last one. A stream of zero-velocity setpoints during
> `MAV_CMD_NAV_TAKEOFF` cancels the climb, because ArduPilot treats any guided
> velocity target as a new command that supersedes the takeoff.

## Run (simulation) — one command

Targets **ROS 2 Lyrical (Nav2 1.5)** on Ubuntu 26.04; override with
`ROS_SETUP=/opt/ros/<distro>/setup.bash` if yours differs.

```bash
./start_sim.sh          # start everything and attach to the tmux session
./start_sim.sh stop     # kill the session and every sim process
./start_sim.sh status   # show what is running
```

That brings up Gazebo, ArduPilot SITL, the gz→ROS bridges, mavros, the TF
bridge, Nav2 and RViz in six tmux panes, and **gets the drone airborne**: pane 5
waits for EKF/GPS lock, switches to GUIDED, arms and climbs to 10 m, then leaves
you a shell there. **Goals are yours to send:**

```bash
python3 lawnmower.py                  # the full survey pattern
python3 send_nav_goal.py 0 45         # or a single goal: x=east, y=north
python3 send_nav_goal.py --cancel
```

### Lawnmower survey with skip-on-unreachable

`lawnmower.py` flies a boustrophedon pattern, one `NavigateToPose` goal per
corner. **If a goal aborts, that waypoint is logged SKIPPED and the mission
moves to the next one** rather than stopping or retrying forever.

```bash
python3 lawnmower.py --dry-run                    # print waypoints, fly nothing
python3 lawnmower.py --lanes -30,0,30 --y0 5 --y1 45
```

`worlds/lawnmower.sdf` parks buildings on two corners deliberately, so the skip
path is exercised every run:

| Waypoint | Obstacle | Failure mode |
|---|---|---|
| `(-15, 45)` | `corner_nw` sits on top of it | goal inside an obstacle |
| `(15, 5)` | `courtyard_se` walls it in | goal free, but no route |

Two more (`block_west` at `(-30, 25)` and the original `building` at `(0, 25)`)
sit mid-lane and are passable — those legs should route around and still reach.
The new buildings are 14 m tall rather than 10 m so the lidar gets a solid
return at the 10 m cruise altitude instead of grazing the roofline.

> **This depends on the custom behaviour tree**,
> `src/obstacle_avoidance/config/navigate_to_pose_no_global_clear.xml`, wired in
> by `nav2_mppi.launch.py`. Nav2's stock tree answers "no valid path" by
> clearing the global costmap — which deletes the obstacle proving the goal is
> unreachable, so the next plan succeeds straight through the building and the
> cycle repeats. It never aborts either: `RecoveryNode` resets its retry counter
> whenever its child succeeds, and the wipe manufactures exactly that success.
> Our tree lets the planning failure propagate. Run with the stock tree to see
> the difference:
> ```bash
> NAV_BT_XML=/opt/ros/lyrical/share/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml ./start_sim.sh
> ```

`map` is the drone's ENU local frame with its origin at the spawn point, so goal
coordinates are just metres east/north of takeoff. RViz's **Nav2 Goal** tool
works too.

Keep goals inside the global costmap, which spans ±50 m from takeoff; a goal
exactly on the edge is rejected as *"outside bounds"*.

Knobs (environment variables):

```bash
TAKEOFF_ALT=15 ./start_sim.sh
AUTO_TAKEOFF=0 ./start_sim.sh      # bring the stack up but don't arm or fly
```

`auto_takeoff.py` is standalone too (pymavlink, no ROS), so you can re-run it
from any pane:

```bash
python3 auto_takeoff.py --alt 15
```

> `tf_publisher.py` publishes **`/odom`** as well as the TF, restamped onto the
> sim clock. MPPI's `OdomSmoother` reads it for the drone's current speed, and
> the motion model only lets it command a couple of acceleration steps away
> from that speed per cycle. With no `/odom` it assumes the drone is stationary
> every cycle, pins `/cmd_vel` to `~2 * ax_max * model_dt` (0.3 m/s) whatever
> `vx_max` says, and can crawl slowly enough that `SimpleProgressChecker`
> aborts the goal with *"Failed to make progress"*. mavros' own
> `/mavros/local_position/odom` is wall-clock stamped and gets discarded.

## Comparison arm: ArduPilot's own avoidance (BendyRuler)

`./start_sim_oa.sh` flies the **same world and the same waypoints** with the
avoidance moved *into the flight controller* — no Nav2, no mavros, no costmaps,
no TF, no RViz. The only ROS process is the `ros_gz` bridge pulling
`/lidar/scan` out of Gazebo; `mavlik_bridge.py` turns that scan into MAVLink
`OBSTACLE_DISTANCE` so ArduPilot sees the identical buildings, and planning
happens there via `OA_TYPE=1` (BendyRuler).

```bash
./start_sim_oa.sh          # start everything and attach (tmux session "simoa")
./start_sim_oa.sh stop     # kill the session and every sim process
./start_sim_oa.sh status   # show what is running

python3 oa_lawnmower.py            # fly the survey under BendyRuler
python3 oa_lawnmower.py --dry-run  # print waypoints, fly nothing
```

Both arms share `pattern.py`, which owns the survey geometry — same lanes, same
corners, no datum conversion in between. **Keep it in step with
`worlds/lawnmower.sdf`**: the buildings are placed against those exact numbers,
and changing one without the other makes the comparison meaningless.

### Parameter files

| File | Role |
|---|---|
| `oa.parm` | The avoidance tuning itself: `PRX1_*` proximity, `OA_*` BendyRuler, `AVOID_*` simple avoidance. Loaded **last** so it layers on top of the frame config. |
| `oa_compare.parm` | Comparison-only overlay — matches `WP_SPD` to Nav2's 1 m/s and unclamps the proximity filter. Without it the vehicle flies the course at the 10 m/s default and hits the first building. |

`oa.parm` enables **two independent layers on purpose**: `OA_*` (BendyRuler) only
runs in AUTO/GUIDED/RTL, while `AVOID_*` (stop/slide) only runs in the
pilot-flown modes. Enabling one leaves the other set of modes unprotected.

> ⚠️ **Separate EEPROM, and it matters.** `sim_vehicle.py` persists parameters to
> `eeprom.bin` in its working directory, so this arm runs out of `.sitl_oa/` with
> its own. Were it to load `oa.parm` into the directory the Nav2 stack uses,
> `OA_TYPE`/`PRX1_TYPE` would still be set on the *next* Nav2 run and Nav2 would
> be flying on top of an ArduPilot avoidance layer with nobody noticing.

> **External dependency:** `mavlik_bridge.py` is not vendored here —
> `start_sim_oa.sh` expects it at `$HOME/mavlik_bridge.py`, overridable with
> `MAV_BRIDGE=...`.

### Reading the two summaries

The runs do **not** report the same way, and the difference is not incidental.
ArduPilot never says "this destination is unreachable" — BendyRuler just keeps
steering around forever — so `oa_lawnmower.py` advances on the `goto_line.py`
rule: once avoidance has carried the drone onto the *next* leg's line, it
retargets. That means it **cannot distinguish "flew around it and carried on"
from "could never get there"**; both simply advance. `lawnmower.py` on the Nav2
side gets an explicit abort and logs the waypoint SKIPPED. Compare with that in
mind: the Nav2 run classifies, this one only progresses.

## Onboard computer (no Gazebo needed)

The OBC only runs the **`obstacle_avoidance_hw`** path — it does **not** need Gazebo
or ardupilot_gazebo. Install just the common + hardware pieces from
[`INSTALL.md`](INSTALL.md) (skip stages 4–7, which are simulation-only).

> ⚠️ **The OBC is on a different ROS release than the dev box.** JetPack 6 is
> Ubuntu 22.04, so the Jetson runs **Humble / Nav2 1.1**, while simulation here
> runs **Lyrical / Nav2 1.5**. The `nav2_params.yaml` in this repo is written for
> 1.5 and will not load on 1.1, and the `TwistStamped` `/cmd_vel` subscribers
> receive **nothing at all** on Humble — silently, so the drone just never moves.
> See [Nav2 1.5 vs 1.1 parameters](INSTALL.md#nav2-15-vs-11-parameters) for the
> three changes needed.

```bash
# 1. workspace
mkdir -p ~/obstacle_avoidance_drone/src && cd ~/obstacle_avoidance_drone/src

# 2. this repo
git clone <THIS_REPO_URL> .

# 3. the LightWare driver (third-party, not vendored here)
git clone https://github.com/LightWare-Optoelectronics/lightwarelidar2

# 4. build the C++ driver (our python nodes run from source, no build needed)
cd ~/obstacle_avoidance_drone
source /opt/ros/humble/setup.bash        # Humble on JetPack 6; the dev box is Lyrical
colcon build --packages-select lightwarelidar2
source install/setup.bash

# 5. serial access for the SF45/B (log out/in after)
sudo usermod -aG dialout $USER
```

### Run (hardware)
```bash
# flight controller must be connected with a POSITION SOURCE (GPS / flow / VIO)
ros2 launch src/obstacle_avoidance_hw/launch/bringup_hw.launch.py   # SF45/B + mavros + tf
# after position fix + takeoff:
ros2 launch src/obstacle_avoidance_hw/launch/nav2_hw.launch.py      # Nav2 MPPI + cmd_vel bridge
rviz2 -d src/obstacle_avoidance_hw/config/nav2_drone.rviz           # send goals with the Nav2 Goal tool
```

> ⚠️ Nav2 and ArduPilot GUIDED both require a position estimate. Without GPS (or
> optical-flow/VIO/mocap) `/mavros/local_position/pose` never publishes, so
> `map→base_link` is absent and nothing navigates. GPS is useless indoors.

## Sending goals from a GCS (socket bridge)

Give goals from a Ground Control Station over UDP instead of RViz. Two pieces:

- **OBC** — `goal_socket_bridge` (run it **separately** from the Nav2 launch): receives
  goals on a UDP port and forwards them to Nav2's `navigate_to_pose` action, replying
  with status (accepted / reached / aborted).
  ```bash
  python3 src/obstacle_avoidance_hw/obstacle_avoidance_hw/goal_socket_bridge.py \
      --ros-args -p bind_port:=9200
  ```
  The bridge converts the incoming **lat/lon** to the local `map` frame using the
  drone's current global fix + local pose (needs a position source).
- **GCS** — `gcs_goal_sender.py` (standalone, Python stdlib only, **no ROS needed**).
  Copy this one file to the GCS.
  ```bash
  # lat lon [yaw]  (decimal degrees, yaw radians); --host = OBC IP
  ./scripts/gcs_goal_sender.py 12.9716 77.5946 --host <OBC_IP> --port 9200
  ./scripts/gcs_goal_sender.py --cancel --host <OBC_IP>       # cancel current goal
  ./scripts/gcs_goal_sender.py --xy 5 2 --host <OBC_IP>       # (advanced) local x/y goal
  ```

Protocol (JSON/UDP): `{"cmd":"goal","lat":..,"lon":..,"yaw":..}` (or `"x"/"y"` for a
local goal) or `{"cmd":"cancel"}`; replies
`{"status":"converted|accepted|reached|aborted|rejected|...","msg":..}`.
Open UDP `9200` on the OBC firewall if the GCS is on another machine.

## Full setup
[`INSTALL.md`](INSTALL.md) is the complete from-scratch install covering both
platforms, with exact download/build commands and a known-good version table.
For a straight-through guide with nothing to substitute, pick your distro:

| Guide | Platform |
|---|---|
| [`INSTALL_LYRICAL.md`](INSTALL_LYRICAL.md) | Ubuntu 26.04 · ROS 2 Lyrical · Nav2 1.5 — dev / simulation box, runs this repo as-is |
| [`INSTALL_HUMBLE.md`](INSTALL_HUMBLE.md) | Ubuntu 22.04 · ROS 2 Humble · Nav2 1.1 — Jetson Orin NX OBC, **includes the Nav2 1.1 port patches** |

The Nav2 config in this repo is **not portable** between the two — see the
porting section in the Humble guide before deploying to the Jetson.

## Key parameters
- Costmaps / MPPI critics: `src/obstacle_avoidance_hw/config/nav2_params.yaml`
- ArduPilot-side avoidance (comparison arm): `oa.parm` — `OA_BR_LOOKAHEAD`,
  `OA_MARGIN_MAX`, `AVOID_MARGIN`, `PRX_FILT`
- Standoff distance: `inflation_radius` + `cost_scaling_factor` + `ObstaclesCritic.repulsion_weight`
- Lidar port/baud/FOV: launch args in `bringup_hw.launch.py`
  (`lidar_port:=/dev/ttyACM0`, `lidar_baud:=115200`, `low_angle`/`high_angle`)

## Note on building the Python packages
`obstacle_avoidance*` are `ament_python` packages. With setuptools ≥ 80 `colcon build`
fails (`--editable not recognized`); the launch files therefore run the Python nodes
directly from source, so **building them is optional**. To build anyway:
`pip install "setuptools<80"` first.
