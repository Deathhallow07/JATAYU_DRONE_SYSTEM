#!/usr/bin/env python3
"""
Live readiness dashboard for the obstacle-avoidance stack (pane 4).

Repaints a compact status table ~1 Hz so you can see, at a glance, which parts
of the stack are actually up and what is still missing before a goal will fly:

    MAVROS link / mode / armed      <- /mavros/state
    GPS fix + satellites            <- /mavros/global_position/raw/fix
    local pose                      <- /mavros/local_position/pose
    lidar rate                      <- /lidar/scan
    map->base_link TF               <- tf2 (published by tf_publisher.py)
    navigate_to_pose action         <- Nav2 bt_navigator
    goal bridge UDP 9200            <- goal_socket_bridge

The bottom line is the one that matters: GOAL READY tells you whether
send_goal.py would be accepted right now, and names what is missing if not.
The bridge needs a global fix AND a local pose to convert lat/lon into the map
frame, and Nav2 needs the TF to plan.
"""

import socket
import subprocess
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan, NavSatFix

RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
DIM = '\033[2m'

# NavSatStatus.status -> label
FIX_LABELS = {-1: 'NO_FIX', 0: 'FIX', 1: 'SBAS', 2: 'GBAS'}


def ok(text):
    return f'{GREEN}{text}{RESET}'


def bad(text):
    return f'{RED}{text}{RESET}'


def warn(text):
    return f'{YELLOW}{text}{RESET}'


class StackMonitor(Node):

    def __init__(self):
        super().__init__('stack_monitor')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.state = None
        self.fix = None
        self.pose = None
        self.scan_count = 0
        self.scan_times = []
        self.port_open = False
        self._last_port_check = 0.0

        self.create_subscription(State, '/mavros/state', self._state_cb, sensor_qos)
        self.create_subscription(
            NavSatFix, '/mavros/global_position/raw/fix', self._fix_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb, sensor_qos)
        self.create_subscription(LaserScan, '/lidar/scan', self._scan_cb, sensor_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_timer(1.0, self._render)

    # ---- subscriptions ----
    def _state_cb(self, msg):
        self.state = msg

    def _fix_cb(self, msg):
        self.fix = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def _scan_cb(self, msg):
        self.scan_count += 1
        now = time.time()
        self.scan_times.append(now)
        # keep a ~5 s window for the rate estimate
        self.scan_times = [t for t in self.scan_times if now - t < 5.0]

    # ---- derived state ----
    def _scan_hz(self):
        if len(self.scan_times) < 2:
            return 0.0
        span = self.scan_times[-1] - self.scan_times[0]
        return (len(self.scan_times) - 1) / span if span > 0 else 0.0

    def _tf_ok(self):
        try:
            return self.tf_buffer.can_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception:
            return False

    def _bridge_port_open(self):
        # ss is cheap but not free; poll it every 2 s rather than every frame.
        now = time.time()
        if now - self._last_port_check > 2.0:
            self._last_port_check = now
            try:
                out = subprocess.run(
                    ['ss', '-lun'], capture_output=True, text=True, timeout=2).stdout
                self.port_open = ':9200' in out
            except (OSError, subprocess.SubprocessError):
                self.port_open = False
        return self.port_open

    # ---- rendering ----
    def _render(self):
        lines = []
        add = lines.append

        add(f'{BOLD}STACK STATUS{RESET}  {DIM}{time.strftime("%H:%M:%S")}{RESET}')
        add('─' * 46)

        # MAVROS
        if self.state is None:
            add(f'MAVROS   {bad("no /mavros/state")}')
            link = False
        else:
            link = self.state.connected
            armed = 'ARMED' if self.state.armed else 'disarmed'
            add(f'MAVROS   {ok("connected") if link else bad("NOT connected")}  '
                f'{self.state.mode}  {armed}')

        # GPS
        if self.fix is None:
            add(f'GPS      {bad("no fix msgs")}')
            has_fix = False
        else:
            st = self.fix.status.status
            has_fix = st >= 0
            label = FIX_LABELS.get(st, str(st))
            txt = ok(label) if has_fix else bad(label)
            add(f'GPS      {txt}  lat {self.fix.latitude:.6f}')

        # local pose
        if self.pose is None:
            add(f'POSE     {bad("no local_position/pose")}')
            has_pose = False
        else:
            has_pose = True
            p = self.pose.pose.position
            add(f'POSE     {ok("ok")}  x {p.x:.1f}  y {p.y:.1f}  z {p.z:.1f}')

        # lidar
        hz = self._scan_hz()
        if self.scan_count == 0:
            add(f'LIDAR    {bad("no scans")}')
        else:
            txt = ok(f'{hz:.1f} Hz') if hz > 1.0 else warn(f'{hz:.1f} Hz')
            add(f'LIDAR    {txt}  ({self.scan_count} scans)')

        # TF
        tf_ok = self._tf_ok()
        add(f'TF       map->base_link {ok("ok") if tf_ok else bad("MISSING")}')

        # Nav2
        nav_ok = self.nav_client.server_is_ready()
        add(f'NAV2     navigate_to_pose '
            f'{ok("ready") if nav_ok else bad("not ready")}')

        # goal bridge
        bridge_ok = self._bridge_port_open()
        add(f'BRIDGE   udp 9200 '
            f'{ok("listening") if bridge_ok else bad("not listening")}')

        add('─' * 46)

        missing = []
        if not link:
            missing.append('MAVROS link')
        if not has_fix:
            missing.append('GPS fix')
        if not has_pose:
            missing.append('local pose')
        if not tf_ok:
            missing.append('TF')
        if not nav_ok:
            missing.append('Nav2')
        if not bridge_ok:
            missing.append('bridge')

        if missing:
            add(f'GOAL READY {bad("NO")} {DIM}- need: ' + ', '.join(missing) + RESET)
        else:
            add(f'GOAL READY {ok("YES")} {DIM}- arm + takeoff, then send_goal.py{RESET}')

        # Repaint in place: home the cursor and clear each line as we go, so the
        # pane does not scroll (no flicker from a full clear).
        out = '\033[H' + '\n'.join(line + '\033[K' for line in lines) + '\033[J'
        print(out, end='', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = StackMonitor()
    print('\033[2J', end='')  # one full clear at startup
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl-C, or SIGTERM from `start_stack.sh stop` — exit quietly rather
        # than dumping a traceback over the last status frame.
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
