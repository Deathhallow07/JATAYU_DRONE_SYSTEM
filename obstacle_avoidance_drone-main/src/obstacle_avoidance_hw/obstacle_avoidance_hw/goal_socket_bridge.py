#!/usr/bin/env python3
"""
Goal socket bridge (runs on the OBC).

Receives navigation goals from a Ground Control Station over a UDP socket and
forwards them to Nav2 via the NavigateToPose action. The GCS sends GEOGRAPHIC
coordinates (lat/lon); this node converts them to the local `map` frame before
publishing the goal. Sends a small JSON status reply back to the sender
(converted / accepted / rejected / reached / aborted / canceled).

Run this SEPARATELY from the Nav2 launch, e.g.:
    python3 obstacle_avoidance_hw/goal_socket_bridge.py \
        --ros-args -p bind_port:=9200

lat/lon -> local conversion:
    Uses the drone's current global fix (/mavros/global_position/global) and
    current local pose (/mavros/local_position/pose). For a target (lat_t, lon_t):
        dNorth = (lat_t - lat_now) * M_PER_DEG
        dEast  = (lon_t - lon_now) * M_PER_DEG * cos(lat_now)
        goal_x (East)  = x_now + dEast
        goal_y (North) = y_now + dNorth
    (equirectangular / flat-earth — sub-metre accurate over the km ranges here).
    Requires a global fix AND local pose; otherwise the goal is rejected.

Wire protocol (JSON, UDP):
    GCS -> OBC : {"cmd": "goal", "lat": 12.9716, "lon": 77.5946, "yaw": 0.0}
                 {"cmd": "goal", "x": 5.0, "y": 2.0, "yaw": 0.0}   # local, optional
                 {"cmd": "cancel"}
    OBC -> GCS : {"status": "...", "msg": "..."}
"""

import json
import math
import socket

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from nav2_msgs.action import NavigateToPose

# WGS-84 metres per degree of latitude (R * pi/180, R = 6378137 m)
M_PER_DEG = 111319.4907932736


class GoalSocketBridge(Node):

    def __init__(self):
        super().__init__('goal_socket_bridge')

        self.declare_parameter('bind_host', '0.0.0.0')
        self.declare_parameter('bind_port', 9200)
        self.declare_parameter('goal_frame', 'map')
        # Throttle feedback status replies to the GCS (seconds); 0 disables.
        self.declare_parameter('feedback_period', 1.0)

        host = self.get_parameter('bind_host').value
        port = int(self.get_parameter('bind_port').value)
        self.goal_frame = self.get_parameter('goal_frame').value
        self.feedback_period = float(self.get_parameter('feedback_period').value)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self._goal_handle = None
        self._last_fb_t = 0.0

        # ---- current global fix + local pose (for lat/lon -> map conversion) ----
        self.cur_lat = None
        self.cur_lon = None
        self.cur_x = None
        self.cur_y = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            NavSatFix, '/mavros/global_position/global', self._global_cb, sensor_qos)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb, sensor_qos)

        # 20 Hz socket drain
        self.create_timer(0.05, self._poll_socket)

        self.get_logger().info(
            f"goal_socket_bridge listening on {host}:{port} "
            f"-> navigate_to_pose (frame '{self.goal_frame}'); expects lat/lon goals")

    # ---- mavros state for the geodetic conversion ----
    def _global_cb(self, msg: NavSatFix):
        # status.status == -1 (NO_FIX); require a fix and finite values
        if msg.status.status < 0:
            return
        if math.isfinite(msg.latitude) and math.isfinite(msg.longitude):
            self.cur_lat = msg.latitude
            self.cur_lon = msg.longitude

    def _pose_cb(self, msg: PoseStamped):
        self.cur_x = msg.pose.position.x
        self.cur_y = msg.pose.position.y

    def _latlon_to_local(self, lat, lon):
        """Target (lat,lon) -> (x_east, y_north) in the map frame, or None."""
        if None in (self.cur_lat, self.cur_lon, self.cur_x, self.cur_y):
            return None
        d_north = (lat - self.cur_lat) * M_PER_DEG
        d_east = (lon - self.cur_lon) * M_PER_DEG * math.cos(math.radians(self.cur_lat))
        return self.cur_x + d_east, self.cur_y + d_north

    # ---- socket helpers ----
    def _reply(self, addr, status, msg=''):
        if addr is None:
            return
        try:
            self.sock.sendto(
                json.dumps({'status': status, 'msg': msg}).encode(), addr)
        except OSError:
            pass

    def _poll_socket(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(2048)
            except (BlockingIOError, OSError):
                break

            try:
                d = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                self._reply(addr, 'rejected', 'malformed JSON')
                continue

            cmd = d.get('cmd', 'goal')
            if cmd == 'cancel':
                self._cancel(addr)
            elif cmd == 'goal':
                self._handle_goal(d, addr)
            else:
                self._reply(addr, 'rejected', f'unknown cmd: {cmd}')

    # ---- goal handling ----
    def _handle_goal(self, d, addr):
        try:
            yaw = float(d.get('yaw', 0.0))
        except (TypeError, ValueError):
            self._reply(addr, 'rejected', 'yaw must be numeric')
            return

        if 'lat' in d and 'lon' in d:
            try:
                lat = float(d['lat'])
                lon = float(d['lon'])
            except (TypeError, ValueError):
                self._reply(addr, 'rejected', 'lat/lon must be numeric')
                return
            local = self._latlon_to_local(lat, lon)
            if local is None:
                self._reply(addr, 'rejected',
                            'no global fix + local pose yet (need position source)')
                return
            x, y = local
            self._reply(addr, 'converted', f'lat/lon -> map x={x:.2f} y={y:.2f}')
        elif 'x' in d and 'y' in d:
            try:
                x = float(d['x'])
                y = float(d['y'])
            except (TypeError, ValueError):
                self._reply(addr, 'rejected', 'x/y must be numeric')
                return
        else:
            self._reply(addr, 'rejected', 'goal needs lat+lon (or x+y)')
            return

        frame = d.get('frame', self.goal_frame)

        if not self.client.server_is_ready():
            self.get_logger().warn('navigate_to_pose server not ready — is Nav2 up?')
            self._reply(addr, 'rejected', 'Nav2 action server not ready')
            return

        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = ps

        self.get_logger().info(
            f"GOAL from {addr[0]}:{addr[1]} -> "
            f"map x={x:.2f} y={y:.2f} yaw={yaw:.2f} frame={frame}")

        send_future = self.client.send_goal_async(
            goal, feedback_callback=lambda fb: self._on_feedback(fb, addr))
        send_future.add_done_callback(lambda f: self._on_goal_response(f, addr))

    def _on_goal_response(self, future, addr):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 rejected the goal')
            self._reply(addr, 'rejected', 'Nav2 rejected the goal')
            return
        self._goal_handle = goal_handle
        self._reply(addr, 'accepted', 'goal accepted')
        goal_handle.get_result_async().add_done_callback(
            lambda f: self._on_result(f, addr))

    def _on_result(self, future, addr):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('goal REACHED')
            self._reply(addr, 'reached', 'goal reached')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('goal canceled')
            self._reply(addr, 'canceled', 'goal canceled')
        else:
            self.get_logger().warn(f'goal ended (status={status})')
            self._reply(addr, 'aborted', f'navigation ended, status={status}')
        self._goal_handle = None

    def _on_feedback(self, feedback_msg, addr):
        if self.feedback_period <= 0.0:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._last_fb_t) < self.feedback_period:
            return
        self._last_fb_t = now
        dist = feedback_msg.feedback.distance_remaining
        self._reply(addr, 'feedback', f'distance_remaining={dist:.2f}m')

    def _cancel(self, addr):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self.get_logger().info('cancel requested')
            self._reply(addr, 'canceling', 'cancel requested')
        else:
            self._reply(addr, 'rejected', 'no active goal to cancel')


def main(args=None):
    rclpy.init(args=args)
    node = GoalSocketBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
