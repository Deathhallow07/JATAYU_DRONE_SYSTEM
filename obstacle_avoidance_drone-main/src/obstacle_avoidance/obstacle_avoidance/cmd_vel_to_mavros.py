#!/usr/bin/env python3
"""
Nav2 /cmd_vel -> ArduPilot velocity via pymavlink SET_POSITION_TARGET_LOCAL_NED.

This is the "send_ned" path: it talks MAVLink directly instead of going through
mavros' /setpoint_velocity, and sends velocity in the BODY_NED frame so
ArduPilot does the body->world rotation itself (no yaw math needed here).

Nav2 publishes /cmd_vel as body FLU (x fwd, y left, z up, angular.z CCW+).
BODY_NED wants FRD (x fwd, y right, z down, yaw_rate CW+), hence the sign flips.

CONNECTION: mavros owns udp 14550 for pose/telemetry. This node needs its OWN
MAVLink stream, so start_sim.sh gives SITL an extra `--out=udp:127.0.0.1:14555`
and we listen there. Pointing both at 14550 makes them steal each other's
packets.

TAKEOFF GATE: setpoints are only emitted while Nav2 is actually publishing
cmd_vel (see `hold_timeout`). This matters -- a stream of zero-velocity
SET_POSITION_TARGET messages arriving during MAV_CMD_NAV_TAKEOFF cancels the
climb, because ArduPilot treats any guided velocity target as a new command
that supersedes the takeoff. Staying silent until the controller has something
to say lets takeoff finish, and going silent again ~1 s after the last cmd_vel
lets ArduPilot's own guided timeout brake and hold position at the goal.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from pymavlink import mavutil


class CmdVelToMavlink(Node):
    def __init__(self):
        super().__init__('cmdvel_to_send_ned')

        self.declare_parameter('connection', 'udpin:0.0.0.0:14555')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('hold_timeout', 1.0)
        conn_str = self.get_parameter('connection').value
        rate = self.get_parameter('rate_hz').value
        self.hold_timeout = self.get_parameter('hold_timeout').value

        self.master = mavutil.mavlink_connection(conn_str)
        self.get_logger().info(f'waiting for heartbeat on {conn_str} ...')
        self.master.wait_heartbeat()
        self.get_logger().info(
            f'heartbeat: system {self.master.target_system}, '
            f'component {self.master.target_component}')

        # Use only velocity (vx, vy, vz) + yaw_rate; ignore everything else.
        self.type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self.vx = self.vy = self.vz = self.yaw_rate = 0.0
        self.last_cmd = None          # wall-clock seconds of last /cmd_vel
        self.streaming = False        # for one-shot log lines

        # Nav2 1.4+ publishes TwistStamped on /cmd_vel and nothing else; a
        # plain Twist subscriber here receives literally nothing on 1.5.
        self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_cb, 10)
        self.create_timer(1.0 / rate, self.send_velocity)
        self.get_logger().info(
            'send_ned bridge ready (idle until Nav2 publishes /cmd_vel)')

    def _now(self):
        # Wall clock on purpose: this node runs on system time, not sim time.
        return self.get_clock().now().nanoseconds / 1e9

    def cmd_vel_cb(self, msg: TwistStamped):
        self.vx = msg.twist.linear.x
        self.vy = -msg.twist.linear.y
        self.vz = -msg.twist.linear.z
        self.yaw_rate = -msg.twist.angular.z   # CCW(+) in ROS -> CW(+) in NED
        self.last_cmd = self._now()
        if not self.streaming:
            self.streaming = True
            self.get_logger().info('cmd_vel seen -> streaming setpoints')

    def send_velocity(self):
        if self.last_cmd is None:
            return
        if self._now() - self.last_cmd > self.hold_timeout:
            if self.streaming:
                self.streaming = False
                self.vx = self.vy = self.vz = self.yaw_rate = 0.0
                self.get_logger().info(
                    'cmd_vel stale -> setpoints off (ArduPilot holds position)')
            return

        self.master.mav.set_position_target_local_ned_send(
            0,                                    # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,   # velocity in body frame
            self.type_mask,
            0.0, 0.0, 0.0,                        # x, y, z position (ignored)
            self.vx, self.vy, self.vz,            # velocity, m/s
            0.0, 0.0, 0.0,                        # acceleration (ignored)
            0.0, self.yaw_rate)                   # yaw (ignored), yaw_rate rad/s


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToMavlink()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
