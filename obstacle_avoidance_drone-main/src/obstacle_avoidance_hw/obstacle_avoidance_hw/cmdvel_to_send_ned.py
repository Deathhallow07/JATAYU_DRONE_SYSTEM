#!/usr/bin/env python3
"""
Nav2 /cmd_vel -> ArduPilot velocity via pymavlink SET_POSITION_TARGET_LOCAL_NED.

Replaces cmd_vel_to_mavros.py (which went through mavros /setpoint_velocity).
This talks MAVLink directly and sends velocity in the BODY_NED frame, so
ArduPilot does the body->world rotation for us (no yaw math needed here).

Nav2 publishes /cmd_vel as body FLU (x fwd, y left, z up, angular.z CCW+).
BODY_NED wants FRD (x fwd, y right, z down, yaw_rate CW+), hence the sign flips.

NOTE ON THE CONNECTION: mavros is still running for pose/telemetry (typically on
udp 14550). This node needs its OWN MAVLink stream, so point `connection` at a
DIFFERENT endpoint (e.g. udpin:0.0.0.0:14551 that SITL/MAVProxy or mavlink-router
forwards to) to avoid clashing with mavros on 14550.

Adapted from the user's drone_controller/cmdvel_to_send_ned.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from pymavlink import mavutil


class CmdVelToMavlink(Node):
    def __init__(self):
        super().__init__('cmdvel_to_send_ned')

        self.declare_parameter('connection', 'udpin:0.0.0.0:14551')
        self.declare_parameter('rate_hz', 10.0)
        conn_str = self.get_parameter('connection').value
        rate = self.get_parameter('rate_hz').value

        self.master = mavutil.mavlink_connection(conn_str)
        self.get_logger().info(f'Waiting for heartbeat on {conn_str} ...')
        self.master.wait_heartbeat()
        self.get_logger().info(
            f'Heartbeat: system {self.master.target_system}, '
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

        # Nav2 publishes TwistStamped on /cmd_vel since 1.4; a Twist
        # subscriber here receives nothing at all on 1.5.
        self.create_subscription(TwistStamped, 'cmd_vel', self.cmd_vel_cb, 10)
        self.create_timer(1.0 / rate, self.send_velocity)

    def cmd_vel_cb(self, msg: TwistStamped):
        self.vx = msg.twist.linear.x
        self.vy = -msg.twist.linear.y
        self.vz = -msg.twist.linear.z
        self.yaw_rate = -msg.twist.angular.z   # CCW(+) in ROS -> CW(+) in NED

    def send_velocity(self):
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
