
#!/usr/bin/env python3
"""
Bridge Nav2's controller output to ArduPilot via MAVROS.

Nav2 (MPPI) publishes geometry_msgs/TwistStamped on /cmd_vel expressed in
the robot BODY frame (base_link / FLU): linear.x = forward, linear.y = left,
angular.z = yaw rate.  (Nav2 switched /cmd_vel from Twist to TwistStamped in
1.4; on 1.5 a Twist subscriber here silently receives nothing at all.)

MAVROS /setpoint_velocity/cmd_vel expects a TwistStamped whose linear
velocity is in the LOCAL ENU world frame (it converts ENU->NED for the FC).
So we rotate the body-frame (vx, vy) into world ENU using the vehicle's
current yaw (taken from /mavros/local_position/pose, which is in map/ENU).

Altitude is held (vz = 0): this is 2-D horizontal obstacle avoidance.

Requires the vehicle to be ARMED, in GUIDED, and already airborne for the
velocity setpoints to take effect.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped


class CmdVelToMavros(Node):

    def __init__(self):
        super().__init__('cmd_vel_to_mavros')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('setpoint_topic', '/mavros/setpoint_velocity/cmd_vel')
        # Republish the last command at this rate so ArduPilot's GUIDED
        # velocity failsafe (default ~3 s) never times out between Nav2 ticks.
        self.declare_parameter('publish_rate', 20.0)
        # Zero the output if no cmd_vel arrives for this long (safety stop).
        self.declare_parameter('cmd_timeout', 0.5)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        setpoint_topic = self.get_parameter('setpoint_topic').value
        rate = float(self.get_parameter('publish_rate').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)

        self.yaw = 0.0
        self.last_cmd = Twist()   # the .twist payload of the last TwistStamped
        self.last_cmd_t = 0.0

        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(TwistStamped, cmd_vel_topic, self.cmd_cb, 10)
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.pose_cb, mavros_qos)

        self.pub = self.create_publisher(TwistStamped, setpoint_topic, 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f"cmd_vel_to_mavros: {cmd_vel_topic} (body) -> {setpoint_topic} (ENU)")

    def cmd_cb(self, msg: TwistStamped):
        self.last_cmd = msg.twist
        self.last_cmd_t = self.now_s()

    def pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        # yaw from quaternion (ENU / map frame)
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def tick(self):
        cmd = self.last_cmd
        stale = (self.now_s() - self.last_cmd_t) > self.cmd_timeout
        if stale:
            # No active Nav2 command -> publish NOTHING. Streaming a zero
            # velocity setpoint in GUIDED means "hold", which would override
            # (block) takeoff and any other GUIDED command while idle. Staying
            # silent lets takeoff / manual GUIDED work; ArduPilot only follows
            # our setpoints while Nav2 is actively navigating a goal.
            return

        vx_b = cmd.linear.x
        vy_b = cmd.linear.y
        wz = cmd.angular.z

        # body FLU -> world ENU rotation by yaw
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vx_w = c * vx_b - s * vy_b
        vy_w = s * vx_b + c * vy_b

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'
        out.twist.linear.x = vx_w
        out.twist.linear.y = vy_w
        out.twist.linear.z = 0.0        # hold altitude
        out.twist.angular.z = wz
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToMavros()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
