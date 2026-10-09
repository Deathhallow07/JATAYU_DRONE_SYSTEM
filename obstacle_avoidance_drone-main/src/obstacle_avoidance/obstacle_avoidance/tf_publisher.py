#!/usr/bin/env python3

import math

import rclpy

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped

from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
class MavrosTFBridge(Node):

    def __init__(self):
        super().__init__('mavros_tf_bridge')

        # -------- Parameters --------

        self.declare_parameter(
            'lidar_frame',
            'iris_with_gimbal/lidar_link/lidar_sensor'
        )

        self.lidar_frame = self.get_parameter(
            'lidar_frame'
        ).value

        mavros_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=20
                )

        # -------- TF Broadcasters --------

        self.tf_broadcaster = TransformBroadcaster(self)

        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Publish static transform once
        self.publish_static_tf()

        # -------- Subscriber --------

        self.subscription = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            mavros_qos
        )

        # -------- /odom for Nav2 --------
        # MPPI's OdomSmoother subscribes to `odom` (nav_msgs/Odometry) to learn
        # the robot's CURRENT speed, and the motion model only lets it command
        # a couple of acceleration steps away from that speed per cycle
        # (ax_max * model_dt). With no publisher it logs "OdomSmoother has not
        # received any data yet, returning empty Twist" and believes the drone
        # is stationary on every single cycle -- so /cmd_vel is pinned at
        # ~2 * ax_max * model_dt (0.30 m/s here) no matter what vx_max says,
        # and from a standing start the drone can crawl slowly enough that
        # SimpleProgressChecker aborts the goal with "Failed to make progress".
        #
        # mavros already publishes /mavros/local_position/odom, but stamped in
        # WALL clock while Nav2 runs on sim time, so OdomSmoother would throw
        # every sample away. Republish here instead, on the sim clock, for the
        # same reason the TF above is restamped.
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.last_twist = TwistStamped()
        self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_body',
            self.twist_callback, mavros_qos)

        self.get_logger().info("MAVROS TF Bridge Started")

    def twist_callback(self, msg):
        self.last_twist = msg

    def publish_static_tf(self):

        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()

        tf.header.frame_id = "base_link"

        tf.child_frame_id = self.lidar_frame

        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0

        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(tf)

    def pose_callback(self, msg):

        tf = TransformStamped()

        # Stamp with the node's (sim) clock, NOT msg.header.stamp: MAVROS
        # publishes pose stamps in wall-clock time while the Gazebo lidar/scan
        # and Nav2 run on sim time. Copying the wall-clock stamp makes
        # map->base_link unusable for transforming the sim-time scan
        # ("timestamp earlier than all data in the transform cache"), so the
        # scan can't be shown/used in the map frame. Using the sim clock here
        # keeps the whole TF tree on the same clock as the sensor data.
        tf.header.stamp = self.get_clock().now().to_msg()

        tf.header.frame_id = "map"

        tf.child_frame_id = "base_link"

        # Flatten the drone into Nav2's 2D plane. Nav2's costmaps, planner,
        # goal and paths all live at z=0. If base_link is published at the
        # true flight altitude, the horizontal lidar's returns land at
        # z=altitude in the map frame and the costmap's obstacle-height filter
        # drops them -- obstacles only appear once the drone descends into the
        # z~0 band. Forcing z=0 keeps base_link (and the lidar riding on it) in
        # the same plane as the costmap, so hits are marked at any altitude,
        # and RViz stays coherent (scan, costmap, goal all at z=0). Real
        # altitude is held independently by ArduPilot.
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = 0.0

        # Keep yaw only (zero roll/pitch) so the 2D scan stays horizontal in
        # the map frame even while the drone pitches/rolls to accelerate.
        q = msg.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(yaw / 2.0)
        tf.transform.rotation.w = math.cos(yaw / 2.0)

        self.tf_broadcaster.sendTransform(tf)
        self.publish_odom(tf)

    def publish_odom(self, tf):
        """Mirror the map->base_link transform as /odom, with the body twist.

        OdomSmoother only ever reads `twist`, but the pose is filled in from
        the same flattened transform so the message is self-consistent.
        """
        odom = Odometry()
        odom.header.stamp = tf.header.stamp
        odom.header.frame_id = tf.header.frame_id
        odom.child_frame_id = tf.child_frame_id

        odom.pose.pose.position.x = tf.transform.translation.x
        odom.pose.pose.position.y = tf.transform.translation.y
        odom.pose.pose.position.z = tf.transform.translation.z
        odom.pose.pose.orientation = tf.transform.rotation

        # velocity_body is already body-frame ENU (FLU), which is the frame
        # Nav2 wants for the current speed.
        odom.twist.twist = self.last_twist.twist

        self.odom_pub.publish(odom)


def main():

    rclpy.init()

    node = MavrosTFBridge()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()