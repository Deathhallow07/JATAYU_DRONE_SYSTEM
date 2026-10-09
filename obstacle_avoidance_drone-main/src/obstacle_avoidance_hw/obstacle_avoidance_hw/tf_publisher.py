#!/usr/bin/env python3

import math

import rclpy

from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
class MavrosTFBridge(Node):

    def __init__(self):
        super().__init__('mavros_tf_bridge')

        # -------- Parameters --------

        self.declare_parameter(
            'lidar_frame',
            'laser'          # LightWare sf45b default frame_id
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

        self.get_logger().info("MAVROS TF Bridge Started")

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


def main():

    rclpy.init()

    node = MavrosTFBridge()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()