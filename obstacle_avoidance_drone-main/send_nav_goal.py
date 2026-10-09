#!/usr/bin/env python3
"""
Send a Nav2 NavigateToPose goal in the `map` frame and print progress.

This is the sim counterpart of send_goal.py (which speaks lat/lon UDP to the
hardware goal_socket_bridge). Here map == the drone's ENU local frame with its
origin at the spawn point, so the goal is just metres east/north of takeoff.

    python3 send_nav_goal.py              # default goal, straight past the building
    python3 send_nav_goal.py 0 60         # x=east, y=north, metres
    python3 send_nav_goal.py --cancel

In runway.sdf the building sits at (0, 25) and is 12 m wide (E-W) x 8 m deep,
so the default goal at (0, 45) is dead ahead through it -- the drone has to go
around to get there, which is the whole point of the exercise.
"""

import argparse
import math
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

GOAL_X = 0.0
GOAL_Y = 45.0
GOAL_YAW = math.pi / 2.0     # face north, i.e. the direction of travel


class GoalSender(Node):
    def __init__(self):
        # The action server stamps and TFs the goal on sim time; a wall-clock
        # stamp from here is decades "in the future" relative to the sim clock
        # and the goal gets rejected. This has to go in as a parameter
        # override at construction -- set_parameters() after the fact raises
        # "handle cannot be modified after node creation".
        super().__init__(
            'send_nav_goal',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        # NB: not self.handle -- rclpy.Node reserves that attribute name.
        self.goal_handle = None
        self._last_fb = 0.0

    def send(self, x, y, yaw, timeout):
        self.get_logger().info('waiting for the navigate_to_pose action server')
        if not self.client.wait_for_server(timeout_sec=timeout):
            self.get_logger().error(
                'no navigate_to_pose server -- is bt_navigator active?')
            return 1

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(f'-> goal map ({x:.1f}, {y:.1f}) yaw {yaw:.2f}')
        send_future = self.client.send_goal_async(goal, self.feedback_cb)
        rclpy.spin_until_future_complete(self, send_future)
        self.goal_handle = send_future.result()
        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().error('goal rejected')
            return 1
        self.get_logger().info('goal accepted')

        result_future = self.goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        # None when the spin was interrupted (Ctrl-C, or an outer `timeout`
        # killing us mid-flight) rather than the goal actually finishing.
        result = result_future.result()
        if result is None:
            self.get_logger().error('interrupted before the goal finished')
            return 1
        status = result.status
        # 4 == STATUS_SUCCEEDED
        if status == 4:
            self.get_logger().info('goal REACHED')
            return 0
        self.get_logger().error(f'navigation finished with status {status}')
        return 1

    def feedback_cb(self, fb):
        # bt_navigator pushes feedback at the controller rate; printing every
        # message buries the interesting lines, so throttle to ~1 Hz.
        f = fb.feedback
        now = time.monotonic()
        if now - self._last_fb < 1.0:
            return
        self._last_fb = now
        self.get_logger().info(
            f'  remaining {f.distance_remaining:5.1f} m  '
            f'recoveries {f.number_of_recoveries}')

    def cancel(self, timeout):
        """Cancel whatever the server is running.

        We have no goal handle in a fresh process, so go to the action's
        cancel service directly: a zeroed goal_id + stamp means "cancel all".
        """
        from action_msgs.srv import CancelGoal

        cli = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        if not cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error('no navigate_to_pose cancel service')
            return 1
        fut = cli.call_async(CancelGoal.Request())
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().info('cancel sent')
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('x', nargs='?', type=float, default=GOAL_X)
    ap.add_argument('y', nargs='?', type=float, default=GOAL_Y)
    ap.add_argument('--yaw', type=float, default=GOAL_YAW)
    ap.add_argument('--cancel', action='store_true')
    ap.add_argument('--server-timeout', type=float, default=120.0)
    args = ap.parse_args()

    rclpy.init()
    node = GoalSender()
    try:
        if args.cancel:
            rc = node.cancel(args.server_timeout)
        else:
            rc = node.send(args.x, args.y, args.yaw, args.server_timeout)
    except KeyboardInterrupt:
        rc = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
