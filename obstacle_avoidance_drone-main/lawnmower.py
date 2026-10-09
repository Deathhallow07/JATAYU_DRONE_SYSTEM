#!/usr/bin/env python3
"""
Fly a lawnmower (boustrophedon) survey, skipping waypoints Nav2 cannot reach.

Sends one NavigateToPose goal per pattern corner, in order. If a goal ABORTS
-- no route to it, or the goal itself sits inside an obstacle -- that waypoint
is logged as SKIPPED and the mission moves straight to the next one instead of
stopping or retrying forever.

    python3 lawnmower.py                 # default pattern (see below)
    python3 lawnmower.py --dry-run       # print the waypoints, fly nothing
    python3 lawnmower.py --lanes -30,0,30 --y0 5 --y1 35

Requires the drone already airborne and hovering (start_sim.sh does that in
pane 5) and Nav2 active.

WHY GOALS ABORT INSTEAD OF HANGING
This depends on the custom behaviour tree in
config/navigate_to_pose_no_global_clear.xml. The stock nav2 tree responds to
"no valid path" by clearing the global costmap, which deletes the obstacle
proving the goal is unreachable and loops forever. With that removed, an
unreachable goal aborts in a few seconds and this script can move on.

GEOMETRY
Lanes run north-south (along +y) at each x in --lanes, alternating direction so
the drone snakes rather than flying back to the start of every lane. Geometry
lives in pattern.py, shared with the ArduPilot arm (oa_lawnmower.py) so both
stacks fly the identical path. It is mirrored in worlds/lawnmower.sdf, which
parks buildings on two of the corners on purpose:
    (-15, 35)  buried inside `corner_nw`     -> goal occupied
    ( 15,  5)  walled in by `courtyard_se`   -> no route
Both should come out SKIPPED. Everything else should be REACHED.

STAY INSIDE THE COSTMAP. Lanes stop at y=35, not y=45, on purpose. A blocked
waypoint near the map edge is a trap: MPPI pushes the drone away from the
obstacle's inflated cost field, and if that shoves it past the +/-50 m costmap
boundary, NavFn can no longer plan from the robot's own position and EVERY
remaining waypoint aborts -- not because it is blocked, but because the robot
is nowhere. The summary flags that case explicitly.
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
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)

# Geometry lives in pattern.py so the ArduPilot stack (oa_lawnmower.py) flies
# the identical path. Do not redefine it here.
from pattern import MAP_EXTENT, build_pattern

# action_msgs/GoalStatus
STATUS_SUCCEEDED = 4
STATUS_ABORTED = 6
STATUS_CANCELED = 5
STATUS_NAME = {STATUS_SUCCEEDED: 'REACHED', STATUS_ABORTED: 'SKIPPED (aborted)',
               STATUS_CANCELED: 'CANCELED'}


class Lawnmower(Node):
    def __init__(self, server_timeout):
        # use_sim_time must be an override at construction; setting it after
        # the fact raises "handle cannot be modified after node creation".
        super().__init__(
            'lawnmower',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.server_timeout = server_timeout
        self._last_fb = 0.0

        # Track where the drone actually is, so an aborted waypoint can say
        # WHY it aborted rather than just that it did.
        self.pos = None
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            lambda m: setattr(self, 'pos',
                              (m.pose.position.x, m.pose.position.y)),
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST, depth=10))

    def check_extent(self):
        """Return a warning string if the drone has left the global costmap.

        Outside it, the planner cannot even resolve the start pose, so every
        subsequent goal aborts regardless of whether it is reachable. Worth
        calling out loudly -- it looks identical to 'everything is blocked'.
        """
        if self.pos is None:
            return None
        x, y = self.pos
        if abs(x) > MAP_EXTENT or abs(y) > MAP_EXTENT:
            return (f'drone at ({x:.1f}, {y:.1f}) is OUTSIDE the '
                    f'+/-{MAP_EXTENT:.0f} m costmap')
        return None

    def wait_for_server(self):
        self.get_logger().info('waiting for the navigate_to_pose action server')
        if not self.client.wait_for_server(timeout_sec=self.server_timeout):
            self.get_logger().error(
                'no navigate_to_pose server -- is bt_navigator active?')
            return False
        return True

    def goto(self, x, y, yaw):
        """Drive one waypoint. Returns a GoalStatus, or None if rejected."""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal = NavigateToPose.Goal()
        goal.pose = pose

        send_future = self.client.send_goal_async(goal, self._feedback)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('  goal REJECTED by the server')
            return None

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error('  interrupted before the goal finished')
            return None
        return result.status

    def _feedback(self, fb):
        # bt_navigator pushes feedback at the controller rate; throttle to 1 Hz
        # so the per-waypoint lines stay readable.
        now = time.monotonic()
        if now - self._last_fb < 1.0:
            return
        self._last_fb = now
        f = fb.feedback
        self.get_logger().info(
            f'    {f.distance_remaining:5.1f} m to go, '
            f'{f.number_of_recoveries} recoveries')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lanes', default=None,
                    help='comma-separated lane x positions (default: pattern.py)')
    ap.add_argument('--y0', type=float, default=None, help='lane start y')
    ap.add_argument('--y1', type=float, default=None, help='lane end y')
    ap.add_argument('--server-timeout', type=float, default=120.0)
    ap.add_argument('--dry-run', action='store_true',
                    help='print the waypoints and exit without flying')
    args = ap.parse_args()

    lanes = ([float(v) for v in args.lanes.split(',') if v.strip()]
             if args.lanes else None)
    pattern = build_pattern(lanes, args.y0, args.y1)

    if args.dry_run:
        print(f'{len(pattern)} waypoints:')
        for i, (x, y) in enumerate(pattern, 1):
            print(f'  {i:2d}  ({x:7.1f}, {y:7.1f})')
        return 0

    rclpy.init()
    node = Lawnmower(args.server_timeout)
    results = []
    try:
        if not node.wait_for_server():
            return 1

        for i, (x, y) in enumerate(pattern, 1):
            # Face along the leg being flown. The lidar has a 40 deg blind
            # spot behind, so pointing the nose down-track keeps it behind us.
            if i < len(pattern):
                nx, ny = pattern[i]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = math.pi / 2.0

            node.get_logger().info(
                f'--- waypoint {i}/{len(pattern)}  ({x:.1f}, {y:.1f}) ---')
            status = node.goto(x, y, yaw)
            label = STATUS_NAME.get(status, f'SKIPPED (status {status})')
            where = f'  at ({node.pos[0]:.1f}, {node.pos[1]:.1f})' if node.pos else ''
            node.get_logger().info(f'  {label}{where}')

            off_map = node.check_extent()
            if off_map:
                node.get_logger().error(f'  {off_map}')
                node.get_logger().error(
                    '  every remaining waypoint will abort for this reason, '
                    'not because it is blocked')
                label += '  [OFF MAP]'
            results.append((i, x, y, label))

            if status in (STATUS_CANCELED, None) and status is not None:
                node.get_logger().warn('canceled -- stopping the mission')
                break
    except KeyboardInterrupt:
        node.get_logger().warn('interrupted')
    finally:
        print('\n=== lawnmower summary ===')
        for i, x, y, label in results:
            print(f'  {i:2d}  ({x:7.1f}, {y:7.1f})  {label}')
        reached = sum(1 for *_, l in results if l == 'REACHED')
        print(f'  {reached}/{len(results)} reached, '
              f'{len(results) - reached} skipped')
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
