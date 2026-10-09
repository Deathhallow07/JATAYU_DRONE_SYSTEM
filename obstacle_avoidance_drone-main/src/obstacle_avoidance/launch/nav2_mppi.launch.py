#!/usr/bin/env python3
"""
Mapless Nav2 (MPPI controller) bring-up for ArduPilot SITL + Gazebo lidar.

No map_server / AMCL: the map->base_link transform is supplied by the
mavros_tf_bridge node (tf_publisher.py) from /mavros/local_position/pose,
and both costmaps run as rolling windows off /lidar/scan.

Nodes started here:
  - controller_server  (nav2_mppi_controller / MPPIController)
  - planner_server     (NavFn A*)
  - behavior_server    (recoveries)
  - bt_navigator
  - waypoint_follower
  - lifecycle_manager  (autostart)
  - cmd_vel_to_mavros  (/cmd_vel body FLU -> MAVLink SET_POSITION_TARGET BODY_NED)

Assumes mavros, the ros_gz lidar bridge, and mavros_tf_bridge are already
running (they are part of your existing SITL/Gazebo launch).

Runs entirely from source (absolute paths), so it does NOT require the
obstacle_avoidance package to be colcon-installed.
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_PARAMS = os.path.join(PKG_DIR, 'config', 'nav2_params.yaml')


def script_path(module):
    """Path to a package .py file, run-from-source or colcon-installed.

    In the source tree launch/ and the python package are siblings; colcon
    splits them (launch -> share/<pkg>/launch, module -> lib/.../site-packages),
    so fall back to importing the module to find it. Source wins when both
    exist, so edits take effect without a rebuild.
    """
    in_source = os.path.join(PKG_DIR, *module.split('.')) + '.py'
    if os.path.exists(in_source):
        return in_source
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise RuntimeError(
            f'cannot find {module}: no {in_source} and not importable '
            '(source the workspace install/setup.bash?)')
    return spec.origin


BRIDGE_PY = script_path('obstacle_avoidance.cmd_vel_to_mavros')

# Behaviour tree for NavigateToPose. Ours drops the global-costmap clearing
# that the stock tree does on a planning failure, so an unreachable goal
# ABORTS instead of looping forever -- which is what lets lawnmower.py skip a
# blocked waypoint. Set NAV_BT_XML to fall back to nav2's stock tree:
#   NAV_BT_XML=/opt/ros/lyrical/share/nav2_bt_navigator/behavior_trees/\
#              navigate_to_pose_w_replanning_and_recovery.xml
BT_XML = os.environ.get(
    'NAV_BT_XML',
    os.path.join(PKG_DIR, 'config', 'navigate_to_pose_no_global_clear.xml'))

# Seconds to wait after the servers are constructed before lifecycle_manager
# starts transitioning them. See the TimerAction below for why this exists.
LIFECYCLE_DELAY = float(os.environ.get('LIFECYCLE_DELAY', '20.0'))


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    lifecycle_nodes = [
        'planner_server',
        'controller_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
    ]

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='nav2_controller', executable='controller_server',
            name='controller_server', output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_planner', executable='planner_server',
            name='planner_server', output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_behaviors', executable='behavior_server',
            name='behavior_server', output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_bt_navigator', executable='bt_navigator',
            name='bt_navigator', output='screen',
            parameters=[params_file, {
                'use_sim_time': use_sim_time,
                'default_nav_to_pose_bt_xml': BT_XML,
            }],
        ),
        Node(
            package='nav2_waypoint_follower', executable='waypoint_follower',
            name='waypoint_follower', output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        # Held back deliberately -- this delay buys discovery time for two
        # separate rmw_fastrtps races, both of which strand bringup:
        #
        #  1. lifecycle_manager creates its change_state service clients and
        #     immediately starts calling them, before they are matched to the
        #     servers. The server's reply is then undeliverable ("failed to
        #     send response to /planner_server/change_state (timeout): client
        #     will not receive response"), and the manager never retries, so
        #     bt_navigator stays `unconfigured` forever.
        #  2. The costmaps' TF listeners have to receive map->base_link before
        #     activate times out ("Failed to activate local_costmap because
        #     transform from base_link to map did not become available before
        #     timeout"), which aborts the whole bringup.
        #
        # Both are won by simply not transitioning the instant the nodes are
        # constructed: the servers' subscriptions and the manager's clients get
        # a head start on discovery. Raise LIFECYCLE_DELAY on a loaded box.
        # start_sim.sh still polls bt_navigator for ACTIVE and respawns this
        # pane as a backstop.
        TimerAction(period=LIFECYCLE_DELAY, actions=[Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_navigation', output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                'node_names': lifecycle_nodes,
                # The *activate* transition blocks inside the costmaps until
                # map->base_link exists, which in SITL means waiting for EKF
                # GPS lock. Nav2 1.5's default service_timeout is far shorter
                # than that, so the manager aborted bringup while the servers
                # were still legitimately waiting for the transform.
                'service_timeout': 120.0,
                'bond_timeout': 40.0, #20.0,
            }],
        )]),

        # Nav2 body-frame /cmd_vel -> pymavlink send_ned velocity setpoint.
        # Run directly from source so no colcon install is required.
        ExecuteProcess(
            cmd=['python3', BRIDGE_PY],
            output='screen',
        ),
    ])
