#!/usr/bin/env python3
"""
Nav2 (MPPI) bring-up for the REAL-lidar setup (obstacle_avoidance_hw).

Identical wiring to the sim nav2 launch, but on WALL time: use_sim_time:=false
everywhere (there is no /clock without Gazebo). Consumes /lidar/scan from the
LightWare sf45b driver (see bringup_hw.launch.py) and drives ArduPilot SITL via
the cmd_vel -> /mavros/setpoint_velocity bridge.

Run bringup_hw.launch.py first; start this once the drone is armed/airborne.
Runs entirely from source (no colcon install of this package required).
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
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


# Velocity out via pymavlink SET_POSITION_TARGET_LOCAL_NED (BODY_NED), NOT mavros.
# cmd_vel_to_mavros.py is kept in the package but no longer launched.
SEND_NED_PY = script_path('obstacle_avoidance_hw.cmdvel_to_send_ned')


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

    # MAVLink endpoint for the velocity sender. Keep it SEPARATE from mavros
    # (mavros is usually on 14550) — point SITL/MAVProxy/mavlink-router here.
    mavlink_conn = LaunchConfiguration('mavlink_conn')

    # Plain script (script_path resolves source or install).
    send_ned = ExecuteProcess(
        cmd=['python3', SEND_NED_PY,
             '--ros-args', '-p', ['connection:=', mavlink_conn]],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS),
        # Real hardware -> wall time.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('mavlink_conn', default_value='udpin:0.0.0.0:14551'),

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
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='nav2_waypoint_follower', executable='waypoint_follower',
            name='waypoint_follower', output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
        Node(
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
                'bond_timeout': 20.0,
            }],
        ),

        # Nav2 body-frame /cmd_vel -> ArduPilot BODY_NED velocity via pymavlink
        # (SET_POSITION_TARGET_LOCAL_NED). Does NOT go through mavros.
        send_ned,
    ])
