#!/usr/bin/env python3
"""
Infrastructure bring-up for the ArduPilot SITL + Gazebo obstacle-avoidance rig.

Starts, in one shot, everything that used to be launched by hand:
  - ros_gz parameter_bridge for /clock  (gz sim time -> ROS)  *** required ***
  - ros_gz parameter_bridge for /lidar/scan (Gazebo lidar -> ROS)
  - MAVROS (apm.launch) connected to ArduPilot SITL
  - mavros_tf_bridge (tf_publisher.py): map->base_link + base_link->lidar

The /clock bridge is the important one: every downstream node runs with
use_sim_time:=true, and without ROS /clock their time is frozen (costmaps
never update, "No map received", lifecycle bringup aborts).

This does NOT start the Nav2 stack — bring that up separately once the drone is
armed/airborne:
    ros2 launch src/obstacle_avoidance/launch/nav2_mppi.launch.py

Gazebo (`gz sim -r runway.sdf`) and ArduPilot SITL are started outside ROS and
are assumed to already be running.
"""

import importlib.util
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CLOCK_BRIDGE = '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
SCAN_BRIDGE = '/lidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'

PKG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


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


TF_PY = script_path('obstacle_avoidance.tf_publisher')


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://127.0.0.1:14550@'),

        # Gazebo -> ROS bridges: sim clock (critical) + lidar scan.
        Node(
            package='ros_gz_bridge', executable='parameter_bridge',
            name='gz_ros_bridge', output='screen',
            arguments=[CLOCK_BRIDGE, SCAN_BRIDGE],
        ),

        # MAVROS <-> ArduPilot SITL.
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(mavros_launch),
            launch_arguments={'fcu_url': fcu_url}.items(),
        ),

        # mavros pose -> map->base_link TF (+ static base_link->lidar).
        # use_sim_time so the TF is stamped on the sim clock, matching the
        # sim-time lidar scan (otherwise the scan can't be transformed to map).
        # Run from source (the colcon console-script is broken by setuptools 82).
        ExecuteProcess(
            cmd=['python3', TF_PY, '--ros-args', '-p', 'use_sim_time:=true'],
            output='screen',
        ),
    ])
