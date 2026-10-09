#!/usr/bin/env python3
"""
Infrastructure bring-up for the REAL LightWare SF45/B lidar (no Gazebo).

Differences from the sim bringup:
  - Gazebo is gone: NO gz parameter_bridge, and NO /clock bridge.
  - Everything runs on WALL time (use_sim_time:=false). Without a /clock
    publisher that is exactly what we want -- do NOT set use_sim_time here.
  - The lidar comes from the LightWare 'lightwarelidar2' ROS 2 driver
    (sf45b node), which publishes sensor_msgs/LaserScan on /scan when
    publishLaserScan:=true. We remap /scan -> /lidar/scan so the Nav2 config
    (which reads /lidar/scan) is unchanged.

Still SITL for the flight controller: MAVROS stays on udp://127.0.0.1:14550@.

Starts:
  - lightwarelidar2 sf45b   (real SF45/B -> /lidar/scan, frame 'laser')
  - MAVROS (apm.launch, ArduPilot SITL over UDP)
  - mavros_tf_bridge (tf_publisher.py): map->base_link (flattened) +
    static base_link->laser

Prereq: build the LightWare driver first (C++, so colcon is fine):
    cd <ws>/src && git clone https://github.com/LightWare-Optoelectronics/lightwarelidar2
    colcon build --packages-select lightwarelidar2 && source install/setup.bash
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
from launch_ros.descriptions import ParameterValue

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


TF_PY = script_path('obstacle_avoidance_hw.tf_publisher')


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    lidar_port = LaunchConfiguration('lidar_port')
    lidar_baud = LaunchConfiguration('lidar_baud')
    lidar_frame = LaunchConfiguration('lidar_frame')
    low_angle = LaunchConfiguration('low_angle')
    high_angle = LaunchConfiguration('high_angle')

    mavros_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'apm.launch')

    # Run the TF bridge as a plain script (script_path resolves source or install).
    tf_node = ExecuteProcess(
        cmd=['python3', TF_PY, '--ros-args', '-p', 'lidar_frame:=laser'],
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://127.0.0.1:14550@'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('lidar_baud', default_value='115200'),
        DeclareLaunchArgument('lidar_frame', default_value='laser'),
        # SF45/B scan sector. Widen for more coverage (sensor supports ~+/-170).
        DeclareLaunchArgument('low_angle', default_value='-100'),
        DeclareLaunchArgument('high_angle', default_value='100'),

        # ---- Real SF45/B lidar -> /lidar/scan ----
        Node(
            package='lightwarelidar2', executable='sf45b',
            name='sf45b', output='screen',
            parameters=[{
                'port': lidar_port,
                'baudrate': ParameterValue(lidar_baud, value_type=int),
                'frame_id': lidar_frame,
                'publishLaserScan': True,
                'lowAngleLimit': ParameterValue(low_angle, value_type=int),
                'highAngleLimit': ParameterValue(high_angle, value_type=int),
            }],
            remappings=[('/scan', '/lidar/scan')],
        ),

        # ---- MAVROS <-> ArduPilot SITL (UDP) ----
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(mavros_launch),
            launch_arguments={'fcu_url': fcu_url}.items(),
        ),

        # ---- mavros pose -> map->base_link TF (+ static base_link->laser) ----
        # Wall time (no use_sim_time). lidar_frame must match the sf45b frame_id.
        tf_node,
    ])
