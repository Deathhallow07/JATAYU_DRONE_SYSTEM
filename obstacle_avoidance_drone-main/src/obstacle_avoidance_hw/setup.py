import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'obstacle_avoidance_hw'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        # *.py, not *: a bare glob picks up scripts/__pycache__ and setuptools
        # dies with "can't copy 'scripts/__pycache__': not a regular file".
        (os.path.join('share', package_name, 'scripts'),
            glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='satvik',
    maintainer_email='youremail@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mavros_tf_bridge = obstacle_avoidance_hw.tf_publisher:main',
            'cmd_vel_to_mavros = obstacle_avoidance_hw.cmd_vel_to_mavros:main',
            'cmdvel_to_send_ned = obstacle_avoidance_hw.cmdvel_to_send_ned:main',
            'goal_socket_bridge = obstacle_avoidance_hw.goal_socket_bridge:main',
        ],
    },
)
