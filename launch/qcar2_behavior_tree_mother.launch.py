#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    behavior_tree_share = get_package_share_directory('qcar2_behavior_tree')
    mixer_share = get_package_share_directory('qcar2_mixer')

    bt_config_default = os.path.join(behavior_tree_share, 'config', 'behavior_tree.yaml')
    mixer_launch_file = os.path.join(mixer_share, 'launch', 'qcar2_hybrid_planner.launch.py')

    bt_config = LaunchConfiguration('bt_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bt_config',
            default_value=bt_config_default,
            description='Behavior tree configuration YAML file.',
        ),
        # Include qcar2_hybrid_planner.launch.py - uses its internal default YAMLs
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mixer_launch_file),
        ),
        Node(
            package='qcar2_behavior_tree',
            executable='qcar2_behavior_tree_manager',
            name='qcar2_behavior_tree_manager',
            output='screen',
            parameters=[bt_config],
            emulate_tty=True,
        ),
    ])
