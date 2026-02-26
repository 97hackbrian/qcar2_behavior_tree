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
    mixer_launch_file = os.path.join(mixer_share, 'launch', 'qcar2_mixer.launch.py')

    bt_config = LaunchConfiguration('bt_config')
    enable_bridge = LaunchConfiguration('enable_bridge')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bt_config',
            default_value=bt_config_default,
            description='Behavior tree configuration YAML file.',
        ),
        DeclareLaunchArgument(
            'enable_bridge',
            default_value='true',
            description='Forward argument to qcar2_mixer launch.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mixer_launch_file),
            launch_arguments={
                'enable_bridge': enable_bridge,
            }.items(),
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
