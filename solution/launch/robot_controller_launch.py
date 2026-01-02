#!/usr/bin/env python3
"""
AURO 2025 Solution Launch File

This launch file starts the robot controller for the barrel collection task.
It is designed to work with the assessment's solution_launch.py.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    robot_namespace_arg = DeclareLaunchArgument(
        'robot_namespace',
        default_value='',
        description='Robot namespace for multi-robot scenarios'
    )
    
    # Robot controller node
    robot_controller_node = Node(
        package='solution',
        executable='improved_controller.py',
        name='robot_controller',
        namespace=LaunchConfiguration('robot_namespace'),
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        robot_namespace_arg,
        robot_controller_node,
    ])
