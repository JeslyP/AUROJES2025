#!/usr/bin/env python3
"""
Solution Launch File for AURO Assessment
This integrates with solution_launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    return LaunchDescription([
        # Robot ID parameter
        DeclareLaunchArgument(
            'robot_id',
            default_value='robot1',
            description='Robot identifier'
        ),
        
        # Use simulation time
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        
        # Autonomous barrel collector node
        Node(
            package='solution',
            executable='autonomous_barrel_collector',
            name='barrel_collector',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'robot_id': LaunchConfiguration('robot_id')
            }]
        ),
    ])