#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Declare arguments
    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value='assessment_map.yaml',
        description='Full path to map yaml file'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Get paths
    nav2_bringup_dir = FindPackageShare('nav2_bringup')
    
    # Nav2 parameters
    nav2_params = PathJoinSubstitution([
        FindPackageShare('assessment'),
        'config',
        'nav2_params.yaml'
    ])
    
    # Include Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                nav2_bringup_dir,
                'launch',
                'bringup_launch.py'
            ])
        ]),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': nav2_params,
            'autostart': 'true'
        }.items()
    )
    
    # Launch autonomous barrel collector node
    barrel_collector_node = Node(
        package='solution',
        executable='autonomous_barrel_collector',
        name='barrel_collector',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )
    
    return LaunchDescription([
        map_file_arg,
        use_sim_time_arg,
        nav2_bringup,
        barrel_collector_node
    ])