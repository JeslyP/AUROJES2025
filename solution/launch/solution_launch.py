import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for the solution."""
    
    # Get package directories
    solution_dir = get_package_share_directory('solution')
    assessment_dir = get_package_share_directory('assessment')
    
    # =========================================================================
    # LAUNCH ARGUMENTS
    # =========================================================================
    
    # Simulation parameters
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', default_value='1',
        description='Number of robots to spawn'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='True',
        description='Whether to start RViz'
    )
    
    use_nav2_arg = DeclareLaunchArgument(
        'use_nav2', default_value='True',
        description='Whether to use Nav2 navigation stack'
    )
    
    map_arg = DeclareLaunchArgument(
        'map', default_value='',
        description='Full path to map yaml file'
    )
    
    sensor_noise_arg = DeclareLaunchArgument(
        'sensor_noise', default_value='False',
        description='Whether to enable sensor noise'
    )
    
    obstacles_arg = DeclareLaunchArgument(
        'obstacles', default_value='True',
        description='Whether world contains obstacles'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='False',
        description='Whether to run Gazebo headless'
    )
    
    limit_real_time_factor_arg = DeclareLaunchArgument(
        'limit_real_time_factor', default_value='True',
        description='Whether to limit real-time factor to 1.0'
    )
    
    experiment_duration_arg = DeclareLaunchArgument(
        'experiment_duration', default_value='840',
        description='Experiment duration in seconds'
    )
    
    random_seed_arg = DeclareLaunchArgument(
        'random_seed', default_value='0',
        description='Random seed for barrel placement'
    )
    
    barrels_arg = DeclareLaunchArgument(
        'barrels', default_value='True',
        description='Whether to enable barrel manager'
    )
    
    vision_sensor_arg = DeclareLaunchArgument(
        'vision_sensor', default_value='True',
        description='Whether to enable vision sensor'
    )
    
    vision_sensor_debug_arg = DeclareLaunchArgument(
        'vision_sensor_debug', default_value='False',
        description='Whether to enable vision sensor debug output'
    )
    
    # Data logging parameters
    data_log_path_arg = DeclareLaunchArgument(
        'data_log_path', default_value='',
        description='Path for data logging'
    )
    
    data_log_filename_arg = DeclareLaunchArgument(
        'data_log_filename', default_value='experiment_log.csv',
        description='Filename for data logging'
    )
    
    # =========================================================================
    # INCLUDE ASSESSMENT LAUNCH
    # =========================================================================
    
    assessment_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(assessment_dir, 'launch', 'assessment.launch.py')
        ),
        launch_arguments={
            'num_robots': LaunchConfiguration('num_robots'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'use_nav2': LaunchConfiguration('use_nav2'),
            'map': LaunchConfiguration('map'),
            'sensor_noise': LaunchConfiguration('sensor_noise'),
            'obstacles': LaunchConfiguration('obstacles'),
            'headless': LaunchConfiguration('headless'),
            'limit_real_time_factor': LaunchConfiguration('limit_real_time_factor'),
            'random_seed': LaunchConfiguration('random_seed'),
            'barrels': LaunchConfiguration('barrels'),
            'vision_sensor': LaunchConfiguration('vision_sensor'),
            'vision_sensor_debug': LaunchConfiguration('vision_sensor_debug'),
        }.items()
    )
    
    # =========================================================================
    # SOLUTION NODES
    # =========================================================================
    
    # Robot controller (delayed start to allow simulation to initialise)
    robot_controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='solution',
                executable='robot_controller',
                name='robot_controller',
                namespace='robot1',
                parameters=[{'use_sim_time': True}],
                output='screen',
            )
        ]
    )
    
    # Data logger
    data_logger = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='solution',
                executable='data_logger',
                name='data_logger',
                parameters=[{
                    'use_sim_time': True,
                    'log_path': LaunchConfiguration('data_log_path'),
                    'log_filename': LaunchConfiguration('data_log_filename'),
                }],
                output='screen',
            )
        ]
    )
    
    # =========================================================================
    # LAUNCH DESCRIPTION
    # =========================================================================
    
    return LaunchDescription([
        # Arguments
        num_robots_arg,
        use_rviz_arg,
        use_nav2_arg,
        map_arg,
        sensor_noise_arg,
        obstacles_arg,
        headless_arg,
        limit_real_time_factor_arg,
        experiment_duration_arg,
        random_seed_arg,
        barrels_arg,
        vision_sensor_arg,
        vision_sensor_debug_arg,
        data_log_path_arg,
        data_log_filename_arg,
        
        # Assessment environment
        assessment_launch,
        
        # Solution nodes
        robot_controller,
        data_logger,
    ])
