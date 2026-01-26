================================================================================
AURO 2025 Coursework - Autonomous Hazardous Material Collection
================================================================================

WORKSPACE STRUCTURE
--------------------------------------------------------------------------------
.
├── .devcontainer/
│   └── devcontainer.json      # 5 scenarios configuration
├── solution/                  # Main ROS2 package
│   ├── solution/
│   │   ├── robot_controller.py   # FSM-based robot controller (8 states)
│   │   └── data_logger.py        # Performance logging node
│   ├── launch/
│   │   └── solution_launch.py    # Main launch file
│   ├── config/
│   │   ├── map2.yaml/pgm         # Pre-built map for Nav2
│   │   └── initial_poses.yaml    # Robot spawn positions
│   └── params/
│       └── custom_nav2_params_namespaced.yaml  # Nav2 configuration
├── assessment/                # Provided assessment package (DO NOT MODIFY)
├── assessment_interfaces/     # Provided message definitions (DO NOT MODIFY)
├── auro_interfaces/           # Custom interface definitions
├── gazebo_ros_link_attacher/  # Gazebo plugin for barrel attachment
├── rosgraph.png              # ROS computation graph
└── README.txt                # This file

BUILDING AND RUNNING
--------------------------------------------------------------------------------
1. Build the workspace:
   $ colcon build

2. Source the workspace:
   $ source install/setup.bash

3. Run a scenario (1-5):
   $ ros2 launch solution solution_launch.py

   Or with specific parameters:
   $ ros2 launch solution solution_launch.py use_nav2:=true use_rviz:=true

KNOWN ISSUES AND WORKAROUNDS
--------------------------------------------------------------------------------
ISSUE: Robot drives off to left corridor at startup
CAUSE: Simulation time synchronisation issue on first run
FIX:   Clean build and re-source the workspace:
       $ rm -rf build/ install/ log/
       $ colcon build
       $ source install/setup.bash

This resolves the sim time synchronisation and the robot will behave correctly.

SCENARIO DESCRIPTIONS
--------------------------------------------------------------------------------
Scenario 1 - Baseline Single Robot
  - Standard conditions with obstacles, no sensor noise
  - Tests basic navigation, search, collection, decontamination

Scenario 2 - Sensor Noise Robustness  
  - Sensor noise enabled (camera, LiDAR, odometry)
  - Tests robustness under realistic noisy conditions

Scenario 3 - Different Barrel Distribution
  - Different random seed (3) for barrel placement
  - Tests adaptability to varying item locations

Scenario 4 - Open Environment
  - No obstacles in main room, random seed 67
  - Tests search efficiency in open space

Scenario 5 - Full Complexity
  - Sensor noise + obstacles + different seed (19)
  - Stress test of overall system robustness

SOLUTION OVERVIEW
--------------------------------------------------------------------------------
The solution uses an 8-state Finite State Machine (FSM) architecture:

  SEARCHING -> APPROACHING -> POSITIONING -> PICKING_UP -> 
  DELIVERING -> OFFLOADING -> CLEARING_SPACE -> DECONTAMINATING

Key features:
- Hybrid navigation: Nav2 for global planning, visual servoing for approach
- 14-waypoint patrol pattern covering entire arena
- Fault tolerance: 30s approach timeout with escape maneuver
- Barrel hysteresis: Prevents target oscillation (1.5x switch threshold)
- Distance filtering: Ignores distant barrels (5000px minimum size)
- Automatic decontamination when radiation >= 50

DEPENDENCIES
--------------------------------------------------------------------------------
All dependencies are provided by the auro-dev Docker image:
- ROS 2 Humble Hawksbill
- Gazebo Classic 11
- Nav2 navigation stack
- TurtleBot3 packages

No additional apt or pip packages required.

================================================================================