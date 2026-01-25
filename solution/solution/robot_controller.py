#!/usr/bin/env python3
"""
AURO 2025 Coursework - Autonomous Hazardous Material Collection System

This module implements an autonomous barrel collection robot using a Finite State Machine (FSM)
architecture. The robot patrols a predefined route, detects and collects contaminated (red) and
clean (blue) barrels, delivers them to designated green zones, and autonomously decontaminates
when radiation levels exceed a threshold.

Date: January 2025
Module: AURO (Autonomous Robotic Systems)
"""

import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from enum import Enum

# ROS Messages
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Custom Interfaces
from assessment_interfaces.msg import BarrelList, BarrelHolders, RadiationList, ZoneList
from auro_interfaces.srv import ItemRequest

# For Dynamic Parameters (LiDAR Mask)
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class State(Enum):
    """
    Main FSM states for the robot controller.
    
    The robot progresses through these states during normal operation:
    SEARCHING -> APPROACHING -> POSITIONING -> PICKING_UP -> DELIVERING -> 
    OFFLOADING -> CLEARING_SPACE -> (DECONTAMINATING if needed) -> SEARCHING
    """
    SEARCHING = 0        # Patrolling waypoints looking for barrels
    APPROACHING = 1      # Moving toward a detected barrel
    POSITIONING = 2      # Turning around and backing up to barrel
    PICKING_UP = 3       # Calling pickup service
    DELIVERING = 4       # Navigating to green collection zone
    OFFLOADING = 5       # Reversing into zone and dropping barrel
    CLEARING_SPACE = 6   # Driving forward to clear the drop zone
    DECONTAMINATING = 7  # Navigating to cyan zone for decontamination


class CollectPhase(Enum):
    """
    Sub-phases for the APPROACHING and POSITIONING states.
    
    These phases handle the visual servoing and positioning
    needed to successfully pick up a barrel.
    """
    ALIGN = 0       # Rotate to center barrel in camera view
    APPROACH = 1    # Drive forward toward barrel
    TURN_AROUND = 2 # Rotate 180 degrees to face away from barrel
    BACKUP = 3      # Reverse toward barrel for pickup


class DecontaminatePhase(Enum):
    """
    Sub-phases for the DECONTAMINATING state.
    
    These phases handle navigation to and interaction with
    the cyan decontamination zone.
    """
    NAVIGATING = 0      # Using Nav2 to reach cyan zone
    REVERSING = 1       # Backing into the zone
    CALLING_SERVICE = 2 # Requesting decontamination service


class RobotController(Node):
    """
    Main robot controller implementing FSM-based autonomous barrel collection.
    
    This node coordinates perception, navigation, and manipulation to:
    - Patrol the arena using Nav2 waypoint navigation
    - Detect barrels using colour-based visual sensing
    - Collect barrels using visual servoing and positioning
    - Deliver barrels to designated green collection zones
    - Monitor radiation and trigger decontamination when threshold exceeded
    
    Attributes:
        state (State): Current FSM state
        collect_phase (CollectPhase): Current sub-phase during collection
        decontaminate_phase (DecontaminatePhase): Current sub-phase during decontamination
        barrels_collected (int): Total count of successfully collected barrels
        radiation_level (int): Current radiation level from contaminated barrels
    """

    def __init__(self):
        """
        Initialise the robot controller node.
        
        Sets up all ROS2 subscriptions, publishers, service clients,
        and initialises the navigation stack.
        """
        super().__init__('robot_controller')

        # =====================================================
        # 1. PARAMETERS
        # ====================================================
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        
        # Get robot namespace for multi-robot support
        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name: 
            self.robot_name = 'robot1'

        # ============================================================
        # 2. NAVIGATION SETUP
        # ============================================================
        self.navigator = BasicNavigator()
        self.set_initial_pose()
        self.navigator.waitUntilNav2Active()

        # ============================================================
        # 3. SENSOR SUBSCRIPTIONS
        # ============================================================
        # Barrel detection from visual sensor
        self.create_subscription(BarrelList, 'barrels', self.barrel_callback, 10)
        
        # LiDAR for obstacle detection and distance measurement
        self.create_subscription(LaserScan, 'scan_filtered', self.scan_callback, 10)
        
        # Global barrel holder status (which robot holds which barrel)
        self.create_subscription(BarrelHolders, '/barrel_holders', self.holders_callback, 10)
        
        # Radiation level monitoring
        self.create_subscription(RadiationList, '/radiation_levels', self.radiation_callback, 10)
        
        # Zone detection (green collection, cyan decontamination)
        self.create_subscription(ZoneList, 'zones', self.zones_callback, 10)

        # ============================================================
        # 4. PUBLISHERS
        # ============================================================
        # Direct velocity control for visual servoing
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ============================================================
        # 5. SERVICE CLIENTS
        # ============================================================
        self.cb_group = ReentrantCallbackGroup()
        
        # Barrel services
        self.pickup_client = self.create_client(
            ItemRequest, '/pick_up_item', callback_group=self.cb_group)
        self.offload_client = self.create_client(
            ItemRequest, '/offload_item', callback_group=self.cb_group)
        self.decontaminate_client = self.create_client(
            ItemRequest, '/decontaminate', callback_group=self.cb_group)
        
        # LiDAR mask service (to ignore carried barrel in obstacle detection)
        self.mask_client = self.create_client(
            SetParameters, 
            f'/{self.robot_name}/dynamic_mask/set_parameters', 
            callback_group=self.cb_group)
        
        # Wait for critical services
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Pickup Service not found!")

        # ============================================================
        # 6. STATE VARIABLES
        # ============================================================
        # FSM state tracking
        self.state = State.SEARCHING
        self.collect_phase = CollectPhase.ALIGN
        self.decontaminate_phase = DecontaminatePhase.NAVIGATING
        
        # Sensor data storage
        self.barrels = []           # Detected barrels from camera
        self.zones = []             # Detected zones from camera
        self.holding_barrel = False # Whether robot currently holds a barrel
        self.radiation_level = 0    # Current radiation level
        
        # Control flags
        self.search_enabled = False     # Enable barrel detection after waypoint 3
        self.nav_goal_sent = False      # Track if navigation goal is active
        
        # Timing variables for timed manoeuvres
        self.phase_start_time = None
        self.offload_start_time = None
        self.forward_start_time = None
        self.decontaminate_start_time = None
        self.approach_start_time = None  # NEW: Timer for approach timeout
        
        # Service call tracking
        self.service_future = None
        
        # Performance metrics
        self.barrels_collected = 0
        
        # Target tracking for hysteresis (prevents oscillation between barrels)
        self.current_target_size = 0

        # ============================================================
        # 7. CONSTANTS
        # ============================================================
        # Zone type identifiers
        self.ZONE_CYAN = 0    # Decontamination zone
        self.ZONE_GREEN = 1   # Collection zone

        # Decontamination threshold (trigger when radiation >= 50)
        self.DECONTAMINATION_THRESHOLD = 50
        
        # Barrel switching threshold (switch to new barrel if 30% larger)
        self.BARREL_SWITCH_THRESHOLD = 1.4
        
        # Minimum barrel size to target (filters out distant barrels)
        # Prevents targeting barrels in big room while still in hallway
        # Adjust this value based on testing (higher = must be closer)
        self.MIN_TARGET_SIZE = 300
        
        # NEW: Maximum time to spend approaching a barrel before giving up (seconds)
        self.MAX_APPROACH_TIME = 120.0

        # LiDAR distance measurements (initialised to infinity)
        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')
        self.back_dist = float('inf')

        # ============================================================
        # 8. PATROL WAYPOINTS
        # ============================================================
        # Waypoints define the patrol route through the arena
        # Coordinates translated for map origin at (-1.41, -23.6)
        self.waypoints = [
            {'x': 0.073,  'y': 0.013,   'name': 'Start Area'},
            {'x': 5.29,   'y': -2.06,   'name': 'Right Corridor Bottom'},
            {'x': 9.371,  'y': -2.3,    'name': 'Right Corridor Top'}, 
            {'x': 8.1,    'y': 1.95,    'name': 'Left Corridor Top'},
            {'x': 10.07,  'y': 7.65,    'name': 'Big Room Entrance'},
            {'x': 6.17,   'y': 7.611,   'name': 'Big Room Bottom Right'},
            {'x': 6.446,  'y': 12.051,  'name': 'Big Room Bottom Center'},
            {'x': 6.386,  'y': 16.079,  'name': 'Big Room Bottom Left'},
            {'x': 10.172, 'y': 15.938,  'name': 'Big Room Middle Left'},
            {'x': 14.453, 'y': 15.824,  'name': 'Big Room Top Left'},
            {'x': 14.284, 'y': 11.48,   'name': 'Big Room Top Middle'},
            {'x': 14.306, 'y': 7.676,   'name': 'Big Room Top Right'},
            {'x': 10.154, 'y': 12.412,  'name': 'Big Room Center'},
            {'x': 10.07,  'y': 7.65,    'name': 'Big Room Entrance (Exit)'},
        ]

        # Decontamination zone location (cyan zone)
        self.decontamination_zone = {'x': 10.22, 'y': -7.53}
        
        # Start patrol at waypoint 1 (skip start area)
        self.current_wp_index = 1

        # ============================================================
        # 9. MAIN CONTROL LOOP
        # ============================================================
        # Timer runs at 10Hz (every 0.1 seconds)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Robot Controller Started")

    # ================================================================
    # INITIALISATION METHODS
    # ================================================================

    def set_initial_pose(self):
        """
        Set the robot's initial pose for AMCL localisation.
        
        The initial pose must match the Gazebo spawn position,
        translated to map coordinates.
        """
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = 0.073
        pose.pose.position.y = 0.013
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(pose)

    # ================================================================
    # LIDAR MASK CONTROL
    # ================================================================

    def set_mask(self, enabled):
        """
        Enable or disable the LiDAR mask to ignore the carried barrel.
        
        When carrying a barrel, it appears in the LiDAR scan and would
        be treated as an obstacle. The mask filters out readings from
        the sector where the barrel is attached.
        
        Args:
            enabled (bool): True to enable mask, False to disable
        """
        req = SetParameters.Request()
        val_enabled = ParameterValue(
            type=ParameterType.PARAMETER_BOOL, 
            bool_value=enabled)
        
        # Mask sector angles (100-260 degrees covers rear of robot)
        val_start = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER, 
            integer_value=100)
        val_end = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER, 
            integer_value=260)

        req.parameters = [
            Parameter(name='mask_enabled', value=val_enabled),
            Parameter(name='ignore_sector_start', value=val_start),
            Parameter(name='ignore_sector_end', value=val_end)
        ]
        self.mask_client.call_async(req)
        self.get_logger().info(f"LiDAR Mask: {enabled}")

    # ================================================================
    # SENSOR CALLBACKS
    # ================================================================

    def barrel_callback(self, msg):
        """
        Process barrel detection messages from visual sensor.
        
        Args:
            msg (BarrelList): List of detected barrels with x, y, size, colour
        """
        self.barrels = msg.data

    def zones_callback(self, msg):
        """
        Process zone detection messages from visual sensor.
        
        Args:
            msg (ZoneList): List of detected zones (green=collection, cyan=decontamination)
        """
        self.zones = msg.data

    def scan_callback(self, msg):
        """
        Process LiDAR scan data to extract distances in key directions.
        
        Divides the 360-degree scan into four sectors (front, left, back, right)
        and extracts the minimum valid distance in each sector.
        
        Args:
            msg (LaserScan): Raw LiDAR scan data
        """
        ranges = msg.ranges
        if not ranges:
            return
            
        n = len(ranges)
        
        # Extract slices for each direction (30-degree sectors)
        front_slice = ranges[0:10] + ranges[-10:]      # Front: 0 degrees
        left_idx = int(n / 4)                           # Left: 90 degrees
        left_slice = ranges[left_idx-15:left_idx+15]
        back_idx = int(n / 2)                           # Back: 180 degrees
        back_slice = ranges[back_idx-15:back_idx+15]
        right_idx = int(3 * n / 4)                      # Right: 270 degrees
        right_slice = ranges[right_idx-15:right_idx+15]

        def get_min(slice_data):
            """Get minimum valid range from a slice of scan data."""
            valid = [r for r in slice_data if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        self.front_dist = get_min(front_slice)
        self.left_dist = get_min(left_slice)
        self.right_dist = get_min(right_slice)
        self.back_dist = get_min(back_slice)

    def holders_callback(self, msg):
        """
        Track whether this robot is currently holding a barrel.
        
        Args:
            msg (BarrelHolders): List of robot-barrel attachments
        """
        self.holding_barrel = False
        for h in msg.data:
            if h.robot_id == self.robot_name:
                self.holding_barrel = True
                break

    def radiation_callback(self, msg):
        """
        Update radiation level for this robot.
        
        Radiation accumulates when handling contaminated (red) barrels
        and resets to zero after decontamination.
        
        Args:
            msg (RadiationList): Radiation levels for all robots
        """
        for radiation in msg.data:
            if radiation.robot_id == self.robot_name:
                self.radiation_level = radiation.level
                break

    # ================================================================
    # HELPER METHODS
    # ================================================================

    def get_best_barrel(self):
        """
        Select the best barrel to target with hysteresis and distance filtering.
        
        Uses a combination of size (larger = closer) and position (centered)
        to select targets. Implements:
        1. Distance filtering: Ignores barrels below MIN_TARGET_SIZE (too far)
        2. Hysteresis: Prevents rapid switching between similar barrels
        3. Proximity override: Allows switching to significantly closer barrels
        
        Behaviour:
        - First filters out distant barrels (size < MIN_TARGET_SIZE)
        - SEARCHING: Always select largest nearby barrel (closest)
        - APPROACHING: Switch to new barrel only if 30% larger than current
                      Otherwise track the most centered barrel
        
        Returns:
            Barrel object or None if no nearby barrels detected
        """
        if not self.barrels:
            self.current_target_size = 0
            return None
        
        # DISTANCE FILTER: Only consider barrels that are close enough
        # This prevents targeting barrels in the big room while in the hallway
        nearby_barrels = [b for b in self.barrels if b.size >= self.MIN_TARGET_SIZE]
        
        if not nearby_barrels:
            # No barrels close enough to target
            self.current_target_size = 0
            return None
        
        CAMERA_CENTER = 320
        
        # Find the largest nearby barrel (typically closest)
        largest = max(nearby_barrels, key=lambda b: b.size)
        
        # Find the most centered nearby barrel (for stable tracking)
        centered = min(nearby_barrels, key=lambda b: abs(b.x - CAMERA_CENTER))
        
        if self.state == State.SEARCHING:
            # In SEARCHING state, always pick the largest (closest) nearby barrel
            self.current_target_size = largest.size
            return largest
        
        elif self.state == State.APPROACHING:
            # Check if a significantly larger barrel has appeared
            # This handles the case where we're driving toward a far barrel
            # and a closer one comes into view
            if largest.size > self.current_target_size * self.BARREL_SWITCH_THRESHOLD:
                self.current_target_size = largest.size
                self.collect_phase = CollectPhase.ALIGN  # Re-align to new target
                self.get_logger().info(
                    f"Switching to closer barrel! Size: {largest.size:.0f} "
                    f"(was {self.current_target_size / self.BARREL_SWITCH_THRESHOLD:.0f})")
                return largest
            
            # Otherwise keep tracking the most centered barrel for stability
            # Update target size to current centered barrel's size
            self.current_target_size = centered.size
            return centered
        
        # Default fallback: return largest nearby barrel
        return largest

    def get_zone(self, zone_type):
        """
        Find a zone of the specified type in camera view.
        
        Args:
            zone_type (int): 0=CYAN (decontamination), 1=GREEN (collection)
            
        Returns:
            Zone object (largest/closest) or None if not found
        """
        matching = [z for z in self.zones if z.zone == zone_type]
        if matching:
            return max(matching, key=lambda z: z.size)
        return None

    def stop_robot(self):
        """Immediately stop all robot motion."""
        self.cmd_vel_pub.publish(Twist())

    def elapsed(self):
        """
        Calculate elapsed time since phase_start_time.
        
        Returns:
            float: Elapsed time in seconds, or 0.0 if timer not started
        """
        if not self.phase_start_time:
            return 0.0
        return (self.get_clock().now() - self.phase_start_time).nanoseconds / 1e9

    def should_decontaminate(self):
        """
        Check if radiation level requires decontamination.
        
        Returns:
            bool: True if radiation >= threshold (50)
        """
        return self.radiation_level >= self.DECONTAMINATION_THRESHOLD

    def escape_backward(self):
        """
        Emergency escape manoeuvre - reverse to get unstuck.
        
        Used when the robot is stuck or approach times out.
        Reverses for 2 seconds then clears costmaps.
        """
        self.get_logger().info("Executing escape manoeuvre (reversing)...")
        twist = Twist()
        twist.linear.x = -0.2  # Reverse at 0.2 m/s
        
        # Reverse for 2 seconds (20 iterations at 10Hz)
        for _ in range(20):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        self.stop_robot()
        self.navigator.clearAllCostmaps()

    # ================================================================
    # MAIN CONTROL LOOP
    # ================================================================

    def control_loop(self):
        """
        Main FSM control loop, called at 10Hz.
        
        Implements the state machine logic for autonomous barrel collection.
        Each state handles its specific behaviour and transitions.
        """
        
        # ============================================================
        # STATE 1: SEARCHING
        # Patrol waypoints and look for barrels
        # ============================================================
        if self.state == State.SEARCHING:
            # Check for barrels if search is enabled (after waypoint 3)
            if self.search_enabled:
                best_barrel = self.get_best_barrel()
                if best_barrel and not self.holding_barrel:
                    self.get_logger().info(f"BARREL SPOTTED! Size: {best_barrel.size:.0f}")
                    self.navigator.cancelTask()
                    self.stop_robot()
                    self.navigator.clearAllCostmaps()
                    self.state = State.APPROACHING
                    self.collect_phase = CollectPhase.ALIGN
                    self.approach_start_time = self.get_clock().now()  # NEW: Start approach timer
                    self.nav_goal_sent = False
                    return

            # Send navigation goal to next waypoint
            if not self.nav_goal_sent:
                wp = self.waypoints[self.current_wp_index]
                self.get_logger().info(f"Patrolling to: {wp['name']}")
                
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.navigator.get_clock().now().to_msg()
                goal.pose.position.x = wp['x']
                goal.pose.position.y = wp['y']
                
                # Special orientation for Left Corridor (face south)
                if wp['name'] == 'Left Corridor Top':
                    goal.pose.orientation.z = 1.0
                    goal.pose.orientation.w = 0.0
                else:
                    # Default: face South
                    goal.pose.orientation.z = 0.0
                    goal.pose.orientation.w = 1.0

                self.navigator.goToPose(goal)
                self.nav_goal_sent = True
            
            # Check if waypoint reached
            elif self.navigator.isTaskComplete():
                # Enable search after reaching waypoint 3 (entering main room)
                if self.current_wp_index == 3:
                    self.search_enabled = True
                    self.get_logger().info("SEARCH ACTIVATED")
                
                # Advance to next waypoint (loop back to waypoint 3)
                self.current_wp_index += 1
                if self.current_wp_index >= len(self.waypoints):
                    self.current_wp_index = 3 
                self.nav_goal_sent = False

        # ============================================================
        # STATE 2: APPROACHING
        # Visual servoing to approach detected barrel
        # ============================================================
        elif self.state == State.APPROACHING:
            # --- NEW: TIMEOUT CHECK ---
            # If approaching for too long, give up and move to next waypoint
            if self.approach_start_time is not None:
                approach_elapsed = (self.get_clock().now() - self.approach_start_time).nanoseconds / 1e9
                if approach_elapsed > self.MAX_APPROACH_TIME:
                    self.get_logger().warn(
                        f"APPROACH TIMEOUT after {approach_elapsed:.1f}s! "
                        f"Giving up on this barrel.")
                    self.stop_robot()
                    
                    # Execute escape manoeuvre to get unstuck
                    self.escape_backward()
                    
                    # Reset and return to searching
                    self.state = State.SEARCHING
                    self.nav_goal_sent = False
                    self.current_target_size = 0
                    self.approach_start_time = None
                    
                    # Move to next waypoint to find different barrels
                    self.current_wp_index += 1
                    if self.current_wp_index >= len(self.waypoints):
                        self.current_wp_index = 3
                    
                    return
            
            target = self.get_best_barrel()
            
            # Safety: Return to searching if barrel lost
            if not target:
                self.get_logger().warn("Lost barrel! Back to patrol.")
                self.state = State.SEARCHING
                self.current_target_size = 0
                self.approach_start_time = None  # NEW: Reset timer
                return

            # Visual servoing parameters
            CAMERA_CENTER = 320     # Image center x-coordinate
            STOP_DISTANCE = 0.55    # Distance to stop from barrel (m)
            MIN_SIZE = 60000        # Minimum barrel size to confirm proximity

            twist = Twist()
            error = target.x - CAMERA_CENTER  # Pixel error from center

            # --- PHASE 1: ALIGN ---
            # Rotate in place to center barrel in camera view
            if self.collect_phase == CollectPhase.ALIGN:
                if abs(error) > 10:  # Deadband of 10 pixels
                    twist.angular.z = -0.002 * error  # P-controller
                    twist.angular.z = max(-0.5, min(0.5, twist.angular.z))  # Clamp
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.collect_phase = CollectPhase.APPROACH

            # --- PHASE 2: APPROACH ---
            # Drive forward while maintaining alignment
            elif self.collect_phase == CollectPhase.APPROACH:
                if self.front_dist < STOP_DISTANCE and target.size > MIN_SIZE:
                    # Close enough to barrel
                    self.stop_robot()
                    self.get_logger().info("Reached Barrel! Starting positioning...")
                    self.state = State.POSITIONING
                    self.navigator.clearAllCostmaps()
                    self.collect_phase = CollectPhase.TURN_AROUND
                    self.phase_start_time = self.get_clock().now()
                    self.current_target_size = 0
                    self.approach_start_time = None  # NEW: Reset timer on success
                else:
                    # Drive forward with steering correction
                    twist.linear.x = 0.15
                    
                    # Steering correction (only if significant error)
                    if abs(error) > 10:
                        steer = -0.0015 * error
                    else:
                        steer = 0.0

                    # Wall avoidance
                    if self.left_dist < 0.35:
                        steer -= 0.3
                    elif self.right_dist < 0.35:
                        steer += 0.3
                    
                    twist.angular.z = steer
                    self.cmd_vel_pub.publish(twist)

        # ============================================================
        # STATE 3: POSITIONING
        # Turn around and backup to position for pickup
        # ============================================================
        elif self.state == State.POSITIONING:
            t = self.elapsed()
            twist = Twist()

            # --- PHASE: TURN_AROUND ---
            # Rotate 180 degrees (pi radians at 0.5 rad/s = 6.28s)
            if self.collect_phase == CollectPhase.TURN_AROUND:
                TURN_DURATION = 6.4  # seconds
                if t < TURN_DURATION:
                    twist.angular.z = 0.5  # rad/s
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.collect_phase = CollectPhase.BACKUP
                    self.phase_start_time = self.get_clock().now()
                    self.get_logger().info("Turn Complete. Backing up...")

            # --- PHASE: BACKUP ---
            # Reverse toward barrel
            elif self.collect_phase == CollectPhase.BACKUP:
                BACKUP_TIME = 1.5  # seconds
                if t < BACKUP_TIME:
                    twist.linear.x = -0.15  # m/s (reverse)
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.get_logger().info("Backup Finished. Attempting Pickup...")
                    self.state = State.PICKING_UP
                    self.service_future = None

        # ============================================================
        # STATE 4: PICKING UP
        # Call pickup service to attach barrel
        # ============================================================
        elif self.state == State.PICKING_UP:
            # Send pickup request
            if self.service_future is None:
                req = ItemRequest.Request()
                req.robot_id = self.robot_name
                self.service_future = self.pickup_client.call_async(req)
            
            # Check for response
            elif self.service_future.done():
                try:
                    res = self.service_future.result()
                    if res.success:
                        self.get_logger().info("PICKUP SUCCESS!")
                        self.holding_barrel = True
                        self.set_mask(True)  # Enable LiDAR mask
                        self.navigator.clearAllCostmaps()
                        self.state = State.DELIVERING
                        self.nav_goal_sent = False
                    else:
                        self.get_logger().warn(f"Pickup Failed: {res.message}")
                        self.navigator.clearAllCostmaps()
                        self.state = State.SEARCHING
                        self.nav_goal_sent = False
                except Exception as e:
                    self.get_logger().error(f"Service error: {e}")
                    self.state = State.SEARCHING
                
                self.service_future = None

        # ============================================================
        # STATE 5: DELIVERING
        # Navigate to green collection zone
        # ============================================================
        elif self.state == State.DELIVERING:
            if not self.nav_goal_sent:
                # Zone configuration for barrel placement grid
                SPACING_X = 0.7     # Row spacing
                SPACING_Y = 0.7     # Column spacing
                ROW_LENGTH = 4      # Barrels per row
                ZONE_CAPACITY = 16  # Barrels per zone
                
                # Two collection zones
                zones = [
                    {'name': 'Zone B', 'start_x': 11.72, 'start_y': -15.5},
                    {'name': 'Zone A', 'start_x': 11.72, 'start_y': -21.8}
                ]

                # Calculate target position based on barrel count
                total_count = self.barrels_collected
                zone_index = (total_count // ZONE_CAPACITY) % len(zones)
                current_zone = zones[zone_index]
                local_index = total_count % ZONE_CAPACITY
                
                col = local_index % ROW_LENGTH
                row = local_index // ROW_LENGTH

                target_x = current_zone['start_x'] - (row * SPACING_X)
                target_y = current_zone['start_y'] + (col * SPACING_Y)

                self.get_logger().info(
                    f"Barrel #{total_count + 1} -> {current_zone['name']} "
                    f"at ({target_x:.2f}, {target_y:.2f})")
                
                # Navigate to target (facing west for reverse parking)
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.navigator.get_clock().now().to_msg()
                goal.pose.position.x = target_x
                goal.pose.position.y = target_y
                goal.pose.orientation.z = 1.0  # Face South
                goal.pose.orientation.w = 0.0
                
                self.navigator.goToPose(goal)
                self.nav_goal_sent = True
            
            # Check if arrived
            elif self.navigator.isTaskComplete():
                if self.navigator.getResult() == TaskResult.SUCCEEDED:
                    self.get_logger().info("Arrived. Starting Reverse Park...")
                    self.state = State.OFFLOADING
                    self.offload_start_time = self.get_clock().now()
                    self.service_future = None
                else:
                    self.get_logger().warn("Delivery Failed. Retrying...")
                    self.nav_goal_sent = False

        # ============================================================
        # STATE 6: OFFLOADING
        # Reverse into zone and drop barrel
        # ============================================================
        elif self.state == State.OFFLOADING:
            # Phase 1: Reverse manoeuvre
            t = (self.get_clock().now() - self.offload_start_time).nanoseconds / 1e9
            REVERSE_TIME = 1.8  # seconds
            
            if t < REVERSE_TIME:
                twist = Twist()
                twist.linear.x = -0.15  # Reverse
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.stop_robot()
            
            # Phase 2: Visual confirmation of green zone
            green_zone = self.get_zone(self.ZONE_GREEN)
            if green_zone:
                self.get_logger().info(f"GREEN zone confirmed! Size: {green_zone.size}")
            
            # Phase 3: Call offload service
            if self.service_future is None:
                req = ItemRequest.Request()
                req.robot_id = self.robot_name
                self.service_future = self.offload_client.call_async(req)
            
            elif self.service_future.done():
                try:
                    res = self.service_future.result()
                    if res.success:
                        self.get_logger().info("OFFLOAD SUCCESS!")
                        self.set_mask(False)  # Disable LiDAR mask
                        self.holding_barrel = False
                        self.barrels_collected += 1
                        
                        self.state = State.CLEARING_SPACE
                        self.forward_start_time = self.get_clock().now()
                    else:
                        self.get_logger().warn("Offload Failed.")
                        self.service_future = None
                except Exception as e:
                    self.get_logger().error(f"Offload error: {e}")
                self.service_future = None

        # ============================================================
        # STATE 7: CLEARING SPACE
        # Drive forward to clear drop zone, check radiation
        # ============================================================
        elif self.state == State.CLEARING_SPACE:
            t = (self.get_clock().now() - self.forward_start_time).nanoseconds / 1e9
            FORWARD_TIME = 1.0  # seconds

            if t < FORWARD_TIME:
                twist = Twist()
                twist.linear.x = 0.15  # Forward
                self.cmd_vel_pub.publish(twist)
            else:
                self.stop_robot()
                self.get_logger().info("Space cleared.")
                
                time.sleep(0.5)
                self.navigator.clearAllCostmaps()
                
                # Check if decontamination needed
                if self.should_decontaminate():
                    self.get_logger().info(
                        f"RADIATION: {self.radiation_level} >= {self.DECONTAMINATION_THRESHOLD}. "
                        f"Going to decontaminate!")
                    self.state = State.DECONTAMINATING
                    self.decontaminate_phase = DecontaminatePhase.NAVIGATING
                    self.nav_goal_sent = False
                else:
                    self.get_logger().info(
                        f"Radiation level: {self.radiation_level}. Resuming patrol.")
                    self.state = State.SEARCHING
                    self.nav_goal_sent = False
                    self.current_wp_index = 3
                    self.search_enabled = False

        # ============================================================
        # STATE 8: DECONTAMINATING
        # Navigate to cyan zone and call decontamination service
        # ============================================================
        elif self.state == State.DECONTAMINATING:
            
            # --- PHASE 1: NAVIGATE TO CYAN ZONE ---
            if self.decontaminate_phase == DecontaminatePhase.NAVIGATING:
                if not self.nav_goal_sent:
                    self.get_logger().info(
                        f"Navigating to decontamination zone at "
                        f"({self.decontamination_zone['x']:.2f}, "
                        f"{self.decontamination_zone['y']:.2f})")
                    
                    goal = PoseStamped()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = self.navigator.get_clock().now().to_msg()
                    goal.pose.position.x = self.decontamination_zone['x']
                    goal.pose.position.y = self.decontamination_zone['y']
                    goal.pose.orientation.z = 0.0  # Face South
                    goal.pose.orientation.w = 1.0
                    
                    self.navigator.goToPose(goal)
                    self.nav_goal_sent = True
                
                elif self.navigator.isTaskComplete():
                    if self.navigator.getResult() == TaskResult.SUCCEEDED:
                        self.get_logger().info("Arrived at decontamination zone. Reversing...")
                        self.decontaminate_phase = DecontaminatePhase.REVERSING
                        self.decontaminate_start_time = self.get_clock().now()
                    else:
                        self.get_logger().warn("Failed to reach decon zone. Retrying...")
                        self.nav_goal_sent = False
            
            # --- PHASE 2: REVERSE INTO ZONE ---
            elif self.decontaminate_phase == DecontaminatePhase.REVERSING:
                t = (self.get_clock().now() - self.decontaminate_start_time).nanoseconds / 1e9
                REVERSE_TIME = 1.5  # seconds
                
                if t < REVERSE_TIME:
                    twist = Twist()
                    twist.linear.x = -0.15  # Reverse
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.get_logger().info("In position. Calling decontaminate service...")
                    self.decontaminate_phase = DecontaminatePhase.CALLING_SERVICE
                    self.service_future = None
            
            # --- PHASE 3: CALL DECONTAMINATE SERVICE ---
            elif self.decontaminate_phase == DecontaminatePhase.CALLING_SERVICE:
                # Visual confirmation of cyan zone
                cyan_zone = self.get_zone(self.ZONE_CYAN)
                if cyan_zone:
                    self.get_logger().info(f"CYAN zone confirmed! Size: {cyan_zone.size}")
                
                # Send decontamination request
                if self.service_future is None:
                    if not self.decontaminate_client.wait_for_service(timeout_sec=0.5):
                        self.get_logger().warn("Decontaminate service not available...")
                        return
                    
                    req = ItemRequest.Request()
                    req.robot_id = self.robot_name
                    self.service_future = self.decontaminate_client.call_async(req)
                    self.get_logger().info(f"Decontaminate request sent for {self.robot_name}")
                
                elif self.service_future.done():
                    try:
                        res = self.service_future.result()
                        if res.success:
                            self.get_logger().info(f"DECONTAMINATION SUCCESS! {res.message}")
                        else:
                            self.get_logger().warn(f"Decontamination failed: {res.message}")
                    except Exception as e:
                        self.get_logger().error(f"Decontaminate service error: {e}")
                    
                    self.service_future = None
                    
                    # Clear costmaps and return to searching
                    time.sleep(0.5)
                    self.navigator.clearAllCostmaps()
                    
                    self.state = State.SEARCHING
                    self.nav_goal_sent = False
                    self.current_wp_index = 3
                    self.search_enabled = False
                    self.get_logger().info("Resuming patrol after decontamination.")

    # ================================================================
    # CLEANUP
    # ================================================================

    def destroy_node(self):
        """Clean shutdown: stop robot before destroying node."""
        self.stop_robot()
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the robot controller node.
    
    Initialises ROS2, creates the controller node, and spins
    until shutdown is requested.
    """
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RobotController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()