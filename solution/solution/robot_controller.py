import sys
import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

# Message types
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Assessment interfaces
from assessment_interfaces.msg import BarrelList, ZoneList, BarrelLog, BarrelHolders, RadiationList

# Service type
from auro_interfaces.srv import ItemRequest

# Nav2 action
from nav2_msgs.action import NavigateToPose


class State(Enum):
    SEARCHING = 0
    APPROACHING = 1
    POSITIONING = 2
    PICKING_UP = 3
    DELIVERING = 4
    OFFLOADING = 5
    DECONTAMINATING = 6


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Parameters
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.initial_x = self.get_parameter('x').get_parameter_value().double_value
        self.initial_y = self.get_parameter('y').get_parameter_value().double_value
        self.initial_yaw = self.get_parameter('yaw').get_parameter_value().double_value

        # Get robot namespace (e.g., "robot1")
        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name:
            self.robot_name = 'robot1'

        self.get_logger().info(f"Starting robot controller for {self.robot_name}")

        # ============================================================
        # STATE MACHINE
        # ============================================================
        self.state = State.SEARCHING
        self.previous_state = None

        # ============================================================
        # DATA STORAGE (updated by subscribers)
        # ============================================================
        self.barrels = []           # List of detected barrels from camera
        self.zones = []             # List of detected zones from camera
        self.robot_x = 0.0          # Robot position x
        self.robot_y = 0.0          # Robot position y
        self.robot_yaw = 0.0        # Robot orientation (yaw)
        self.radiation_level = 0    # Robot contamination level
        self.holding_barrel = False # Whether robot is holding a barrel
        self.scan_data = []         # LiDAR scan data

        # Target tracking
        self.target_barrel = None   # Current barrel we're going for
        self.last_barrel = None     # Barrel from last frame (for bonus scoring)
        self.centered_count = 0     # Count frames where barrel is centered

        # ============================================================
        # ZONE POSITIONS (from RViz measurements)
        # ============================================================
        # Green collection zones
        self.collection_zone_1 = {'x': 9.53, 'y': -6.45}
        self.collection_zone_2 = {'x': 9.41, 'y': -13.0}
        # Cyan decontamination zone
        self.decontamination_zone = {'x': 9.58, 'y': -0.33}

        # ============================================================
        # NAV2 WAYPOINTS - Patrol route to search for barrels
        # ============================================================
        self.waypoints = [
            # Starting area
            {'x': 0.05, 'y': 7.21, 'name': 'Start'},
            
            # Left corridor (bottom to top)
            {'x': 2.29, 'y': 8.91, 'name': 'Left corridor bottom'},
            {'x': 9.75, 'y': 8.84, 'name': 'Left corridor top'},
            
            # Big room patrol
            {'x': 10.05, 'y': 14.85, 'name': 'Big room entrance'},
            {'x': 6.15, 'y': 14.81, 'name': 'Big room bottom right'},
            {'x': 6.43, 'y': 19.25, 'name': 'Big room bottom center'},
            {'x': 6.37, 'y': 23.28, 'name': 'Big room bottom left'},
            {'x': 10.15, 'y': 23.14, 'name': 'Big room middle left'},
            {'x': 14.43, 'y': 23.02, 'name': 'Big room top left'},
            {'x': 14.26, 'y': 18.68, 'name': 'Big room top middle'},
            {'x': 14.29, 'y': 14.88, 'name': 'Big room top right'},
            {'x': 10.13, 'y': 19.61, 'name': 'Big room center'},
            
            # Return via big room entrance
            {'x': 10.05, 'y': 14.85, 'name': 'Big room entrance'},
            
            # Right corridor (top to bottom)
            {'x': 9.35, 'y': 4.68, 'name': 'Right corridor top'},
            {'x': 2.26, 'y': 5.95, 'name': 'Right corridor bottom'},
            
            # Back to start
            {'x': 0.05, 'y': 7.21, 'name': 'Start'},
        ]
        self.current_waypoint_index = 0

        # ============================================================
        # CALLBACK GROUPS (for async service/action calls)
        # ============================================================
        self.callback_group = ReentrantCallbackGroup()

        # ============================================================
        # SUBSCRIBERS
        # ============================================================
        
        # Barrel detection from camera (namespaced)
        self.barrel_subscriber = self.create_subscription(
            BarrelList,
            'barrels',
            self.barrel_callback,
            10
        )

        # Zone detection from camera (namespaced)
        self.zone_subscriber = self.create_subscription(
            ZoneList,
            'zones',
            self.zone_callback,
            10
        )

        # Robot odometry (namespaced)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )

        # LiDAR scan data (namespaced, filtered)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            'scan_filtered',
            self.scan_callback,
            10
        )

        # Barrel collection log (global)
        self.barrel_log_subscriber = self.create_subscription(
            BarrelLog,
            '/barrel_log',
            self.barrel_log_callback,
            10
        )

        # Which robot holds which barrel (global)
        self.barrel_holders_subscriber = self.create_subscription(
            BarrelHolders,
            '/barrel_holders',
            self.barrel_holders_callback,
            10
        )

        # Radiation levels for all robots (global)
        self.radiation_subscriber = self.create_subscription(
            RadiationList,
            '/radiation_levels',
            self.radiation_callback,
            10
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================
        
        # Velocity commands for direct robot control (namespaced)
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        # Initial pose publisher for AMCL (use absolute path)
        initialpose_topic = f'/{self.robot_name}/initialpose'
        self.get_logger().info(f"Publishing initial pose to: {initialpose_topic}")
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            initialpose_topic,
            10
        )

        # Flag to publish initial pose only once
        self.initial_pose_set = False

        # ============================================================
        # SERVICE CLIENTS
        # ============================================================
        
        # Pick up barrel service (global)
        self.pick_up_client = self.create_client(
            ItemRequest,
            '/pick_up_item',
            callback_group=self.callback_group
        )

        # Offload barrel service (global)
        self.offload_client = self.create_client(
            ItemRequest,
            '/offload_item',
            callback_group=self.callback_group
        )

        # Decontaminate robot service (global)
        self.decontaminate_client = self.create_client(
            ItemRequest,
            '/decontaminate',
            callback_group=self.callback_group
        )

        # ============================================================
        # NAV2 ACTION CLIENT
        # ============================================================
        
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
            callback_group=self.callback_group
        )

        # Navigation state tracking
        self.goal_handle = None
        self.navigation_complete = False
        self.navigation_result = None
        self.navigation_started = False  # Track if we've sent a goal

        # ============================================================
        # STARTUP DELAY
        # ============================================================
        self.startup_delay_count = 0
        self.startup_delay_max = 900  # Wait 90 seconds (900 x 0.1s) for Nav2 to fully initialize
        self.nav2_ready = False
        
        # Delay between waypoints
        self.waypoint_delay_count = 0
        self.waypoint_delay_max = 20  # Wait 2 seconds between waypoints
        self.waiting_between_waypoints = False

        # ============================================================
        # CONTROL LOOP TIMER
        # ============================================================
        self.timer_period = 0.1  # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info(f"Initial pose - x: {self.initial_x}, y: {self.initial_y}, yaw: {self.initial_yaw}")
        self.get_logger().info("Robot controller initialized. Starting in SEARCHING state.")

    # ================================================================
    # SUBSCRIBER CALLBACKS
    # ================================================================

    def barrel_callback(self, msg):
        """Callback for barrel detection from camera."""
        self.barrels = msg.data
        # Debug: log when barrels are detected
        if len(self.barrels) > 0:
            self.get_logger().info(f"Barrel callback: detected {len(self.barrels)} barrel(s)")

    def zone_callback(self, msg):
        """Callback for zone detection from camera."""
        self.zones = msg.data

    def odom_callback(self, msg):
        """Callback for robot odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        """Callback for LiDAR scan data."""
        self.scan_data = msg.ranges

    def barrel_log_callback(self, msg):
        """Callback for barrel collection log."""
        pass

    def barrel_holders_callback(self, msg):
        """Callback for barrel holders info."""
        self.holding_barrel = False
        for holder in msg.data:
            if holder.robot_id == self.robot_name:
                self.holding_barrel = True
                break

    def radiation_callback(self, msg):
        """Callback for radiation levels."""
        for radiation in msg.data:
            if radiation.robot_id == self.robot_name:
                self.radiation_level = radiation.level
                break

    # ================================================================
    # HELPER METHODS
    # ================================================================

    def set_initial_pose(self):
        """Publish initial pose for AMCL."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Map coordinates (found by manually setting pose in RViz)
        # The Gazebo spawn (0, -2) corresponds to map position (0.08, 7.23)
        map_x = 0.08
        map_y = 7.23
        # Rotate 90 degrees to the right (-π/2 radians)
        map_yaw = self.initial_yaw - (math.pi / 2.0)
        
        msg.pose.pose.position.x = map_x
        msg.pose.pose.position.y = map_y
        msg.pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(map_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(map_yaw / 2.0)
        
        # Set covariance (small values = high confidence)
        msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.068]
        
        self.initial_pose_publisher.publish(msg)
        self.get_logger().info(f"Published initial pose: x={map_x}, y={map_y}, yaw={map_yaw}")
        
        # Republish a few times to ensure AMCL receives it
        self._republish_count = 0
        self._map_x = map_x
        self._map_y = map_y
        self._map_yaw = map_yaw
        self.republish_timer = self.create_timer(0.5, self._republish_initial_pose)
        
    def _republish_initial_pose(self):
        """Republish initial pose a few times to ensure AMCL gets it."""
        self._republish_count += 1
            
        if self._republish_count <= 3:
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.position.x = self._map_x
            msg.pose.pose.position.y = self._map_y
            msg.pose.pose.orientation.z = math.sin(self._map_yaw / 2.0)
            msg.pose.pose.orientation.w = math.cos(self._map_yaw / 2.0)
            msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.068]
            self.initial_pose_publisher.publish(msg)
        else:
            # Stop the timer after 3 republishes
            self.republish_timer.cancel()

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)

    def distance_to(self, x, y):
        """Calculate distance from robot to a point."""
        return math.sqrt((x - self.robot_x)**2 + (y - self.robot_y)**2)

    # ================================================================
    # NAV2 METHODS
    # ================================================================

    def navigate_to_pose(self, x, y, yaw=0.0):
        """Send a navigation goal to Nav2."""
        
        # Wait for Nav2 action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return False

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending Nav2 goal: x={x:.2f}, y={y:.2f}")
        
        # Reset navigation state
        self.navigation_complete = False
        self.navigation_result = None
        
        # Send goal
        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_goal_response_callback)
        
        return True

    def nav_goal_response_callback(self, future):
        """Callback when Nav2 goal is accepted/rejected."""
        self.goal_handle = future.result()
        
        if not self.goal_handle.accepted:
            self.get_logger().warn("Nav2 goal was rejected!")
            self.navigation_complete = True
            self.navigation_result = 'rejected'
            return

        self.get_logger().info("Nav2 goal accepted, navigating...")
        
        # Get result when navigation completes
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        """Callback for Nav2 navigation feedback."""
        pass

    def nav_result_callback(self, future):
        """Callback when Nav2 navigation completes."""
        result = future.result()
        status = result.status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Navigation succeeded!")
            self.navigation_result = 'succeeded'
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn("Navigation aborted!")
            self.navigation_result = 'aborted'
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn("Navigation canceled!")
            self.navigation_result = 'canceled'
        else:
            self.get_logger().warn(f"Navigation finished with status: {status}")
            self.navigation_result = 'unknown'
        
        self.navigation_complete = True

    def cancel_navigation(self):
        """Cancel current navigation goal."""
        if self.goal_handle is not None:
            self.get_logger().info("Cancelling navigation...")
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
            self.navigation_complete = True
            self.navigation_result = 'canceled'

    # ================================================================
    # BARREL SELECTION
    # ================================================================

    def select_best_barrel(self):
        """Select the best barrel to target based on size and center position."""
        
        CAMERA_CENTER = 320.0       # Camera center in pixels
        MIN_BARREL_SIZE = 50        # Minimum size to consider
        LAST_BARREL_BONUS = 100     # Bonus for same barrel as last frame
        
        if len(self.barrels) == 0:
            return None
        
        best_barrel = None
        best_score = -999999
        
        for barrel in self.barrels:
            # Skip barrels that are too small (too far away)
            if barrel.size < MIN_BARREL_SIZE:
                continue
            
            # Calculate center penalty (how far from camera center)
            # barrel.y is the horizontal offset in pixels
            center_penalty = abs(barrel.y)
            
            # Score = size - center_penalty (bigger and more centered = better)
            score = barrel.size - center_penalty
            
            # Add bonus if this is the same barrel as last frame
            if self.last_barrel is not None:
                size_diff = abs(barrel.size - self.last_barrel.size)
                pos_diff = abs(barrel.y - self.last_barrel.y)
                # If similar size and position, it's probably the same barrel
                if size_diff < 500 and pos_diff < 100:
                    score += LAST_BARREL_BONUS
            
            # Update best barrel if this one has higher score
            if score > best_score:
                best_score = score
                best_barrel = barrel
        
        return best_barrel

    # ================================================================
    # STATE METHODS (TODO: Implement these)
    # ================================================================

    def searching(self):
        """SEARCHING state: Navigate to waypoints using Nav2, look for barrels."""
        
        # Check if we see any barrels while navigating
        if len(self.barrels) > 0:
            # Find the best barrel to target
            best_barrel = self.select_best_barrel()
            
            if best_barrel is not None:
                self.get_logger().info(
                    f"Found barrel! Colour: {best_barrel.colour}, "
                    f"Size: {best_barrel.size:.1f}, Y: {best_barrel.y:.1f}"
                )
                
                # Cancel current navigation
                self.cancel_navigation()
                
                # Set target and switch state
                self.target_barrel = best_barrel
                self.last_barrel = best_barrel
                self.state = State.APPROACHING
                
                # Reset navigation flags
                self.navigation_started = False
                self.navigation_complete = False
                self.waiting_between_waypoints = False
                return
        
        # If waiting between waypoints, count down
        if self.waiting_between_waypoints:
            self.waypoint_delay_count += 1
            if self.waypoint_delay_count >= self.waypoint_delay_max:
                self.waiting_between_waypoints = False
                self.waypoint_delay_count = 0
                self.get_logger().info("Delay complete, sending next waypoint")
            return
        
        # Check if we need to start a new navigation
        if not self.navigation_started:
            # Start navigating to current waypoint
            waypoint = self.waypoints[self.current_waypoint_index]
            self.get_logger().info(
                f"SEARCHING: Navigating to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                f"'{waypoint['name']}' at ({waypoint['x']:.2f}, {waypoint['y']:.2f})"
            )
            self.navigate_to_pose(waypoint['x'], waypoint['y'])
            self.navigation_started = True
            return
        
        # Wait for navigation to complete
        if not self.navigation_complete:
            # Still navigating, do nothing
            return
        
        # Navigation completed, process result
        waypoint = self.waypoints[self.current_waypoint_index]
        if self.navigation_result == 'succeeded':
            self.get_logger().info(f"Reached waypoint '{waypoint['name']}'")
        else:
            self.get_logger().warn(f"Navigation to '{waypoint['name']}' failed: {self.navigation_result}")
        
        # Move to next waypoint
        self.current_waypoint_index += 1
        if self.current_waypoint_index >= len(self.waypoints):
            self.current_waypoint_index = 0  # Loop back
            self.get_logger().info("Completed all waypoints, starting patrol again")
        
        # Reset for next navigation
        self.goal_handle = None
        self.navigation_complete = False
        self.navigation_started = False
        
        # Add delay before next waypoint
        self.waiting_between_waypoints = True
        self.waypoint_delay_count = 0
        self.get_logger().info("Waiting before next waypoint...")

    def approaching(self):
        """APPROACHING state: Drive towards barrel using camera feedback."""
        
        # Constants
        CAMERA_CENTER = 320.0       # Camera center in pixels
        STOP_SIZE = 15000           # Size threshold to stop (close to barrel)
        SLOW_SIZE = 8000            # Size threshold to slow down
        FAST_SPEED = 0.25           # Speed when far from barrel
        SLOW_SPEED = 0.15           # Speed when close to barrel
        MAX_TURN = 0.5              # Maximum angular velocity
        TURN_GAIN = 0.002           # Proportional gain for turning
        WALL_CLEARANCE = 0.35       # Minimum distance to walls (meters)
        CENTERED_THRESHOLD = 30     # Pixels from center to be "centered"
        CENTERED_FRAMES_NEEDED = 3  # Frames needed to confirm centered
        
        # If no target, go back to searching
        if self.target_barrel is None:
            self.get_logger().warn("APPROACHING: No target barrel, returning to SEARCHING")
            self.state = State.SEARCHING
            return
        
        # Check if we still see barrels
        if len(self.barrels) == 0:
            self.get_logger().warn("APPROACHING: Lost sight of barrel, returning to SEARCHING")
            self.target_barrel = None
            self.last_barrel = None
            self.state = State.SEARCHING
            return
        
        # Update target to best barrel (maintains tracking)
        best_barrel = self.select_best_barrel()
        if best_barrel is not None:
            self.target_barrel = best_barrel
            self.last_barrel = best_barrel
        
        # Get barrel info
        barrel_size = self.target_barrel.size
        barrel_y = self.target_barrel.y  # Horizontal offset from center (positive = right)
        
        # Calculate steering angle
        # error_x: positive means barrel is to the right, need to turn right (negative angular.z)
        error_x = barrel_y
        turn = -TURN_GAIN * error_x  # Negative because positive error needs negative turn
        
        # Cap the turn rate
        turn = max(-MAX_TURN, min(MAX_TURN, turn))
        
        # Check LiDAR for wall clearance
        if len(self.scan_data) > 0:
            # Left side (around 45-90 degrees)
            left_ranges = [r for r in self.scan_data[30:90] if 0.1 < r < 10.0]
            # Right side (around 270-330 degrees)
            right_ranges = [r for r in self.scan_data[270:330] if 0.1 < r < 10.0]
            
            left_min = min(left_ranges) if left_ranges else 10.0
            right_min = min(right_ranges) if right_ranges else 10.0
            
            # Adjust turn if too close to wall
            if left_min < WALL_CLEARANCE and turn > 0:
                turn = -0.2  # Turn right instead
                self.get_logger().info(f"Wall on left ({left_min:.2f}m), turning right")
            elif right_min < WALL_CLEARANCE and turn < 0:
                turn = 0.2  # Turn left instead
                self.get_logger().info(f"Wall on right ({right_min:.2f}m), turning left")
        
        # Determine speed based on barrel size
        if barrel_size > STOP_SIZE:
            # Very close to barrel - check if centered
            if abs(error_x) < CENTERED_THRESHOLD:
                self.centered_count += 1
                self.get_logger().info(f"APPROACHING: Centered! Count: {self.centered_count}/{CENTERED_FRAMES_NEEDED}")
                
                if self.centered_count >= CENTERED_FRAMES_NEEDED:
                    # Close enough and centered - switch to POSITIONING
                    self.get_logger().info("APPROACHING: Close and centered, switching to POSITIONING")
                    self.stop_robot()
                    self.centered_count = 0
                    self.state = State.POSITIONING
                    return
            else:
                self.centered_count = 0
            
            # Stop forward motion, just turn to center
            speed = 0.0
            
        elif barrel_size > SLOW_SIZE:
            # Getting close - slow down
            speed = SLOW_SPEED
            self.centered_count = 0
        else:
            # Far away - go faster
            speed = FAST_SPEED
            self.centered_count = 0
        
        # Publish velocity command
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = turn
        self.cmd_vel_publisher.publish(twist)
        
        self.get_logger().info(
            f"APPROACHING: size={barrel_size:.0f}, y={barrel_y:.0f}, speed={speed:.2f}, turn={turn:.2f}",
            throttle_duration_sec=0.5
        )

    def positioning(self):
        """POSITIONING state: Maneuver barrel behind robot."""
        pass

    def picking_up(self):
        """PICKING_UP state: Call pick_up service."""
        pass

    def delivering(self):
        """DELIVERING state: Navigate to collection zone."""
        pass

    def offloading(self):
        """OFFLOADING state: Call offload service."""
        pass

    def decontaminating(self):
        """DECONTAMINATING state: Go to cyan zone and decontaminate."""
        pass

    # ================================================================
    # MAIN CONTROL LOOP
    # ================================================================

    def control_loop(self):
        """Main control loop - runs at 10 Hz."""
        
        # Set initial pose for AMCL (only once)
        if not self.initial_pose_set:
            self.set_initial_pose()
            self.initial_pose_set = True
            return  # Give AMCL time to process
        
        # Wait for Nav2 to be ready after initial pose
        if not self.nav2_ready:
            self.startup_delay_count += 1
            if self.startup_delay_count >= self.startup_delay_max:
                self.nav2_ready = True
                self.get_logger().info("Nav2 startup delay complete, beginning navigation")
            elif self.startup_delay_count % 100 == 0:  # Log every 10 seconds
                seconds_waited = self.startup_delay_count / 10
                seconds_total = self.startup_delay_max / 10
                self.get_logger().info(f"Waiting for Nav2... {seconds_waited:.0f}/{seconds_total:.0f} seconds")
            return
        
        # Log state changes
        if self.state != self.previous_state:
            self.get_logger().info(f"State: {self.previous_state} -> {self.state}")
            self.previous_state = self.state

        # Execute current state
        if self.state == State.SEARCHING:
            self.searching()
        elif self.state == State.APPROACHING:
            self.approaching()
        elif self.state == State.POSITIONING:
            self.positioning()
        elif self.state == State.PICKING_UP:
            self.picking_up()
        elif self.state == State.DELIVERING:
            self.delivering()
        elif self.state == State.OFFLOADING:
            self.offloading()
        elif self.state == State.DECONTAMINATING:
            self.decontaminating()

    # ================================================================
    # CLEANUP
    # ================================================================

    def destroy_node(self):
        self.cancel_navigation()
        self.stop_robot()
        super().destroy_node()


def main(args=None):
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