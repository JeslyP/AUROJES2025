#!/usr/bin/env python3
"""
AURO 2025 Coursework - Autonomous Barrel Collection Robot Controller

This module implements an autonomous robot controller for collecting barrels
and depositing them in collection zones. It uses Nav2 for navigation and
visual servoing for precise barrel pickup.

State Machine:
    SEARCHING -> APPROACHING_BARREL -> ALIGNING_FOR_PICKUP -> PICKING_UP
    -> NAVIGATING_TO_ZONE -> DEPOSITING -> (back to SEARCHING)
    
    If contaminated (holding red barrel):
    PICKING_UP -> NAVIGATING_TO_DECONTAMINATION -> DECONTAMINATING -> NAVIGATING_TO_ZONE
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from std_srvs.srv import Empty

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

# Assessment interfaces - these are provided by the assessment package
from assessment_interfaces.msg import ItemList, ZoneList, Item, Zone, ItemLog
from assessment_interfaces.srv import PickUpItem, OffloadItem, Decontaminate

import math
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple, List
import time


class RobotState(Enum):
    """Robot state machine states"""
    INITIALIZING = auto()
    SEARCHING = auto()
    APPROACHING_BARREL = auto()
    ALIGNING_FOR_PICKUP = auto()
    PICKING_UP = auto()
    NAVIGATING_TO_DECONTAMINATION = auto()
    DECONTAMINATING = auto()
    NAVIGATING_TO_ZONE = auto()
    DEPOSITING = auto()
    RECOVERY = auto()


class BarrelColor(Enum):
    """Barrel color types"""
    RED = "red"      # Contaminated
    BLUE = "blue"    # Non-contaminated


class RobotController(Node):
    """
    Main robot controller for autonomous barrel collection.
    
    This controller implements:
    - State machine for task sequencing
    - Nav2 integration for waypoint navigation
    - Visual servoing for precise barrel approach
    - Service calls for pickup/offload/decontamination
    - Obstacle avoidance using LIDAR
    """

    # Zone coordinates (centers of 3m x 3m zones)
    COLLECTION_ZONE_A = (13.5, 9.4)
    COLLECTION_ZONE_B = (19.5, 9.4)
    DECONTAMINATION_ZONE = (7.5, 9.4)
    
    # Search waypoints - cover the map systematically
    SEARCH_WAYPOINTS = [
        (3.0, 3.0), (7.0, 3.0), (11.0, 3.0), (15.0, 3.0), (19.0, 3.0),
        (19.0, 7.0), (15.0, 7.0), (11.0, 7.0), (7.0, 7.0), (3.0, 7.0),
        (3.0, 11.0), (7.0, 11.0), (11.0, 11.0), (15.0, 11.0), (19.0, 11.0),
        (19.0, 15.0), (15.0, 15.0), (11.0, 15.0), (7.0, 15.0), (3.0, 15.0),
    ]
    
    # Pickup geometry constraints
    PICKUP_DISTANCE = 0.45      # Maximum distance for pickup (meters)
    PICKUP_ANGLE = 15.0         # Maximum angle from behind robot (degrees)
    
    # Visual servoing parameters
    APPROACH_LINEAR_SPEED = 0.15
    APPROACH_ANGULAR_SPEED = 0.3
    ALIGNMENT_ANGULAR_SPEED = 0.2
    
    # Safety parameters
    OBSTACLE_DISTANCE = 0.35    # Minimum distance to obstacles
    
    def __init__(self, robot_namespace: str = ''):
        super().__init__('robot_controller')
        
        self.namespace = robot_namespace
        self.callback_group = ReentrantCallbackGroup()
        
        # State machine
        self.state = RobotState.INITIALIZING
        self.previous_state = None
        
        # Robot state tracking
        self.current_pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw
        self.holding_barrel = False
        self.holding_barrel_color: Optional[BarrelColor] = None
        self.is_contaminated = False
        self.radiation_level = 0.0
        
        # Detected objects
        self.detected_barrels: List[Item] = []
        self.detected_zones: List[Zone] = []
        self.target_barrel: Optional[Item] = None
        
        # Navigation state
        self.current_waypoint_index = 0
        self.nav_goal_active = False
        self.nav_goal_result = None
        
        # LIDAR data
        self.laser_ranges: Optional[np.ndarray] = None
        self.obstacle_detected = False
        
        # Collected barrels tracking
        self.collected_barrel_ids = set()
        self.total_collected = 0
        
        # Timing
        self.state_start_time = time.time()
        self.pickup_attempts = 0
        self.max_pickup_attempts = 3
        
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        self._setup_action_clients()
        
        # Main control timer (10 Hz)
        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.callback_group
        )
        
        self.get_logger().info(f'Robot Controller initialized (namespace: {self.namespace})')

    def _setup_publishers(self):
        """Set up ROS publishers"""
        prefix = f'{self.namespace}/' if self.namespace else ''
        
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{prefix}cmd_vel', 10
        )
        
    def _setup_subscribers(self):
        """Set up ROS subscribers"""
        prefix = f'{self.namespace}/' if self.namespace else ''
        
        # Odometry for pose tracking
        self.odom_sub = self.create_subscription(
            Odometry, f'{prefix}odom', self.odom_callback, 10,
            callback_group=self.callback_group
        )
        
        # LIDAR for obstacle detection
        self.laser_sub = self.create_subscription(
            LaserScan, f'{prefix}scan', self.laser_callback, 10,
            callback_group=self.callback_group
        )
        
        # Visual detection of barrels (from item_sensor node)
        self.barrels_sub = self.create_subscription(
            ItemList, f'{prefix}items', self.barrels_callback, 10,
            callback_group=self.callback_group
        )
        
        # Visual detection of zones (from item_sensor node)
        self.zones_sub = self.create_subscription(
            ZoneList, f'{prefix}zones', self.zones_callback, 10,
            callback_group=self.callback_group
        )
        
        # Radiation level monitoring
        self.radiation_sub = self.create_subscription(
            Float32, f'{prefix}radiation_level', self.radiation_callback, 10,
            callback_group=self.callback_group
        )
        
    def _setup_services(self):
        """Set up service clients"""
        prefix = f'{self.namespace}/' if self.namespace else ''
        
        self.pickup_client = self.create_client(
            PickUpItem, f'{prefix}pick_up_item',
            callback_group=self.callback_group
        )
        
        self.offload_client = self.create_client(
            OffloadItem, f'{prefix}offload_item',
            callback_group=self.callback_group
        )
        
        self.decontaminate_client = self.create_client(
            Decontaminate, f'{prefix}decontaminate',
            callback_group=self.callback_group
        )
        
    def _setup_action_clients(self):
        """Set up Nav2 action client"""
        prefix = f'{self.namespace}/' if self.namespace else ''
        
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, f'{prefix}navigate_to_pose',
            callback_group=self.callback_group
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def odom_callback(self, msg: Odometry):
        """Process odometry data to track robot pose"""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1.0 - 2.0 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.current_pose = (pos.x, pos.y, yaw)
        
    def laser_callback(self, msg: LaserScan):
        """Process LIDAR data for obstacle detection"""
        self.laser_ranges = np.array(msg.ranges)
        
        # Check front sector for obstacles (roughly -30 to +30 degrees)
        num_ranges = len(self.laser_ranges)
        front_start = int(num_ranges * 0.4)  # -36 degrees
        front_end = int(num_ranges * 0.6)    # +36 degrees
        
        front_ranges = self.laser_ranges[front_start:front_end]
        valid_ranges = front_ranges[np.isfinite(front_ranges)]
        
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.obstacle_detected = min_distance < self.OBSTACLE_DISTANCE
        else:
            self.obstacle_detected = False
            
    def barrels_callback(self, msg: ItemList):
        """Process detected barrels from visual sensor"""
        self.detected_barrels = list(msg.data)
        
    def zones_callback(self, msg: ZoneList):
        """Process detected zones from visual sensor"""
        self.detected_zones = list(msg.data)
        
    def radiation_callback(self, msg: Float32):
        """Monitor radiation level"""
        self.radiation_level = msg.data
        self.is_contaminated = self.radiation_level > 0.1

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    
    def control_loop(self):
        """Main state machine control loop"""
        if self.current_pose is None:
            return  # Wait for odometry
            
        # Log state changes
        if self.state != self.previous_state:
            self.get_logger().info(f'State: {self.previous_state} -> {self.state}')
            self.previous_state = self.state
            self.state_start_time = time.time()
            
        # Execute current state
        if self.state == RobotState.INITIALIZING:
            self._state_initializing()
        elif self.state == RobotState.SEARCHING:
            self._state_searching()
        elif self.state == RobotState.APPROACHING_BARREL:
            self._state_approaching_barrel()
        elif self.state == RobotState.ALIGNING_FOR_PICKUP:
            self._state_aligning_for_pickup()
        elif self.state == RobotState.PICKING_UP:
            self._state_picking_up()
        elif self.state == RobotState.NAVIGATING_TO_DECONTAMINATION:
            self._state_navigating_to_decontamination()
        elif self.state == RobotState.DECONTAMINATING:
            self._state_decontaminating()
        elif self.state == RobotState.NAVIGATING_TO_ZONE:
            self._state_navigating_to_zone()
        elif self.state == RobotState.DEPOSITING:
            self._state_depositing()
        elif self.state == RobotState.RECOVERY:
            self._state_recovery()

    # =========================================================================
    # STATE IMPLEMENTATIONS
    # =========================================================================
    
    def _state_initializing(self):
        """Wait for all systems to be ready"""
        # Check if Nav2 is available
        if self.nav_to_pose_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().info('Nav2 ready, starting search')
            self.state = RobotState.SEARCHING
        else:
            self.get_logger().info('Waiting for Nav2...', throttle_duration_sec=2.0)
            
    def _state_searching(self):
        """Search for barrels by navigating through waypoints"""
        # Check if we see any barrels
        available_barrels = self._get_available_barrels()
        
        if available_barrels:
            # Found a barrel! Select the closest one
            self.target_barrel = self._select_best_barrel(available_barrels)
            if self.target_barrel:
                self.get_logger().info(
                    f'Found barrel: color={self.target_barrel.colour}, '
                    f'size={self.target_barrel.diameter:.3f}'
                )
                self.state = RobotState.APPROACHING_BARREL
                self.pickup_attempts = 0
                self._cancel_navigation()
                return
                
        # Continue searching - navigate to waypoints
        if not self.nav_goal_active:
            self._navigate_to_next_waypoint()
            
    def _state_approaching_barrel(self):
        """Approach the target barrel using visual servoing"""
        if self.target_barrel is None:
            self.state = RobotState.SEARCHING
            return
            
        # Look for the target barrel in current detections
        current_barrel = self._find_barrel_in_view()
        
        if current_barrel is None:
            # Lost sight of barrel - rotate to find it
            elapsed = time.time() - self.state_start_time
            if elapsed > 5.0:
                self.get_logger().warn('Lost barrel, returning to search')
                self.target_barrel = None
                self.state = RobotState.SEARCHING
                return
            self._rotate_to_find_barrel()
            return
            
        # Visual servoing: barrel x offset in image tells us angular error
        # x < 0 means barrel is to the left, x > 0 means to the right
        x_offset = current_barrel.x  # Normalized offset from center
        barrel_size = current_barrel.diameter  # Size indicates distance
        
        # Estimate distance from barrel size (larger = closer)
        # This is approximate - tune based on your camera setup
        estimated_distance = self._estimate_distance_from_size(barrel_size)
        
        cmd = Twist()
        
        # Angular control: center the barrel in view
        if abs(x_offset) > 0.05:  # Dead zone
            cmd.angular.z = -self.APPROACH_ANGULAR_SPEED * np.sign(x_offset) * min(abs(x_offset) * 2, 1.0)
        
        # Linear control: approach until close enough
        if estimated_distance > self.PICKUP_DISTANCE + 0.1:
            # Check for obstacles
            if self.obstacle_detected:
                cmd.linear.x = 0.0
                self.get_logger().warn('Obstacle detected during approach')
            else:
                cmd.linear.x = self.APPROACH_LINEAR_SPEED * min(estimated_distance / 2.0, 1.0)
        else:
            # Close enough, start alignment for pickup
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.state = RobotState.ALIGNING_FOR_PICKUP
            return
            
        self.cmd_vel_pub.publish(cmd)
        
    def _state_aligning_for_pickup(self):
        """
        Align robot so barrel is directly BEHIND it (for pickup geometry).
        The barrel must be within ±15° of directly behind the robot.
        """
        current_barrel = self._find_barrel_in_view()
        
        if current_barrel is None:
            # Lost the barrel during alignment
            elapsed = time.time() - self.state_start_time
            if elapsed > 3.0:
                self.state = RobotState.APPROACHING_BARREL
            return
            
        cmd = Twist()
        
        # We need to rotate 180° so the barrel is behind us
        # First, turn away from the barrel, then back up toward it
        
        x_offset = current_barrel.x
        
        # Phase 1: Turn so barrel is behind us (we lose sight of it)
        # We're going to turn right (negative angular) if barrel was on our left
        # This is a 180 degree turn maneuver
        
        # Simpler approach: back up toward the barrel while keeping it in view
        # Then when very close, call pickup
        
        barrel_size = current_barrel.diameter
        estimated_distance = self._estimate_distance_from_size(barrel_size)
        
        # Center the barrel first
        if abs(x_offset) > 0.03:
            cmd.angular.z = -self.ALIGNMENT_ANGULAR_SPEED * np.sign(x_offset)
            cmd.linear.x = 0.0
        else:
            # Barrel is centered, now we need to get it behind us
            # Turn 180 degrees and back up
            if estimated_distance > self.PICKUP_DISTANCE:
                # Still need to get closer - back up toward it
                # But we're facing it... so we need to turn around
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn around
            else:
                # We're close enough - try pickup
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
                self.state = RobotState.PICKING_UP
                return
                
        self.cmd_vel_pub.publish(cmd)
        
    def _state_picking_up(self):
        """Attempt to pick up the barrel"""
        self.pickup_attempts += 1
        
        # Stop the robot
        self.cmd_vel_pub.publish(Twist())
        
        # Call the pickup service
        if not self.pickup_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Pickup service not available')
            self.state = RobotState.RECOVERY
            return
            
        request = PickUpItem.Request()
        
        future = self.pickup_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            result = future.result()
            if result.success:
                self.get_logger().info(f'Successfully picked up barrel!')
                self.holding_barrel = True
                
                # Determine barrel color from what we were tracking
                if self.target_barrel:
                    color_str = self.target_barrel.colour.lower()
                    if 'red' in color_str:
                        self.holding_barrel_color = BarrelColor.RED
                    else:
                        self.holding_barrel_color = BarrelColor.BLUE
                        
                self.total_collected += 1
                self.target_barrel = None
                
                # Decide next state based on contamination
                if self.holding_barrel_color == BarrelColor.RED:
                    self.state = RobotState.NAVIGATING_TO_DECONTAMINATION
                else:
                    self.state = RobotState.NAVIGATING_TO_ZONE
            else:
                self.get_logger().warn(f'Pickup failed: {result.message}')
                if self.pickup_attempts >= self.max_pickup_attempts:
                    self.get_logger().error('Max pickup attempts reached')
                    self.target_barrel = None
                    self.state = RobotState.SEARCHING
                else:
                    # Try to realign
                    self.state = RobotState.ALIGNING_FOR_PICKUP
        else:
            self.get_logger().error('Pickup service call failed')
            self.state = RobotState.RECOVERY
            
    def _state_navigating_to_decontamination(self):
        """Navigate to decontamination zone"""
        if not self.nav_goal_active:
            self._navigate_to_point(
                self.DECONTAMINATION_ZONE[0],
                self.DECONTAMINATION_ZONE[1],
                0.0  # Face any direction
            )
            
        # Check if we've arrived
        if self._is_in_zone(self.DECONTAMINATION_ZONE):
            self._cancel_navigation()
            self.state = RobotState.DECONTAMINATING
            
    def _state_decontaminating(self):
        """Call decontamination service"""
        self.cmd_vel_pub.publish(Twist())
        
        if not self.decontaminate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Decontaminate service not available')
            self.state = RobotState.RECOVERY
            return
            
        request = Decontaminate.Request()
        future = self.decontaminate_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Decontamination successful!')
            self.is_contaminated = False
            self.state = RobotState.NAVIGATING_TO_ZONE
        else:
            self.get_logger().warn('Decontamination failed, retrying...')
            # Stay in this state to retry
            
    def _state_navigating_to_zone(self):
        """Navigate to collection zone to deposit barrel"""
        if not self.holding_barrel:
            self.state = RobotState.SEARCHING
            return
            
        # Choose the closer collection zone
        zone = self._get_closer_collection_zone()
        
        if not self.nav_goal_active:
            self._navigate_to_point(zone[0], zone[1], 0.0)
            
        # Check if we've arrived
        if self._is_in_zone(zone):
            self._cancel_navigation()
            self.state = RobotState.DEPOSITING
            
    def _state_depositing(self):
        """Deposit the barrel in the collection zone"""
        self.cmd_vel_pub.publish(Twist())
        
        if not self.offload_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Offload service not available')
            self.state = RobotState.RECOVERY
            return
            
        request = OffloadItem.Request()
        future = self.offload_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info(
                f'Successfully deposited barrel! Total: {self.total_collected}'
            )
            self.holding_barrel = False
            self.holding_barrel_color = None
            self.state = RobotState.SEARCHING
        else:
            self.get_logger().warn('Deposit failed, moving and retrying...')
            # Move slightly and retry
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.5)
            self.cmd_vel_pub.publish(Twist())
            
    def _state_recovery(self):
        """Recovery state for handling errors"""
        self.cmd_vel_pub.publish(Twist())
        
        elapsed = time.time() - self.state_start_time
        if elapsed > 2.0:
            # After timeout, return to searching
            self.target_barrel = None
            self.state = RobotState.SEARCHING

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_available_barrels(self) -> List[Item]:
        """Get barrels that haven't been collected yet"""
        # Filter out already collected barrels (by tracking position)
        return [b for b in self.detected_barrels if b.diameter > 0.01]
        
    def _select_best_barrel(self, barrels: List[Item]) -> Optional[Item]:
        """Select the best barrel to pursue (largest = closest)"""
        if not barrels:
            return None
        # Sort by size (larger = closer)
        sorted_barrels = sorted(barrels, key=lambda b: b.diameter, reverse=True)
        return sorted_barrels[0]
        
    def _find_barrel_in_view(self) -> Optional[Item]:
        """Find a barrel in current camera view matching our target"""
        if not self.detected_barrels:
            return None
            
        # If we have a target color, prefer that
        if self.target_barrel:
            target_color = self.target_barrel.colour
            matching = [b for b in self.detected_barrels 
                       if b.colour == target_color and b.diameter > 0.01]
            if matching:
                return max(matching, key=lambda b: b.diameter)
                
        # Otherwise return the largest/closest barrel
        valid_barrels = [b for b in self.detected_barrels if b.diameter > 0.01]
        if valid_barrels:
            return max(valid_barrels, key=lambda b: b.diameter)
        return None
        
    def _estimate_distance_from_size(self, size: float) -> float:
        """
        Estimate distance to barrel from its apparent size in camera.
        This is an approximation - tune the constants for your setup.
        """
        if size <= 0:
            return float('inf')
        # Inverse relationship: larger size = smaller distance
        # These values need calibration for the actual camera
        return max(0.3, 2.0 / (size * 10 + 0.1))
        
    def _rotate_to_find_barrel(self):
        """Rotate in place to find a lost barrel"""
        cmd = Twist()
        cmd.angular.z = 0.3
        self.cmd_vel_pub.publish(cmd)
        
    def _is_in_zone(self, zone_center: Tuple[float, float], radius: float = 1.0) -> bool:
        """Check if robot is within a zone"""
        if self.current_pose is None:
            return False
        dx = self.current_pose[0] - zone_center[0]
        dy = self.current_pose[1] - zone_center[1]
        return math.sqrt(dx*dx + dy*dy) < radius
        
    def _get_closer_collection_zone(self) -> Tuple[float, float]:
        """Return the closer of the two collection zones"""
        if self.current_pose is None:
            return self.COLLECTION_ZONE_A
            
        dist_a = math.sqrt(
            (self.current_pose[0] - self.COLLECTION_ZONE_A[0])**2 +
            (self.current_pose[1] - self.COLLECTION_ZONE_A[1])**2
        )
        dist_b = math.sqrt(
            (self.current_pose[0] - self.COLLECTION_ZONE_B[0])**2 +
            (self.current_pose[1] - self.COLLECTION_ZONE_B[1])**2
        )
        return self.COLLECTION_ZONE_A if dist_a < dist_b else self.COLLECTION_ZONE_B
        
    def _navigate_to_next_waypoint(self):
        """Navigate to the next search waypoint"""
        if self.current_waypoint_index >= len(self.SEARCH_WAYPOINTS):
            self.current_waypoint_index = 0  # Loop back
            
        wp = self.SEARCH_WAYPOINTS[self.current_waypoint_index]
        self._navigate_to_point(wp[0], wp[1], 0.0)
        self.current_waypoint_index += 1
        
    def _navigate_to_point(self, x: float, y: float, yaw: float):
        """Send navigation goal to Nav2"""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 not available')
            return
            
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.nav_goal_active = True
        self._nav_future = self.nav_to_pose_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback_callback
        )
        self._nav_future.add_done_callback(self._nav_goal_response_callback)
        
        self.get_logger().info(f'Navigating to ({x:.1f}, {y:.1f})')
        
    def _nav_goal_response_callback(self, future):
        """Handle nav goal acceptance"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Navigation goal rejected')
            self.nav_goal_active = False
            return
            
        self._nav_result_future = goal_handle.get_result_async()
        self._nav_result_future.add_done_callback(self._nav_result_callback)
        
    def _nav_result_callback(self, future):
        """Handle nav goal completion"""
        self.nav_goal_active = False
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation goal reached')
        else:
            self.get_logger().warn(f'Navigation failed with status: {result.status}')
            
    def _nav_feedback_callback(self, feedback):
        """Handle nav feedback"""
        pass  # Can be used for progress monitoring
        
    def _cancel_navigation(self):
        """Cancel current navigation goal"""
        self.nav_goal_active = False
        # Nav2 goal cancellation would go here if needed


def main(args=None):
    rclpy.init(args=args)
    
    controller = RobotController()
    
    executor = MultiThreadedExecutor()
    executor.add_node(controller)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
