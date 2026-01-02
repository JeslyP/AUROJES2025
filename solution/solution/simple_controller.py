#!/usr/bin/env python3
"""
AURO 2025 - Simplified Robust Robot Controller

A more straightforward implementation focusing on reliability.
Uses a behavior-based approach with clear state transitions.

Key simplifications:
1. Uses Nav2 for all major navigation
2. Visual servoing only for final approach
3. Clear, testable state machine
4. Robust error handling
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

# Assessment interfaces
from assessment_interfaces.msg import ItemList, ZoneList, Item, Zone
from assessment_interfaces.srv import PickUpItem, OffloadItem, Decontaminate

import math
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time


class State(Enum):
    """Robot states"""
    INIT = auto()
    EXPLORE = auto()          # Navigate to search waypoints
    APPROACH = auto()         # Move toward detected barrel
    POSITION_FOR_PICKUP = auto()  # Get into pickup position
    PICKUP = auto()           # Execute pickup
    GO_TO_DECONTAM = auto()   # Navigate to decontamination
    DECONTAMINATE = auto()    # Execute decontamination
    GO_TO_DROPOFF = auto()    # Navigate to collection zone
    DROPOFF = auto()          # Execute dropoff
    STUCK = auto()            # Recovery from stuck


@dataclass
class BarrelInfo:
    """Information about a detected barrel"""
    x_offset: float      # Position in camera frame (-1 to 1)
    y_offset: float
    size: float          # Apparent size
    color: str           # 'red' or 'blue'
    

class SimpleRobotController(Node):
    """
    Simplified robot controller for barrel collection.
    """
    
    # World coordinates
    DECONTAM_ZONE = (7.5, 9.4)
    COLLECTION_ZONE_A = (13.5, 9.4)
    COLLECTION_ZONE_B = (19.5, 9.4)
    
    # Exploration waypoints - systematic coverage
    WAYPOINTS = [
        # Row 1 (y ≈ 3)
        (4.0, 3.0), (8.0, 3.0), (12.0, 3.0), (16.0, 3.0), (20.0, 3.0),
        # Row 2 (y ≈ 6)
        (20.0, 6.0), (16.0, 6.0), (12.0, 6.0), (8.0, 6.0), (4.0, 6.0),
        # Row 3 (y ≈ 9) - avoid zone areas
        (4.0, 9.0), (10.0, 9.0), (16.5, 9.0),
        # Row 4 (y ≈ 12)
        (4.0, 12.0), (8.0, 12.0), (12.0, 12.0), (16.0, 12.0), (20.0, 12.0),
        # Row 5 (y ≈ 15)
        (20.0, 15.0), (16.0, 15.0), (12.0, 15.0), (8.0, 15.0), (4.0, 15.0),
    ]
    
    # Control parameters
    MAX_LINEAR = 0.22
    MAX_ANGULAR = 1.0
    APPROACH_SPEED = 0.15
    REVERSE_SPEED = 0.08
    
    # Pickup geometry (from assessment spec)
    PICKUP_DISTANCE = 0.45
    PICKUP_ANGLE_DEG = 15.0
    
    def __init__(self):
        super().__init__('simple_robot_controller')
        
        # Callback groups for concurrent operation
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_cb_group = ReentrantCallbackGroup()
        self.srv_cb_group = MutuallyExclusiveCallbackGroup()
        
        # State
        self.state = State.INIT
        self.prev_state = None
        self.state_start_time = time.time()
        
        # Robot pose
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        
        # Detections
        self.barrels: List[Item] = []
        self.zones: List[Zone] = []
        self.target_barrel: Optional[BarrelInfo] = None
        
        # Status
        self.holding_barrel = False
        self.barrel_color: Optional[str] = None
        self.radiation = 0.0
        self.barrels_collected = 0
        
        # Navigation
        self.waypoint_idx = 0
        self.nav_active = False
        self.nav_succeeded = False
        
        # LIDAR
        self.front_clear = True
        
        # Setup ROS interfaces
        self._setup_ros()
        
        # Control loop at 10Hz
        self.timer = self.create_timer(
            0.1, self._control_loop, callback_group=self.timer_cb_group
        )
        
        self.get_logger().info('Simple Robot Controller started')
        
    def _setup_ros(self):
        """Setup publishers, subscribers, services, actions"""
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(
            Odometry, 'odom', self._odom_cb, 10, 
            callback_group=self.sub_cb_group
        )
        self.create_subscription(
            LaserScan, 'scan', self._scan_cb, 10,
            callback_group=self.sub_cb_group
        )
        self.create_subscription(
            ItemList, 'items', self._items_cb, 10,
            callback_group=self.sub_cb_group
        )
        self.create_subscription(
            ZoneList, 'zones', self._zones_cb, 10,
            callback_group=self.sub_cb_group
        )
        self.create_subscription(
            Float32, 'radiation_level', self._radiation_cb, 10,
            callback_group=self.sub_cb_group
        )
        
        # Service clients
        self.pickup_cli = self.create_client(
            PickUpItem, 'pick_up_item', callback_group=self.srv_cb_group
        )
        self.offload_cli = self.create_client(
            OffloadItem, 'offload_item', callback_group=self.srv_cb_group
        )
        self.decontam_cli = self.create_client(
            Decontaminate, 'decontaminate', callback_group=self.srv_cb_group
        )
        
        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.sub_cb_group
        )
        
    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def _odom_cb(self, msg: Odometry):
        """Update robot pose from odometry"""
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        
        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose_yaw = math.atan2(siny_cosp, cosy_cosp)
        
    def _scan_cb(self, msg: LaserScan):
        """Process LIDAR for obstacle detection"""
        ranges = np.array(msg.ranges)
        n = len(ranges)
        
        # Check front arc (~60 degrees)
        front_start = int(n * 0.42)
        front_end = int(n * 0.58)
        front = ranges[front_start:front_end]
        front = front[np.isfinite(front)]
        
        if len(front) > 0:
            self.front_clear = np.min(front) > 0.35
        else:
            self.front_clear = True
            
    def _items_cb(self, msg: ItemList):
        """Update detected barrels"""
        self.barrels = list(msg.data)
        
    def _zones_cb(self, msg: ZoneList):
        """Update detected zones"""
        self.zones = list(msg.data)
        
    def _radiation_cb(self, msg: Float32):
        """Update radiation level"""
        self.radiation = msg.data
        
    # =========================================================================
    # STATE MACHINE
    # =========================================================================
    
    def _control_loop(self):
        """Main control loop"""
        # Log state changes
        if self.state != self.prev_state:
            self.get_logger().info(f'State: {self.prev_state} -> {self.state}')
            self.prev_state = self.state
            self.state_start_time = time.time()
            
        # Execute state
        if self.state == State.INIT:
            self._do_init()
        elif self.state == State.EXPLORE:
            self._do_explore()
        elif self.state == State.APPROACH:
            self._do_approach()
        elif self.state == State.POSITION_FOR_PICKUP:
            self._do_position()
        elif self.state == State.PICKUP:
            self._do_pickup()
        elif self.state == State.GO_TO_DECONTAM:
            self._do_go_decontam()
        elif self.state == State.DECONTAMINATE:
            self._do_decontam()
        elif self.state == State.GO_TO_DROPOFF:
            self._do_go_dropoff()
        elif self.state == State.DROPOFF:
            self._do_dropoff()
        elif self.state == State.STUCK:
            self._do_stuck()
            
    def _do_init(self):
        """Initialize - wait for Nav2"""
        if self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info('Nav2 ready!')
            self.state = State.EXPLORE
        else:
            self.get_logger().info('Waiting for Nav2...', throttle_duration_sec=2.0)
            
    def _do_explore(self):
        """Explore environment looking for barrels"""
        # Check for barrels
        barrel = self._find_best_barrel()
        if barrel is not None:
            self.target_barrel = barrel
            self._cancel_nav()
            self.state = State.APPROACH
            return
            
        # Continue exploration
        if not self.nav_active:
            wp = self.WAYPOINTS[self.waypoint_idx]
            self._start_nav(wp[0], wp[1])
            self.waypoint_idx = (self.waypoint_idx + 1) % len(self.WAYPOINTS)
            
    def _do_approach(self):
        """Approach the target barrel using visual servoing"""
        if self.target_barrel is None:
            self.state = State.EXPLORE
            return
            
        # Find barrel in current view
        barrel = self._find_target_barrel()
        
        if barrel is None:
            # Lost barrel - search briefly
            elapsed = time.time() - self.state_start_time
            if elapsed > 4.0:
                self.target_barrel = None
                self.state = State.EXPLORE
                return
            self._rotate_search()
            return
            
        # Update target info
        self.target_barrel = barrel
        
        # Visual servoing
        cmd = Twist()
        x_err = barrel.x_offset
        dist_est = self._estimate_distance(barrel.size)
        
        # Angular: center the barrel
        if abs(x_err) > 0.03:
            cmd.angular.z = -1.0 * x_err
            cmd.angular.z = np.clip(cmd.angular.z, -0.5, 0.5)
            
        # Linear: approach
        if dist_est > 0.55:  # Stop a bit before pickup distance
            if self.front_clear:
                cmd.linear.x = min(self.APPROACH_SPEED, dist_est * 0.3)
            else:
                cmd.linear.x = 0.0
        else:
            # Close enough - position for pickup
            self._stop()
            self.state = State.POSITION_FOR_PICKUP
            return
            
        self.cmd_vel_pub.publish(cmd)
        
    def _do_position(self):
        """Position robot so barrel is behind it for pickup"""
        elapsed = time.time() - self.state_start_time
        
        # Phase 1: Turn 180 degrees (0-3 seconds)
        if elapsed < 3.0:
            cmd = Twist()
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Phase 2: Back up toward barrel (3-6 seconds)  
        if elapsed < 6.0:
            cmd = Twist()
            cmd.linear.x = -self.REVERSE_SPEED
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Ready for pickup
        self._stop()
        self.state = State.PICKUP
        
    def _do_pickup(self):
        """Execute barrel pickup"""
        self._stop()
        
        if not self.pickup_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Pickup service not available')
            self.state = State.STUCK
            return
            
        req = PickUpItem.Request()
        future = self.pickup_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Pickup SUCCESS!')
            self.holding_barrel = True
            self.barrel_color = self.target_barrel.color if self.target_barrel else 'unknown'
            self.target_barrel = None
            self.barrels_collected += 1
            
            # Red barrels need decontamination
            if 'red' in self.barrel_color.lower():
                self.state = State.GO_TO_DECONTAM
            else:
                self.state = State.GO_TO_DROPOFF
        else:
            self.get_logger().warn('Pickup failed, retrying position...')
            self.state = State.POSITION_FOR_PICKUP
            self.state_start_time = time.time()  # Reset for another attempt
            
    def _do_go_decontam(self):
        """Navigate to decontamination zone"""
        if not self.nav_active:
            self._start_nav(self.DECONTAM_ZONE[0], self.DECONTAM_ZONE[1])
            
        # Check if arrived
        if self._in_zone(self.DECONTAM_ZONE):
            self._cancel_nav()
            self.state = State.DECONTAMINATE
            
    def _do_decontam(self):
        """Execute decontamination"""
        self._stop()
        
        if not self.decontam_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Decontaminate service not available')
            return
            
        req = Decontaminate.Request()
        future = self.decontam_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Decontamination SUCCESS!')
            self.state = State.GO_TO_DROPOFF
        else:
            self.get_logger().warn('Decontamination failed, retrying...')
            
    def _do_go_dropoff(self):
        """Navigate to collection zone"""
        if not self.holding_barrel:
            self.state = State.EXPLORE
            return
            
        zone = self._get_closer_zone()
        
        if not self.nav_active:
            self._start_nav(zone[0], zone[1])
            
        if self._in_zone(zone):
            self._cancel_nav()
            self.state = State.DROPOFF
            
    def _do_dropoff(self):
        """Execute barrel dropoff"""
        self._stop()
        
        if not self.offload_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Offload service not available')
            return
            
        req = OffloadItem.Request()
        future = self.offload_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'Dropoff SUCCESS! Total collected: {self.barrels_collected}')
            self.holding_barrel = False
            self.barrel_color = None
            self.state = State.EXPLORE
        else:
            self.get_logger().warn('Dropoff failed, repositioning...')
            # Move a bit and retry
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.5)
            self._stop()
            
    def _do_stuck(self):
        """Recovery from stuck state"""
        elapsed = time.time() - self.state_start_time
        
        if elapsed < 2.0:
            # Back up
            cmd = Twist()
            cmd.linear.x = -0.1
            self.cmd_vel_pub.publish(cmd)
        elif elapsed < 4.0:
            # Turn
            cmd = Twist()
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
        else:
            self._stop()
            self.target_barrel = None
            self.state = State.EXPLORE
            
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _find_best_barrel(self) -> Optional[BarrelInfo]:
        """Find the best barrel to pursue from detections"""
        if not self.barrels:
            return None
            
        # Filter and sort by size (larger = closer)
        valid = [b for b in self.barrels if b.diameter > 0.01]
        if not valid:
            return None
            
        best = max(valid, key=lambda b: b.diameter)
        return BarrelInfo(
            x_offset=best.x,
            y_offset=best.y,
            size=best.diameter,
            color=best.colour
        )
        
    def _find_target_barrel(self) -> Optional[BarrelInfo]:
        """Find the target barrel in current detections"""
        if not self.barrels or self.target_barrel is None:
            return None
            
        # Find barrel matching target color
        target_color = self.target_barrel.color
        matching = [b for b in self.barrels 
                   if b.colour.lower() == target_color.lower() and b.diameter > 0.01]
        
        if matching:
            best = max(matching, key=lambda b: b.diameter)
            return BarrelInfo(
                x_offset=best.x,
                y_offset=best.y,
                size=best.diameter,
                color=best.colour
            )
        return None
        
    def _estimate_distance(self, size: float) -> float:
        """Estimate distance from barrel apparent size"""
        if size <= 0:
            return float('inf')
        # Tune this based on your camera
        return max(0.3, 0.2 / (size + 0.01))
        
    def _rotate_search(self):
        """Rotate in place to search for lost barrel"""
        cmd = Twist()
        cmd.angular.z = 0.3
        self.cmd_vel_pub.publish(cmd)
        
    def _stop(self):
        """Stop the robot"""
        self.cmd_vel_pub.publish(Twist())
        
    def _in_zone(self, zone: Tuple[float, float], radius: float = 1.0) -> bool:
        """Check if robot is within a zone"""
        dx = self.pose_x - zone[0]
        dy = self.pose_y - zone[1]
        return math.sqrt(dx*dx + dy*dy) < radius
        
    def _get_closer_zone(self) -> Tuple[float, float]:
        """Get the closer collection zone"""
        dist_a = math.sqrt(
            (self.pose_x - self.COLLECTION_ZONE_A[0])**2 +
            (self.pose_y - self.COLLECTION_ZONE_A[1])**2
        )
        dist_b = math.sqrt(
            (self.pose_x - self.COLLECTION_ZONE_B[0])**2 +
            (self.pose_y - self.COLLECTION_ZONE_B[1])**2
        )
        return self.COLLECTION_ZONE_A if dist_a < dist_b else self.COLLECTION_ZONE_B
        
    def _start_nav(self, x: float, y: float):
        """Start navigation to a point"""
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 not ready')
            return
            
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        
        self.nav_active = True
        self._nav_future = self.nav_client.send_goal_async(goal)
        self._nav_future.add_done_callback(self._nav_response_cb)
        
        self.get_logger().info(f'Navigating to ({x:.1f}, {y:.1f})')
        
    def _nav_response_cb(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.nav_active = False
            return
        self._nav_result_future = goal_handle.get_result_async()
        self._nav_result_future.add_done_callback(self._nav_result_cb)
        
    def _nav_result_cb(self, future):
        """Handle navigation result"""
        self.nav_active = False
        
    def _cancel_nav(self):
        """Cancel navigation"""
        self.nav_active = False


def main(args=None):
    rclpy.init(args=args)
    
    controller = SimpleRobotController()
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
