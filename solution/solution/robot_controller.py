import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException



from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from auro_interfaces.srv import ItemRequest
from assessment_interfaces.msg import Barrel, BarrelList, Zone, ZoneList, BarrelLog

import math
import numpy as np
from enum import Enum, auto
from typing import Optional, List, Tuple
import time


class State(Enum):
    """Robot state machine states."""
    INIT = auto()
    SEARCHING = auto()
    APPROACHING = auto()
    POSITIONING = auto()
    PICKING_UP = auto()
    GOING_TO_DECONTAMINATION = auto()
    DECONTAMINATING = auto()
    GOING_TO_COLLECTION = auto()
    DEPOSITING = auto()
    RECOVERY = auto()


class RobotController(Node):
    """
    Autonomous robot controller for the AURO barrel collection task.
    
    This controller implements a state machine that:
    1. Searches for barrels using visual detection
    2. Approaches and picks up barrels
    3. Decontaminates if carrying a red barrel
    4. Deposits barrels in collection zones
    """

    # Zone coordinates (centres of 3m x 3m zones)
    # From assessment.world: storage2 at (13.5, 9.4), decontamination at (7.5, 9.4)
    COLLECTION_ZONE_A = (13.5, 9.4)
    COLLECTION_ZONE_B = (13.5, 9.4)  # Only one collection zone in this world
    DECONTAMINATION_ZONE = (7.5, 9.4)
    
    # Search waypoints - barrels spawn in NEGATIVE X area (around -1 to -15, y: 3-11)
    SEARCH_WAYPOINTS = [
        # Near spawn area first
        (-2.0, 3.0), (-2.0, 5.0), (-2.0, 7.0),
        # Main barrel area
        (-4.0, 4.0), (-6.0, 4.0), (-8.0, 4.0), (-10.0, 4.0), (-12.0, 4.0), (-14.0, 4.0),
        (-14.0, 6.0), (-12.0, 6.0), (-10.0, 6.0), (-8.0, 6.0), (-6.0, 6.0), (-4.0, 6.0),
        (-4.0, 8.0), (-6.0, 8.0), (-8.0, 8.0), (-10.0, 8.0), (-12.0, 8.0), (-14.0, 8.0),
        (-14.0, 10.0), (-12.0, 10.0), (-10.0, 10.0), (-8.0, 10.0), (-6.0, 10.0), (-4.0, 10.0),
    ]

    def __init__(self):
        super().__init__('robot_controller')
        
        # Callback groups
        self.callback_group = ReentrantCallbackGroup()
        self.timer_group = MutuallyExclusiveCallbackGroup()
        
        # State machine
        self.state = State.INIT
        self.prev_state = None
        self.state_start_time = time.time()
        
        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Barrel detection
        self.detected_barrels: List[Barrel] = []
        self.target_barrel: Optional[Barrel] = None
        
        # Zone detection
        self.detected_zones: List[Zone] = []
        
        # Robot status
        self.holding_barrel = False
        self.holding_red = False
        self.radiation_level = 0.0
        self.barrels_collected = 0
        
        # Navigation
        self.waypoint_index = 0
        self.nav_active = False
        
        # LIDAR
        self.front_clear = True
        
        # Setup ROS interfaces
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        self._setup_actions()
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.timer_group
        )
        
        self.get_logger().info('Robot controller initialised')

    def _setup_publishers(self):
        """Setup ROS publishers."""
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _setup_subscribers(self):
        """Setup ROS subscribers."""
        self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            BarrelList, 'barrels', self.barrels_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            ZoneList, 'zones', self.zones_callback, 10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Float32, 'radiation_level', self.radiation_callback, 10,
            callback_group=self.callback_group
        )

    def _setup_services(self):
        """Setup ROS service clients."""
        self.pick_up_client = self.create_client(
            ItemRequest, '/pick_up_item', callback_group=self.callback_group
        )
        self.offload_client = self.create_client(
            ItemRequest, '/offload_item', callback_group=self.callback_group
        )
        self.decontaminate_client = self.create_client(
            ItemRequest, '/decontaminate', callback_group=self.callback_group
        )

    def _setup_actions(self):
        """Setup Nav2 action client."""
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.callback_group
        )

    # =========================================================================
    # SUBSCRIBER CALLBACKS
    # =========================================================================
    
    def odom_callback(self, msg: Odometry):
        """Process odometry data."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg: LaserScan):
        """Process LIDAR scan for obstacle detection."""
        ranges = np.array(msg.ranges)
        n = len(ranges)
        
        # Check front arc (~60 degrees)
        front_start = int(n * 0.42)
        front_end = int(n * 0.58)
        front = ranges[front_start:front_end]
        front = front[np.isfinite(front)]
        
        self.front_clear = len(front) == 0 or np.min(front) > 0.35

    def barrels_callback(self, msg: BarrelList):
        """Process detected barrels from visual sensor."""
        self.detected_barrels = list(msg.data)

    def zones_callback(self, msg: ZoneList):
        """Process detected zones from visual sensor."""
        self.detected_zones = list(msg.data)

    def radiation_callback(self, msg: Float32):
        """Process radiation level."""
        self.radiation_level = msg.data

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    
    def control_loop(self):
        """Main state machine control loop."""
        # Log state changes
        if self.state != self.prev_state:
            self.get_logger().info(f'State: {self.prev_state} -> {self.state}')
            self.prev_state = self.state
            self.state_start_time = time.time()
        
        # Execute current state
        state_handlers = {
            State.INIT: self.state_init,
            State.SEARCHING: self.state_searching,
            State.APPROACHING: self.state_approaching,
            State.POSITIONING: self.state_positioning,
            State.PICKING_UP: self.state_picking_up,
            State.GOING_TO_DECONTAMINATION: self.state_going_to_decontamination,
            State.DECONTAMINATING: self.state_decontaminating,
            State.GOING_TO_COLLECTION: self.state_going_to_collection,
            State.DEPOSITING: self.state_depositing,
            State.RECOVERY: self.state_recovery,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            handler()

    # =========================================================================
    # STATE HANDLERS
    # =========================================================================
    
    def state_init(self):
        """Wait for Nav2 to be ready."""
        if self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().info('Nav2 ready, starting search')
            self.state = State.SEARCHING
        else:
            self.get_logger().info('Waiting for Nav2...', throttle_duration_sec=2.0)

    def state_searching(self):
        """Search for barrels by navigating through waypoints."""
        # Check if we see any barrels
        barrel = self.find_best_barrel()
        if barrel is not None:
            self.target_barrel = barrel
            self.cancel_navigation()
            self.state = State.APPROACHING
            return
        
        # Continue search pattern
        if not self.nav_active:
            self.navigate_to_next_waypoint()

    def state_approaching(self):
        """Approach the target barrel using visual servoing."""
        if self.target_barrel is None:
            self.state = State.SEARCHING
            return
        
        # Find barrel in current view
        barrel = self.find_target_barrel()
        
        if barrel is None:
            # Lost barrel - search briefly
            elapsed = time.time() - self.state_start_time
            if elapsed > 4.0:
                self.target_barrel = None
                self.state = State.SEARCHING
                return
            self.rotate_search()
            return
        
        # Visual servoing
        # barrel.x is pixel coordinate (0-640), centre at 320
        x_offset = (barrel.x - 320) / 320.0  # Normalised to [-1, 1]
        distance_estimate = self.estimate_distance(barrel.size)
        
        cmd = Twist()
        
        # Angular: centre the barrel
        if abs(x_offset) > 0.05:
            cmd.angular.z = -0.8 * x_offset
            cmd.angular.z = np.clip(cmd.angular.z, -0.5, 0.5)
        
        # Linear: approach
        if distance_estimate > 0.6:
            if self.front_clear:
                cmd.linear.x = min(0.15, distance_estimate * 0.2)
        else:
            # Close enough - position for pickup
            self.stop()
            self.state = State.POSITIONING
            return
        
        self.cmd_vel_pub.publish(cmd)

    def state_positioning(self):
        """Position robot so barrel is behind it for pickup."""
        elapsed = time.time() - self.state_start_time
        
        # Phase 1: Drive past the barrel (0-2 seconds)
        if elapsed < 2.0:
            cmd = Twist()
            if self.front_clear:
                cmd.linear.x = 0.12
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Phase 2: Turn around (2-5 seconds)
        if elapsed < 5.0:
            cmd = Twist()
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Phase 3: Back up toward barrel (5-7 seconds)
        if elapsed < 7.0:
            cmd = Twist()
            cmd.linear.x = -0.08
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Ready for pickup
        self.stop()
        self.state = State.PICKING_UP

    def state_picking_up(self):
        """Attempt to pick up the barrel."""
        self.stop()
        
        if not self.pick_up_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Pick up service not available')
            self.state = State.RECOVERY
            return
        
        # Get robot namespace from node name
        robot_id = self.get_namespace().strip('/')
        if not robot_id:
            robot_id = 'robot1'
        
        request = ItemRequest.Request()
        request.robot_id = robot_id
        
        future = self.pick_up_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Pickup SUCCESS!')
            self.holding_barrel = True
            
            # Check if it's a red barrel
            if self.target_barrel and self.target_barrel.colour == Barrel.RED:
                self.holding_red = True
            else:
                self.holding_red = False
            
            self.barrels_collected += 1
            self.target_barrel = None
            
            # Red barrels need decontamination
            if self.holding_red:
                self.state = State.GOING_TO_DECONTAMINATION
            else:
                self.state = State.GOING_TO_COLLECTION
        else:
            self.get_logger().warn('Pickup failed, repositioning...')
            self.state = State.POSITIONING
            self.state_start_time = time.time()

    def state_going_to_decontamination(self):
        """Navigate to decontamination zone."""
        if not self.nav_active:
            self.navigate_to_point(
                self.DECONTAMINATION_ZONE[0],
                self.DECONTAMINATION_ZONE[1]
            )
        
        if self.in_zone(self.DECONTAMINATION_ZONE):
            self.cancel_navigation()
            self.state = State.DECONTAMINATING

    def state_decontaminating(self):
        """Decontaminate the robot."""
        self.stop()
        
        if not self.decontaminate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Decontaminate service not available')
            return
        
        robot_id = self.get_namespace().strip('/') or 'robot1'
        
        request = ItemRequest.Request()
        request.robot_id = robot_id
        
        future = self.decontaminate_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Decontamination SUCCESS!')
            self.state = State.GOING_TO_COLLECTION
        else:
            self.get_logger().warn('Decontamination failed, retrying...')

    def state_going_to_collection(self):
        """Navigate to collection zone."""
        if not self.holding_barrel:
            self.state = State.SEARCHING
            return
        
        zone = self.get_nearest_collection_zone()
        
        if not self.nav_active:
            self.navigate_to_point(zone[0], zone[1])
        
        if self.in_zone(zone):
            self.cancel_navigation()
            self.state = State.DEPOSITING

    def state_depositing(self):
        """Deposit the barrel in collection zone."""
        self.stop()
        
        if not self.offload_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Offload service not available')
            return
        
        robot_id = self.get_namespace().strip('/') or 'robot1'
        
        request = ItemRequest.Request()
        request.robot_id = robot_id
        
        future = self.offload_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'Deposit SUCCESS! Total: {self.barrels_collected}')
            self.holding_barrel = False
            self.holding_red = False
            self.state = State.SEARCHING
        else:
            self.get_logger().warn('Deposit failed, repositioning...')
            # Move slightly and retry
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.5)
            self.stop()

    def state_recovery(self):
        """Recovery from errors."""
        elapsed = time.time() - self.state_start_time
        
        if elapsed < 2.0:
            cmd = Twist()
            cmd.linear.x = -0.1
            self.cmd_vel_pub.publish(cmd)
        elif elapsed < 4.0:
            cmd = Twist()
            cmd.angular.z = 0.5
            self.cmd_vel_pub.publish(cmd)
        else:
            self.stop()
            self.target_barrel = None
            self.state = State.SEARCHING

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def find_best_barrel(self) -> Optional[Barrel]:
        """Find the best barrel to pursue (largest = closest)."""
        if not self.detected_barrels:
            return None
        
        valid = [b for b in self.detected_barrels if b.size > 100]
        if not valid:
            return None
        
        return max(valid, key=lambda b: b.size)

    def find_target_barrel(self) -> Optional[Barrel]:
        """Find the target barrel in current detections."""
        if not self.detected_barrels or self.target_barrel is None:
            return None
        
        # Match by colour
        target_colour = self.target_barrel.colour
        matching = [b for b in self.detected_barrels 
                   if b.colour == target_colour and b.size > 100]
        
        if matching:
            return max(matching, key=lambda b: b.size)
        return None

    def estimate_distance(self, size: float) -> float:
        """Estimate distance from barrel size (area in pixels)."""
        if size <= 0:
            return 10.0
        # Larger area = closer
        return max(0.3, 50.0 / math.sqrt(size + 1))

    def rotate_search(self):
        """Rotate to search for lost barrel."""
        cmd = Twist()
        cmd.angular.z = 0.3
        self.cmd_vel_pub.publish(cmd)

    def stop(self):
        """Stop the robot."""
        self.cmd_vel_pub.publish(Twist())

    def in_zone(self, zone_centre: Tuple[float, float], radius: float = 1.0) -> bool:
        """Check if robot is within a zone."""
        dx = self.robot_x - zone_centre[0]
        dy = self.robot_y - zone_centre[1]
        return math.sqrt(dx*dx + dy*dy) < radius

    def get_nearest_collection_zone(self) -> Tuple[float, float]:
        """Get the nearest collection zone."""
        dist_a = math.sqrt(
            (self.robot_x - self.COLLECTION_ZONE_A[0])**2 +
            (self.robot_y - self.COLLECTION_ZONE_A[1])**2
        )
        dist_b = math.sqrt(
            (self.robot_x - self.COLLECTION_ZONE_B[0])**2 +
            (self.robot_y - self.COLLECTION_ZONE_B[1])**2
        )
        return self.COLLECTION_ZONE_A if dist_a < dist_b else self.COLLECTION_ZONE_B

    def navigate_to_next_waypoint(self):
        """Navigate to the next search waypoint."""
        if self.waypoint_index >= len(self.SEARCH_WAYPOINTS):
            self.waypoint_index = 0
        
        wp = self.SEARCH_WAYPOINTS[self.waypoint_index]
        self.navigate_to_point(wp[0], wp[1])
        self.waypoint_index += 1

    def navigate_to_point(self, x: float, y: float):
        """Send navigation goal to Nav2."""
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Nav2 not available')
            return
        
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        
        self.nav_active = True
        self._nav_future = self.nav_client.send_goal_async(goal)
        self._nav_future.add_done_callback(self._nav_goal_response)
        
        self.get_logger().info(f'Navigating to ({x:.1f}, {y:.1f})')

    def _nav_goal_response(self, future):
        """Handle navigation goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.nav_active = False
            return
        
        self._nav_result_future = goal_handle.get_result_async()
        self._nav_result_future.add_done_callback(self._nav_result)

    def _nav_result(self, future):
        """Handle navigation result."""
        self.nav_active = False

    def cancel_navigation(self):
        """Cancel current navigation."""
        self.nav_active = False


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