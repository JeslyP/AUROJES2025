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
    SET_INITIAL_POSE = 0
    SEARCHING = 1
    APPROACHING = 2
    POSITIONING = 3
    PICKING_UP = 4
    DELIVERING = 5
    OFFLOADING = 6
    DECONTAMINATING = 7


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
        self.state = State.SET_INITIAL_POSE
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

        # Known zone positions (from barrel_manager.py)
        # Green collection zones
        self.collection_zone_1 = {'x': 13.5, 'y': 9.4}
        self.collection_zone_2 = {'x': 19.5, 'y': 9.4}
        # Cyan decontamination zone
        self.decontamination_zone = {'x': 7.5, 'y': 9.4}

        # ============================================================
        # NAV2 WAYPOINTS - Locations to search for barrels
        # TODO: Update these coordinates based on your map!
        # ============================================================
        self.waypoints = [
            {'x': 3.0, 'y': 0.0},      # Waypoint 1
            {'x': 6.0, 'y': 0.0},      # Waypoint 2
            {'x': 9.0, 'y': 3.0},      # Waypoint 3
            {'x': 12.0, 'y': 6.0},     # Waypoint 4
            {'x': 15.0, 'y': 9.0},     # Waypoint 5
            {'x': 18.0, 'y': 6.0},     # Waypoint 6
            {'x': 12.0, 'y': 3.0},     # Waypoint 7
            {'x': 6.0, 'y': 6.0},      # Waypoint 8
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
        
        # Velocity commands for direct robot control
        cmd_vel_topic = f'/{self.robot_name}/cmd_vel'
        self.get_logger().info(f"Publishing velocity commands to: {cmd_vel_topic}")
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        # Initial pose publisher for AMCL (namespaced)
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            'initialpose',
            10
        )

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

        # ============================================================
        # INITIAL POSE TRACKING
        # ============================================================
        self.initial_pose_count = 0
        self.initial_pose_max = 5  # Publish initial pose 5 times

        # ============================================================
        # CONTROL LOOP TIMER
        # ============================================================
        self.timer_period = 0.1  # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info(f"Initial pose - x: {self.initial_x}, y: {self.initial_y}, yaw: {self.initial_yaw}")
        self.get_logger().info("Robot controller initialized. Starting in SET_INITIAL_POSE state.")

    # ================================================================
    # SUBSCRIBER CALLBACKS
    # ================================================================

    def barrel_callback(self, msg):
        """Callback for barrel detection from camera."""
        self.barrels = msg.data
        if len(self.barrels) > 0:
            self.get_logger().info(
                f"Barrel callback: detected {len(self.barrels)} barrel(s)",
                throttle_duration_sec=2.0
            )

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
    # NAV2 METHODS
    # ================================================================

    def set_initial_pose(self):
        """Publish initial pose for AMCL."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Map coordinates - robot spawns at Gazebo (0, -2) which maps to (0, 0)
        map_x = 0.0
        map_y = 0.0
        map_yaw = self.initial_yaw - (math.pi / 2.0)  # Adjust for orientation offset
        
        msg.pose.pose.position.x = map_x
        msg.pose.pose.position.y = map_y
        msg.pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(map_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(map_yaw / 2.0)
        
        # Set covariance
        msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.068]
        
        self.initial_pose_publisher.publish(msg)
        self.get_logger().info(
            f"Published initial pose ({self.initial_pose_count + 1}/{self.initial_pose_max}): "
            f"x={map_x}, y={map_y}, yaw={map_yaw:.2f}"
        )

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
    # HELPER METHODS
    # ================================================================

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)

    def distance_to(self, x, y):
        """Calculate distance from robot to a point."""
        return math.sqrt((x - self.robot_x)**2 + (y - self.robot_y)**2)

    # ================================================================
    # STATE METHODS
    # ================================================================

    def searching(self):
        """SEARCHING state: Navigate to waypoints using Nav2, look for barrels."""
        
        # Check if we see any barrels while navigating
        if len(self.barrels) > 0:
            # Found barrel(s)! Cancel navigation and go to it
            largest_barrel = max(self.barrels, key=lambda b: b.size)
            
            self.get_logger().info(
                f"Found barrel during search! Colour: {largest_barrel.colour}, "
                f"Size: {largest_barrel.size:.1f}"
            )
            
            # Cancel current navigation
            self.cancel_navigation()
            
            # Set target and switch state
            self.target_barrel = largest_barrel
            self.state = State.APPROACHING
            return
        
        # Check if we're currently navigating
        if self.goal_handle is None:
            # Start navigating to current waypoint
            waypoint = self.waypoints[self.current_waypoint_index]
            self.get_logger().info(
                f"SEARCHING: Navigating to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                f"at ({waypoint['x']}, {waypoint['y']})"
            )
            self.navigate_to_pose(waypoint['x'], waypoint['y'])
            return
        
        # Check if navigation completed
        if self.navigation_complete:
            if self.navigation_result == 'succeeded':
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_index + 1}")
            else:
                self.get_logger().warn(f"Navigation to waypoint failed: {self.navigation_result}")
            
            # Move to next waypoint
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = 0  # Loop back
                self.get_logger().info("Completed all waypoints, starting over")
            
            # Reset for next navigation
            self.goal_handle = None
            self.navigation_complete = False

    def approaching(self):
        """APPROACHING state: Move towards barrel using cmd_vel."""
        
        # Check if we still see barrels
        if len(self.barrels) == 0:
            self.get_logger().warn("Lost sight of barrel, returning to SEARCHING")
            self.target_barrel = None
            self.state = State.SEARCHING
            return
        
        # Update target to largest visible barrel
        self.target_barrel = max(self.barrels, key=lambda b: b.size)
        
        barrel_y = self.target_barrel.y  # Left/right offset
        barrel_size = self.target_barrel.size
        
        self.get_logger().info(
            f"APPROACHING: y={barrel_y:.1f}, size={barrel_size:.1f}",
            throttle_duration_sec=1.0
        )
        
        # If barrel is close enough, switch to POSITIONING
        if barrel_size > 5000:  # Adjust threshold as needed
            self.get_logger().info("Close enough to barrel, switching to POSITIONING")
            self.stop_robot()
            self.state = State.POSITIONING
            return
        
        # Move towards the barrel
        twist = Twist()
        twist.linear.x = 0.2  # Move forward
        
        # Steer towards barrel (barrel_y: positive = left)
        angular_gain = 0.002
        twist.angular.z = angular_gain * barrel_y
        twist.angular.z = max(-0.5, min(0.5, twist.angular.z))  # Clamp
        
        self.cmd_vel_publisher.publish(twist)

    def positioning(self):
        """POSITIONING state: Maneuver barrel behind robot."""
        # TODO: Implement positioning logic
        self.get_logger().info("POSITIONING: Not implemented yet", throttle_duration_sec=2.0)
        self.stop_robot()

    def picking_up(self):
        """PICKING_UP state: Call pick_up service."""
        # TODO: Implement pick up logic
        self.get_logger().info("PICKING_UP: Not implemented yet", throttle_duration_sec=2.0)
        self.stop_robot()

    def delivering(self):
        """DELIVERING state: Navigate to collection zone."""
        # TODO: Implement delivering logic
        self.get_logger().info("DELIVERING: Not implemented yet", throttle_duration_sec=2.0)
        self.stop_robot()

    def offloading(self):
        """OFFLOADING state: Call offload service."""
        # TODO: Implement offload logic
        self.get_logger().info("OFFLOADING: Not implemented yet", throttle_duration_sec=2.0)
        self.stop_robot()

    def decontaminating(self):
        """DECONTAMINATING state: Go to cyan zone and decontaminate."""
        # TODO: Implement decontamination logic
        self.get_logger().info("DECONTAMINATING: Not implemented yet", throttle_duration_sec=2.0)
        self.stop_robot()

    # ================================================================
    # MAIN CONTROL LOOP
    # ================================================================

    def control_loop(self):
        """Main control loop - runs at 10 Hz."""
        
        # Log state changes
        if self.state != self.previous_state:
            self.get_logger().info(f"State: {self.previous_state} -> {self.state}")
            self.previous_state = self.state

        # State machine
        if self.state == State.SET_INITIAL_POSE:
            # Publish initial pose multiple times
            self.set_initial_pose()
            self.initial_pose_count += 1
            
            if self.initial_pose_count >= self.initial_pose_max:
                self.get_logger().info("Initial pose set, switching to SEARCHING")
                self.state = State.SEARCHING
                
        elif self.state == State.SEARCHING:
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