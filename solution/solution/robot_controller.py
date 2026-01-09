import sys
import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.action import ActionClient

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

        # Known zone positions (from barrel_manager.py)
        # Green collection zones
        self.collection_zone_1 = {'x': 13.5, 'y': 9.4}
        self.collection_zone_2 = {'x': 19.5, 'y': 9.4}
        # Cyan decontamination zone
        self.decontamination_zone = {'x': 7.5, 'y': 9.4}

        # ============================================================
        # CALLBACK GROUPS (for async service calls)
        # ============================================================
        self.callback_group = MutuallyExclusiveCallbackGroup()

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

        # Initial pose publisher for AMCL (namespaced)
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            'initialpose',
            10
        )

        # Flag to track if initial pose has been set
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

        # Navigation state
        self.nav_goal_handle = None
        self.nav_in_progress = False

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

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)

    def publish_initial_pose(self):
        """Publish initial pose to AMCL for localization."""
        if self.initial_pose_set:
            return
        
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.pose.position.x = self.initial_x
        msg.pose.pose.position.y = self.initial_y
        msg.pose.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(self.initial_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.initial_yaw / 2.0)
        
        # Set covariance (small values = confident in position)
        msg.pose.covariance[0] = 0.25  # x variance
        msg.pose.covariance[7] = 0.25  # y variance
        msg.pose.covariance[35] = 0.0685  # yaw variance
        
        self.initial_pose_publisher.publish(msg)
        self.initial_pose_set = True
        self.get_logger().info(f"Published initial pose: x={self.initial_x}, y={self.initial_y}, yaw={self.initial_yaw}")

    def distance_to(self, x, y):
        """Calculate distance from robot to a point."""
        return math.sqrt((x - self.robot_x)**2 + (y - self.robot_y)**2)

    # ================================================================
    # STATE METHODS (TODO: Implement these)
    # ================================================================

    def searching(self):
        """SEARCHING state: Look for barrels by rotating."""
        
        # Check if we see any barrels
        if len(self.barrels) > 0:
            # Found barrel(s)! Pick the largest one (closest/most visible)
            largest_barrel = max(self.barrels, key=lambda b: b.size)
            
            self.target_barrel = largest_barrel
            self.get_logger().info(
                f"Found barrel! Colour: {largest_barrel.colour}, "
                f"Size: {largest_barrel.size:.2f}, "
                f"Offset: x={largest_barrel.x:.2f}, y={largest_barrel.y:.2f}"
            )
            
            # Stop rotating
            self.stop_robot()
            
            # Switch to APPROACHING state
            self.state = State.APPROACHING
            return
        
        # No barrel found - rotate to search
        twist = Twist()
        twist.angular.z = 0.5  # Rotate at 0.5 rad/s (about 30 deg/s)
        self.cmd_vel_publisher.publish(twist)

    def approaching(self):
        """APPROACHING state: Navigate to barrel."""
        pass

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
            self.publish_initial_pose()
            return  # Wait for localization to initialize
        
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