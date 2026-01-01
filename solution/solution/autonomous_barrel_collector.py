#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from enum import Enum
import math

class State(Enum):
    EXPLORING = 1
    APPROACHING_BARREL = 2
    PICKING_UP = 3
    AVOIDING_WALL = 4
    NAVIGATING = 5

class BarrelColor(Enum):
    # Define barrel color codes and zones
    # Barrel colors
    RED = ([0, 150, 52], [10, 255, 255])
    BLUE = ([111, 150, 51], [130, 255, 255])
    #Zone colors
    CYAN_ZONE = ([82, 97, 0], [179, 255, 255])
    GREEN_ZONE = ([60, 97, 0], [79, 255, 255])



class AutonomousBarrelCollector(Node):
    def __init__(self):
        super().__init__('autonomous_barrel_collector')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, '/robot1/scan', self.laser_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/robot1/camera/image_raw', self.camera_callback, 10)
        
        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # State machine
        self.state = State.EXPLORING
        self.previous_state = None
        
        # Barrel detection
        self.bridge = CvBridge()
        self.detected_barrel = None
        self.barrel_center_x = None
        
        # Laser scan data
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.scan_ranges = None
        
        # Parameters
        self.barrel_approach_distance = 0.3  # Stop 30cm from barrel
        self.wall_stop_distance = 0.5  # Stop 50cm from wall
        self.rotation_angle = 30.0  # Rotate 30 degrees when avoiding walls
        self.angular_speed = 0.5
        self.linear_speed = 0.2
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Autonomous Barrel Collector Node Started')
    
    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.scan_ranges = msg.ranges
        
        # Get distances in key directions
        # Front (0 degrees)
        front_indices = list(range(0, 10)) + list(range(len(msg.ranges)-10, len(msg.ranges)))
        self.front_distance = min([msg.ranges[i] for i in front_indices if not math.isinf(msg.ranges[i])] or [float('inf')])
        
        # Left (90 degrees)
        left_start = len(msg.ranges) // 4
        left_indices = range(left_start - 10, left_start + 10)
        self.left_distance = min([msg.ranges[i] for i in left_indices if i < len(msg.ranges) and not math.isinf(msg.ranges[i])] or [float('inf')])
        
        # Right (270 degrees)
        right_start = 3 * len(msg.ranges) // 4
        right_indices = range(right_start - 10, right_start + 10)
        self.right_distance = min([msg.ranges[i] for i in right_indices if i < len(msg.ranges) and not math.isinf(msg.ranges[i])] or [float('inf')])
    
    def camera_callback(self, msg):
        """Process camera images for barrel detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detect barrels by color
            barrel_detected = False
            largest_contour = None
            largest_area = 0
            
            for color in BarrelColor:
                lower, upper = color.value
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum area threshold
                        if area > largest_area:
                            largest_area = area
                            largest_contour = contour
                            self.detected_barrel = color.name
                            barrel_detected = True
            
            if barrel_detected and largest_contour is not None:
                # Get barrel position
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    self.barrel_center_x = int(M['m10'] / M['m00'])
                    image_center = cv_image.shape[1] / 2
                    
                    # Log detection
                    self.get_logger().info(f'Detected {self.detected_barrel} barrel at x={self.barrel_center_x}')
            else:
                self.detected_barrel = None
                self.barrel_center_x = None
                
        except Exception as e:
            self.get_logger().error(f'Camera processing error: {str(e)}')
    
    def control_loop(self):
        """Main control loop implementing state machine"""
        twist = Twist()
        
        # State machine logic
        if self.state == State.EXPLORING:
            self.explore(twist)
        elif self.state == State.APPROACHING_BARREL:
            self.approach_barrel(twist)
        elif self.state == State.AVOIDING_WALL:
            self.avoid_wall(twist)
        elif self.state == State.PICKING_UP:
            self.pickup_barrel(twist)
        
        self.cmd_vel_pub.publish(twist)
    
    def explore(self, twist):
        """Exploration behavior - move forward and look for barrels"""
        # Check for walls first
        if self.front_distance < self.wall_stop_distance:
            self.get_logger().info('Wall detected! Switching to avoidance')
            self.state = State.AVOIDING_WALL
            return
        
        # Check for barrels
        if self.detected_barrel is not None:
            self.get_logger().info(f'Barrel detected! Approaching {self.detected_barrel} barrel')
            self.state = State.APPROACHING_BARREL
            return
        
        # Continue exploring
        twist.linear.x = self.linear_speed
        twist.angular.z = 0.0
    
    def approach_barrel(self, twist):
        """Approach detected barrel"""
        if self.detected_barrel is None:
            self.get_logger().warn('Lost sight of barrel, returning to exploration')
            self.state = State.EXPLORING
            return
        
        # Check if we're close enough
        if self.front_distance < self.barrel_approach_distance:
            self.get_logger().info('Reached barrel! Initiating pickup')
            self.state = State.PICKING_UP
            return
        
        # Check for walls while approaching
        if self.front_distance < self.wall_stop_distance:
            self.get_logger().warn('Wall detected while approaching barrel')
            self.state = State.AVOIDING_WALL
            return
        
        # Align with barrel using camera
        if self.barrel_center_x is not None:
            image_center = 320  # Assuming 640px wide image
            error = (self.barrel_center_x - image_center) / image_center
            
            # Move towards barrel while aligning
            twist.linear.x = self.linear_speed * 0.5  # Slower approach
            twist.angular.z = -error * 0.5  # Proportional control
        else:
            # Lost visual, creep forward slowly
            twist.linear.x = 0.1
    
    def avoid_wall(self, twist):
        """Avoid walls by stopping and rotating"""
        # Stop first
        twist.linear.x = 0.0
        
        # Decide which way to turn based on available space
        if self.left_distance > self.right_distance:
            # More space on left, turn left
            self.get_logger().info(f'Turning LEFT (left:{self.left_distance:.2f}m > right:{self.right_distance:.2f}m)')
            twist.angular.z = self.angular_speed
        else:
            # More space on right, turn right
            self.get_logger().info(f'Turning RIGHT (right:{self.right_distance:.2f}m > left:{self.left_distance:.2f}m)')
            twist.angular.z = -self.angular_speed
        
        # Check if we've cleared the wall
        if self.front_distance > self.wall_stop_distance * 1.5:
            self.get_logger().info('Wall cleared! Returning to exploration')
            self.state = State.EXPLORING
    
    def pickup_barrel(self, twist):
        """Execute barrel pickup procedure"""
        # Stop movement
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        self.get_logger().info(f'Picking up {self.detected_barrel} barrel')
        
        # TODO: Add actual pickup service call here
        # For now, simulate pickup delay
        # After pickup, return to exploration
        self.state = State.EXPLORING
        self.detected_barrel = None
    
    def navigate_to_goal(self, x, y, theta=0.0):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # Convert theta to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2)
        
        self.get_logger().info(f'Sending goal: x={x}, y={y}, theta={theta}')
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousBarrelCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()