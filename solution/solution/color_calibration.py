#!/usr/bin/env python3
"""
Color Calibration Tool for Barrel Detection
This tool helps you find the correct HSV ranges for barrel colors in your environment.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorCalibration(Node):
    def __init__(self):
        super().__init__('color_calibration')
        
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image, '/robot1/camera/image_raw', self.camera_callback, 10)
        
        # Create trackbars window
        cv2.namedWindow('HSV Calibration')
        cv2.createTrackbar('H Low', 'HSV Calibration', 0, 179, self.nothing)
        cv2.createTrackbar('H High', 'HSV Calibration', 179, 179, self.nothing)
        cv2.createTrackbar('S Low', 'HSV Calibration', 0, 255, self.nothing)
        cv2.createTrackbar('S High', 'HSV Calibration', 255, 255, self.nothing)
        cv2.createTrackbar('V Low', 'HSV Calibration', 0, 255, self.nothing)
        cv2.createTrackbar('V High', 'HSV Calibration', 255, 255, self.nothing)
        
        self.get_logger().info('Color Calibration Tool Started')
        self.get_logger().info('Adjust trackbars to isolate barrel color')
        self.get_logger().info('Press "s" to save current values')
        self.get_logger().info('Press "q" to quit')
    
    def nothing(self, x):
        pass
    
    def camera_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Get trackbar values
            h_low = cv2.getTrackbarPos('H Low', 'HSV Calibration')
            h_high = cv2.getTrackbarPos('H High', 'HSV Calibration')
            s_low = cv2.getTrackbarPos('S Low', 'HSV Calibration')
            s_high = cv2.getTrackbarPos('S High', 'HSV Calibration')
            v_low = cv2.getTrackbarPos('V Low', 'HSV Calibration')
            v_high = cv2.getTrackbarPos('V High', 'HSV Calibration')
            
            # Create mask
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])
            mask = cv2.inRange(hsv_image, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on original image
            result = cv_image.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            # Add info text
            info_text = f'HSV: [{h_low}, {s_low}, {v_low}] to [{h_high}, {s_high}, {v_high}]'
            cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Count and display contours
            contour_text = f'Contours: {len(contours)}'
            cv2.putText(result, contour_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Display images
            cv2.imshow('Original', cv_image)
            cv2.imshow('HSV Calibration', hsv_image)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', result)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('Quitting...')
                rclpy.shutdown()
            elif key == ord('s'):
                self.save_values(h_low, h_high, s_low, s_high, v_low, v_high)
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def save_values(self, h_low, h_high, s_low, s_high, v_low, v_high):
        """Save HSV values to console"""
        self.get_logger().info('='*60)
        self.get_logger().info('Current HSV Values:')
        self.get_logger().info(f'Lower: [{h_low}, {s_low}, {v_low}]')
        self.get_logger().info(f'Upper: [{h_high}, {s_high}, {v_high}]')
        self.get_logger().info('')
        self.get_logger().info('Add to BarrelColor enum in autonomous_barrel_collector.py:')
        self.get_logger().info(f'YOUR_COLOR = ([{h_low}, {s_low}, {v_low}], [{h_high}, {s_high}, {v_high}])')
        self.get_logger().info('='*60)

def main(args=None):
    rclpy.init(args=args)
    node = ColorCalibration()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()