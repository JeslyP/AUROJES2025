#!/usr/bin/env python3
"""
Simple keyboard teleop for testing robot movement.
Usage: ros2 run solution teleop_keyboard.py (or run directly with Python)

Controls:
  UP/DOWN arrow or W/S: forward/backward
  LEFT/RIGHT arrow or A/D: rotate left/right
  SPACE: stop
  Q: quit
"""

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import termios
import tty

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # Subscribe to the namespace parameter if running in a namespace
        namespace = (self.get_namespace() or '').strip('/')
        self.robot_ns = namespace if namespace else 'robot1'
        
        self.pub = self.create_publisher(Twist, f'/{self.robot_ns}/teleop_twist', 10)
        self.get_logger().info(f'Publishing to /{self.robot_ns}/teleop_twist')
        self.get_logger().info('Keyboard controls: UP/W=forward, DOWN/S=backward, LEFT/A=rotate-left, RIGHT/D=rotate-right, SPACE=stop, Q=quit')
        
    def get_key(self):
        """Get a single key press."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def run(self):
        """Main teleop loop."""
        print('Starting keyboard teleop. Press keys to control robot...')
        while rclpy.ok():
            try:
                # For arrow keys, we need to read the escape sequence
                ch = self.get_key()
                
                if ch == 'q' or ch == 'Q':
                    print('Quitting...')
                    break
                elif ch == ' ':
                    # Space: stop
                    msg = Twist()
                    msg.linear.x = 0.0
                    msg.angular.z = 0.0
                    self.pub.publish(msg)
                    print('STOP')
                elif ch == 'w' or ch == 'W':
                    # W: forward
                    msg = Twist()
                    msg.linear.x = 0.2
                    msg.angular.z = 0.0
                    self.pub.publish(msg)
                    print('FORWARD')
                elif ch == 's' or ch == 'S':
                    # S: backward
                    msg = Twist()
                    msg.linear.x = -0.2
                    msg.angular.z = 0.0
                    self.pub.publish(msg)
                    print('BACKWARD')
                elif ch == 'a' or ch == 'A':
                    # A: rotate left
                    msg = Twist()
                    msg.linear.x = 0.0
                    msg.angular.z = 0.5
                    self.pub.publish(msg)
                    print('ROTATE LEFT')
                elif ch == 'd' or ch == 'D':
                    # D: rotate right
                    msg = Twist()
                    msg.linear.x = 0.0
                    msg.angular.z = -0.5
                    self.pub.publish(msg)
                    print('ROTATE RIGHT')
                elif ch == '\x1b':
                    # ESC or arrow key escape sequence
                    ch2 = self.get_key()
                    if ch2 == '[':
                        ch3 = self.get_key()
                        if ch3 == 'A':  # UP arrow
                            msg = Twist()
                            msg.linear.x = 0.2
                            msg.angular.z = 0.0
                            self.pub.publish(msg)
                            print('FORWARD (arrow)')
                        elif ch3 == 'B':  # DOWN arrow
                            msg = Twist()
                            msg.linear.x = -0.2
                            msg.angular.z = 0.0
                            self.pub.publish(msg)
                            print('BACKWARD (arrow)')
                        elif ch3 == 'C':  # RIGHT arrow
                            msg = Twist()
                            msg.linear.x = 0.0
                            msg.angular.z = -0.5
                            self.pub.publish(msg)
                            print('ROTATE RIGHT (arrow)')
                        elif ch3 == 'D':  # LEFT arrow
                            msg = Twist()
                            msg.linear.x = 0.0
                            msg.angular.z = 0.5
                            self.pub.publish(msg)
                            print('ROTATE LEFT (arrow)')
            except KeyboardInterrupt:
                print('\nInterrupted. Stopping robot...')
                msg = Twist()
                self.pub.publish(msg)
                break
            except Exception as e:
                self.get_logger().warn(f'Error reading key: {e}')
                break

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        node.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
