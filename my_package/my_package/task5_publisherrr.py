import rclpy from rclpy.node import Node
from geometry_msgs.msg import Twist

class Task_5_Publisher(Node):

    def __init__(self):

        super().__init__('task_5_publisher')

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.angular.z = 1.0
        self.publisher_.publish(msg)
        self.get_logger().info(f"Publishing: '{msg}'")


def main(args=None):
    rclpy.init(args=args)
    node = Task_5_Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally
        node.destroy_node()
        rclpy.try_shutdown()




if __name__ == '__main__':
    main()

