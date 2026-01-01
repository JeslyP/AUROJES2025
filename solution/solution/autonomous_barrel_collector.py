import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import BarrelList, ZoneList
from auro_interfaces.srv import ItemRequest
from enum import Enum
import math

class State(Enum):
    EXPLORING = 1
    APPROACHING_BARREL = 2
    PICKING_UP = 3
    CARRYING_TO_ZONE = 4
    APPROACHING_ZONE = 5
    DEPOSITING = 6
    NEEDS_DECONTAMINATION = 7
    APPROACHING_DECON = 8
    DECONTAMINATING = 9
    AVOIDING_WALL = 10

class AutonomousBarrelCollector(Node):
    def __init__(self):
        super().__init__('autonomous_barrel_collector')
        
        # Parameters
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'/{self.robot_id}/cmd_vel', 10)
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, f'/{self.robot_id}/scan', self.laser_callback, 10)
        self.barrels_sub = self.create_subscription(
            BarrelList, f'/{self.robot_id}/barrels', self.barrels_callback, 10)
        self.zones_sub = self.create_subscription(
            ZoneList, f'/{self.robot_id}/zones', self.zones_callback, 10)
        
        # Service clients
        self.pickup_client = self.create_service_client(ItemRequest, '/pick_up_item')
        self.offload_client = self.create_service_client(ItemRequest, '/offload_item')
        self.decon_client = self.create_service_client(ItemRequest, '/decontaminate')
        
        # State machine
        self.state = State.EXPLORING
        
        # Detection data
        self.visible_barrels = []
        self.visible_zones = []
        self.target_barrel = None
        self.target_zone = None
        
        # Carrying state
        self.carrying_barrel = False
        self.barrel_color = None
        self.is_contaminated = False
        
        # Laser scan data
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        
        # Parameters
        self.barrel_approach_threshold = 0.35  # When to pick up
        self.zone_approach_threshold = 0.25    # When to deposit
        self.decon_approach_threshold = 0.25   # When to decontaminate
        self.wall_stop_distance = 0.5
        self.linear_speed = 0.2
        self.approach_speed = 0.12
        self.angular_speed = 0.5
        
        # Statistics
        self.barrels_collected = 0
        self.barrels_deposited = 0
        
        # Control timer
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('='*60)
        self.get_logger().info(f'Autonomous Barrel Collector Started')
        self.get_logger().info(f'Robot ID: {self.robot_id}')
        self.get_logger().info('Task: Collect barrels → Deposit in GREEN zones')
        self.get_logger().info('='*60)
    
    def create_service_client(self, srv_type, srv_name):
        """Create service client and wait for service"""
        client = self.create_client(srv_type, srv_name)
        return client
    
    def laser_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Front distance
        front_indices = list(range(0, 10)) + list(range(len(msg.ranges)-10, len(msg.ranges)))
        front_readings = [msg.ranges[i] for i in front_indices if not math.isinf(msg.ranges[i])]
        self.front_distance = min(front_readings) if front_readings else float('inf')
        
        # Left distance (90 degrees)
        left_start = len(msg.ranges) // 4
        left_indices = range(left_start - 10, left_start + 10)
        left_readings = [msg.ranges[i] for i in left_indices if i < len(msg.ranges) and not math.isinf(msg.ranges[i])]
        self.left_distance = min(left_readings) if left_readings else float('inf')
        
        # Right distance (270 degrees)
        right_start = 3 * len(msg.ranges) // 4
        right_indices = range(right_start - 10, right_start + 10)
        right_readings = [msg.ranges[i] for i in right_indices if i < len(msg.ranges) and not math.isinf(msg.ranges[i])]
        self.right_distance = min(right_readings) if right_readings else float('inf')
    
    def barrels_callback(self, msg):
        """Process barrel detections from visual sensor"""
        self.visible_barrels = msg.barrels
    
    def zones_callback(self, msg):
        """Process zone detections from visual sensor"""
        self.visible_zones = msg.zones
    
    def control_loop(self):
        """Main control loop with state machine"""
        twist = Twist()
        
        # State machine
        if self.state == State.EXPLORING:
            self.explore(twist)
        elif self.state == State.APPROACHING_BARREL:
            self.approach_barrel(twist)
        elif self.state == State.PICKING_UP:
            self.pickup_barrel(twist)
        elif self.state == State.CARRYING_TO_ZONE:
            self.carry_to_zone(twist)
        elif self.state == State.APPROACHING_ZONE:
            self.approach_zone(twist)
        elif self.state == State.DEPOSITING:
            self.deposit_barrel(twist)
        elif self.state == State.NEEDS_DECONTAMINATION:
            self.seek_decontamination(twist)
        elif self.state == State.APPROACHING_DECON:
            self.approach_decon(twist)
        elif self.state == State.DECONTAMINATING:
            self.decontaminate(twist)
        elif self.state == State.AVOIDING_WALL:
            self.avoid_wall(twist)
        
        self.cmd_vel_pub.publish(twist)
    
    def explore(self, twist):
        """Explore and search for barrels"""
        # Check for wall
        if self.front_distance < self.wall_stop_distance:
            self.get_logger().info('Wall detected! Avoiding...')
            self.state = State.AVOIDING_WALL
            return
        
        # Look for barrels
        if len(self.visible_barrels) > 0:
            # Find closest barrel
            self.target_barrel = min(self.visible_barrels, key=lambda b: abs(b.x))
            self.get_logger().info(f'Barrel detected! Color: {self.target_barrel.colour}, x={self.target_barrel.x:.2f}')
            self.state = State.APPROACHING_BARREL
            return
        
        # Continue exploring
        twist.linear.x = self.linear_speed
        twist.angular.z = 0.0
    
    def approach_barrel(self, twist):
        """Approach detected barrel"""
        if len(self.visible_barrels) == 0 or self.target_barrel is None:
            self.get_logger().warn('Lost sight of barrel, returning to exploration')
            self.target_barrel = None
            self.state = State.EXPLORING
            return
        
        # Update target to closest barrel
        self.target_barrel = min(self.visible_barrels, key=lambda b: abs(b.x))
        
        # Check if close enough
        if self.front_distance < self.barrel_approach_threshold:
            self.get_logger().info('Reached barrel! Initiating pickup...')
            self.state = State.PICKING_UP
            return
        
        # Check for walls
        if self.front_distance < self.wall_stop_distance:
            self.state = State.AVOIDING_WALL
            return
        
        # Align and approach
        error = self.target_barrel.x / 320.0  # Normalize by image width
        twist.linear.x = self.approach_speed
        twist.angular.z = -error * 0.8  # Proportional control
    
    def pickup_barrel(self, twist):
        """Pick up barrel using service"""
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        # Call pickup service
        request = ItemRequest.Request()
        request.robot_id = self.robot_id
        
        if self.pickup_client.wait_for_service(timeout_sec=1.0):
            future = self.pickup_client.call_async(request)
            future.add_done_callback(self.pickup_response_callback)
        else:
            self.get_logger().error('Pickup service not available')
            self.state = State.EXPLORING
    
    def pickup_response_callback(self, future):
        """Handle pickup service response"""
        try:
            response = future.result()
            if response.success:
                self.carrying_barrel = True
                self.barrel_color = self.target_barrel.colour if self.target_barrel else 'unknown'
                self.is_contaminated = (self.barrel_color.lower() == 'red')
                
                self.barrels_collected += 1
                self.get_logger().info(f'✅ Picked up {self.barrel_color} barrel!')
                self.get_logger().info(f'Contaminated: {self.is_contaminated}')
                self.get_logger().info(f'Total collected: {self.barrels_collected}')
                
                self.target_barrel = None
                self.state = State.CARRYING_TO_ZONE
            else:
                self.get_logger().warn(f'Failed to pick up barrel: {response.message}')
                self.state = State.EXPLORING
        except Exception as e:
            self.get_logger().error(f'Pickup service call failed: {e}')
            self.state = State.EXPLORING
    
    def carry_to_zone(self, twist):
        """Search for green collection zone while carrying barrel"""
        # Check for wall
        if self.front_distance < self.wall_stop_distance:
            self.state = State.AVOIDING_WALL
            return
        
        # Look for GREEN zones
        green_zones = [z for z in self.visible_zones if z.colour.lower() == 'green']
        
        if len(green_zones) > 0:
            self.target_zone = min(green_zones, key=lambda z: abs(z.x))
            self.get_logger().info(f'Green zone detected! x={self.target_zone.x:.2f}')
            self.state = State.APPROACHING_ZONE
            return
        
        # Continue searching
        twist.linear.x = self.linear_speed * 0.8  # Slower while carrying
        twist.angular.z = 0.1  # Gentle turn to scan area
    
    def approach_zone(self, twist):
        """Approach green collection zone"""
        # Look for green zones
        green_zones = [z for z in self.visible_zones if z.colour.lower() == 'green']
        
        if len(green_zones) == 0:
            self.get_logger().warn('Lost sight of zone')
            self.state = State.CARRYING_TO_ZONE
            return
        
        self.target_zone = min(green_zones, key=lambda z: abs(z.x))
        
        # Check if close enough to deposit
        if self.front_distance < self.zone_approach_threshold or self.target_zone.size > 15000:
            self.get_logger().info('Reached zone! Depositing barrel...')
            self.state = State.DEPOSITING
            return
        
        # Check for walls
        if self.front_distance < self.wall_stop_distance:
            self.state = State.AVOIDING_WALL
            return
        
        # Align and approach
        error = self.target_zone.x / 320.0
        twist.linear.x = self.approach_speed
        twist.angular.z = -error * 0.6
    
    def deposit_barrel(self, twist):
        """Deposit barrel in zone"""
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        # Call offload service
        request = ItemRequest.Request()
        request.robot_id = self.robot_id
        
        if self.offload_client.wait_for_service(timeout_sec=1.0):
            future = self.offload_client.call_async(request)
            future.add_done_callback(self.offload_response_callback)
        else:
            self.get_logger().error('Offload service not available')
            self.carrying_barrel = False
            self.state = State.EXPLORING
    
    def offload_response_callback(self, future):
        """Handle offload service response"""
        try:
            response = future.result()
            if response.success:
                self.barrels_deposited += 1
                self.get_logger().info(f'✅ Deposited {self.barrel_color} barrel successfully!')
                self.get_logger().info(f'Total deposited: {self.barrels_deposited}')
                
                # Check if need decontamination
                if self.is_contaminated:
                    self.get_logger().warn('⚠️  Robot contaminated! Seeking decontamination...')
                    self.state = State.NEEDS_DECONTAMINATION
                else:
                    self.carrying_barrel = False
                    self.barrel_color = None
                    self.state = State.EXPLORING
            else:
                self.get_logger().warn(f'Failed to deposit: {response.message}')
                self.state = State.CARRYING_TO_ZONE
        except Exception as e:
            self.get_logger().error(f'Offload service call failed: {e}')
            self.state = State.CARRYING_TO_ZONE
    
    def seek_decontamination(self, twist):
        """Search for cyan decontamination zone"""
        # Look for CYAN zones
        cyan_zones = [z for z in self.visible_zones if z.colour.lower() == 'cyan']
        
        if len(cyan_zones) > 0:
            self.target_zone = min(cyan_zones, key=lambda z: abs(z.x))
            self.get_logger().info(f'Cyan decon zone found! x={self.target_zone.x:.2f}')
            self.state = State.APPROACHING_DECON
            return
        
        # Continue searching
        twist.linear.x = self.linear_speed
        twist.angular.z = 0.2  # Turn to scan
    
    def approach_decon(self, twist):
        """Approach cyan decontamination zone"""
        # Look for cyan zones
        cyan_zones = [z for z in self.visible_zones if z.colour.lower() == 'cyan']
        
        if len(cyan_zones) == 0:
            self.get_logger().warn('Lost sight of decon zone')
            self.state = State.NEEDS_DECONTAMINATION
            return
        
        self.target_zone = min(cyan_zones, key=lambda z: abs(z.x))
        
        # Check if in zone
        if self.front_distance < self.decon_approach_threshold or self.target_zone.size > 15000:
            self.get_logger().info('In decon zone! Decontaminating...')
            self.state = State.DECONTAMINATING
            return
        
        # Align and approach
        error = self.target_zone.x / 320.0
        twist.linear.x = self.approach_speed
        twist.angular.z = -error * 0.6
    
    def decontaminate(self, twist):
        """Call decontamination service"""
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        request = ItemRequest.Request()
        request.robot_id = self.robot_id
        
        if self.decon_client.wait_for_service(timeout_sec=1.0):
            future = self.decon_client.call_async(request)
            future.add_done_callback(self.decon_response_callback)
        else:
            self.get_logger().error('Decontamination service not available')
            self.is_contaminated = False
            self.carrying_barrel = False
            self.state = State.EXPLORING
    
    def decon_response_callback(self, future):
        """Handle decontamination response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('✅ Decontamination successful!')
                self.is_contaminated = False
                self.carrying_barrel = False
                self.barrel_color = None
                self.state = State.EXPLORING
            else:
                self.get_logger().warn(f'Decontamination failed: {response.message}')
                self.state = State.NEEDS_DECONTAMINATION
        except Exception as e:
            self.get_logger().error(f'Decontamination service failed: {e}')
            self.state = State.NEEDS_DECONTAMINATION
    
    def avoid_wall(self, twist):
        """Avoid walls by stopping and rotating"""
        twist.linear.x = 0.0
        
        # Decide rotation direction based on space
        if self.left_distance > self.right_distance:
            self.get_logger().info(f'Turning LEFT (L:{self.left_distance:.2f}m > R:{self.right_distance:.2f}m)')
            twist.angular.z = self.angular_speed
        else:
            self.get_logger().info(f'Turning RIGHT (R:{self.right_distance:.2f}m > L:{self.left_distance:.2f}m)')
            twist.angular.z = -self.angular_speed
        
        # Check if cleared
        if self.front_distance > self.wall_stop_distance * 1.5:
            self.get_logger().info('Wall cleared!')
            if self.carrying_barrel:
                self.state = State.CARRYING_TO_ZONE
            else:
                self.state = State.EXPLORING

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