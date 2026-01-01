import sys
import rclpy
import random
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from auro_interfaces.srv import ItemRequest
from assessment_interfaces.msg import BarrelList, ZoneList, Barrel, Zone, RadiationList

# --- State Definitions ---
STATE_SEARCH_BARREL = 0
STATE_APPROACH_BARREL = 1
STATE_COLLECT_BARREL = 2
STATE_SEARCH_ZONE = 3
STATE_APPROACH_ZONE = 4
STATE_DEPOSIT_BARREL = 5
STATE_SEARCH_DECON = 6
STATE_APPROACH_DECON = 7
STATE_DECONTAMINATE = 8

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # --- Parameters ---
        self.declare_parameter('robot_id', 'robot1')
        ns = self.get_namespace().strip('/')
        self.robot_id = ns if ns else self.get_parameter('robot_id').value
        self.get_logger().info(f"Controller Started for {self.robot_id}")

        # --- State & Thresholds ---
        self.state = STATE_SEARCH_BARREL
        self.held_item = None
        self.radiation_level = 0
        self.radiation_limit = 40       # Decontaminate if rads > 40
        self.pixel_dist_close = 7500.0  # Size of barrel when close
        self.pixel_dist_zone = 16000.0  # Size of zone when close
        self.obs_dist = 0.5             # Meters to obstacle

        # --- Communication ---
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(BarrelList, 'barrels', self.cb_barrels, qos)
        self.create_subscription(ZoneList, 'zones', self.cb_zones, qos)
        self.create_subscription(LaserScan, 'scan', self.cb_scan, qos)
        self.create_subscription(RadiationList, '/radiation_levels', self.cb_rads, qos)

        self.srv_pickup = self.create_client(ItemRequest, '/pick_up_item')
        self.srv_offload = self.create_client(ItemRequest, '/offload_item')
        self.srv_decon = self.create_client(ItemRequest, '/decontaminate')

        # --- Data Cache ---
        self.barrels = []
        self.zones = []
        self.scan_front = []

        # --- Anti-stuck / escape control ---
        self.escape_end_time = None
        self.turn_dir = 1  # toggles between left/right when escaping

        # --- Startup straight-line phase ---
        self.declare_parameter('start_drive_seconds', 3.0)
        try:
            start_sec = float(self.get_parameter('start_drive_seconds').value)
        except Exception:
            start_sec = 3.0
        self.start_drive_end_time = self.get_clock().now() + rclpy.time.Duration(seconds=start_sec)

        # --- Loop ---
        self.timer = self.create_timer(0.1, self.control_loop)

    def cb_barrels(self, msg): self.barrels = msg.data
    def cb_zones(self, msg): self.zones = msg.data
    def cb_scan(self, msg): 
        # Cache front 60 degrees of scan
        if not msg.ranges: return
        n = len(msg.ranges)
        w = 30 # degrees side
        self.scan_front = msg.ranges[-w:] + msg.ranges[:w]

    def front_min(self) -> float:
        vals = [r for r in self.scan_front if r > 0.01]
        return min(vals) if vals else float('inf')

    def cb_rads(self, msg):
        for r in msg.data:
            if r.robot_id == self.robot_id:
                self.radiation_level = r.level
                break

    def get_target(self, items, type_filter=None):
        best = None
        max_sz = -1.0
        for i in items:
            if type_filter is not None:
                # Barrel uses 'colour', Zone uses 'zone'
                val = getattr(i, 'colour', getattr(i, 'zone', -1))
                if val != type_filter: continue
            if i.size > max_sz:
                max_sz = i.size
                best = i
        return best

    def check_safety(self):
        # Return True if obstacle imminent
        if not self.scan_front: return False
        # Filter 0.0 values (sensor errors) and check threshold
        return any(0.01 < r < self.obs_dist for r in self.scan_front)

    def call_srv(self, client):
        if client.wait_for_service(0.5):
            req = ItemRequest.Request()
            req.robot_id = self.robot_id
            client.call_async(req)

    def control_loop(self):
        twist = Twist()
        
        # --- High Priority: Decontamination Trigger ---
        if self.radiation_level >= self.radiation_limit:
            # If not already dealing with decon, switch state
            if self.state < STATE_SEARCH_DECON:
                self.get_logger().warn(f"Radiation {self.radiation_level}! Seeking Decon.")
                self.state = STATE_SEARCH_DECON

        # --- Timed Escape (anti-stuck) ---
        if self.escape_end_time is not None:
            if self.get_clock().now() < self.escape_end_time:
                twist.linear.x = -0.12
                twist.angular.z = 0.6 * self.turn_dir
                self.cmd_vel_pub.publish(twist)
                return
            self.escape_end_time = None

        # --- High Priority: Safety Override ---
        # If obstacle very close, initiate escape (unless strictly interacting)
        interacting = self.state in [STATE_COLLECT_BARREL, STATE_DEPOSIT_BARREL, STATE_DECONTAMINATE]
        if not interacting:
            fm = self.front_min()
            if fm < 0.35:
                self.turn_dir *= -1
                self.escape_end_time = self.get_clock().now() + rclpy.time.Duration(seconds=1.4)
                twist.linear.x = -0.12
                twist.angular.z = 0.6 * self.turn_dir
                self.cmd_vel_pub.publish(twist)
                return

        # --- Startup straight-line drive ---
        if self.start_drive_end_time is not None:
            if self.get_clock().now() < self.start_drive_end_time:
                twist.linear.x = 0.22
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.start_drive_end_time = None

        # --- FSM Logic ---
        if self.state == STATE_SEARCH_BARREL:
            target = self.get_target(self.barrels) # Any barrel
            if target:
                self.state = STATE_APPROACH_BARREL
            else:
                twist.linear.x = 0.15  # Move forward while searching
                twist.angular.z = 0.3   # Gentle turn to scan

        elif self.state == STATE_APPROACH_BARREL:
            target = self.get_target(self.barrels)
            if not target: 
                self.state = STATE_SEARCH_BARREL
                return
            # Visual servoing with clamped angular rate and minimum forward speed
            twist.linear.x = 0.20
            twist.angular.z = max(min(0.002 * target.x, 0.4), -0.4)
            
            if target.size > self.pixel_dist_close:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.held_item = target.colour
                self.state = STATE_COLLECT_BARREL

        elif self.state == STATE_COLLECT_BARREL:
            self.call_srv(self.srv_pickup)
            self.state = STATE_SEARCH_ZONE

        elif self.state == STATE_SEARCH_ZONE:
            target = self.get_target(self.zones, Zone.ZONE_GREEN)
            if target:
                self.state = STATE_APPROACH_ZONE
            else:
                twist.linear.x = 0.12  # Move forward while searching
                twist.angular.z = -0.35 # Turn opposite direction from barrel search

        elif self.state == STATE_APPROACH_ZONE:
            target = self.get_target(self.zones, Zone.ZONE_GREEN)
            if not target:
                self.state = STATE_SEARCH_ZONE
                return
            twist.linear.x = 0.18
            twist.angular.z = max(min(0.002 * target.x, 0.4), -0.4)
            
            if target.size > self.pixel_dist_zone:
                self.state = STATE_DEPOSIT_BARREL

        elif self.state == STATE_DEPOSIT_BARREL:
            twist.linear.x = 0.0
            self.call_srv(self.srv_offload)
            self.held_item = None
            self.state = STATE_SEARCH_BARREL

        elif self.state == STATE_SEARCH_DECON:
            target = self.get_target(self.zones, Zone.ZONE_CYAN)
            if target:
                self.state = STATE_APPROACH_DECON
            else:
                twist.linear.x = 0.12
                twist.angular.z = 0.4  # Moderate turn rate

        elif self.state == STATE_APPROACH_DECON:
            target = self.get_target(self.zones, Zone.ZONE_CYAN)
            if not target:
                self.state = STATE_SEARCH_DECON
                return
            twist.linear.x = 0.18
            twist.angular.z = max(min(0.002 * target.x, 0.4), -0.4)
            
            if target.size > self.pixel_dist_zone:
                self.state = STATE_DECONTAMINATE

        elif self.state == STATE_DECONTAMINATE:
            twist.linear.x = 0.0
            self.call_srv(self.srv_decon)
            if self.radiation_level < 5:
                self.get_logger().info("Clean. Resuming.")
                self.state = STATE_SEARCH_ZONE if self.held_item else STATE_SEARCH_BARREL

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__': main()