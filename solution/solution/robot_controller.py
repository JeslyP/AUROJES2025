import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from enum import Enum

# ROS Messages
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# Custom Interfaces
from assessment_interfaces.msg import BarrelList, BarrelHolders, RadiationList
from auro_interfaces.srv import ItemRequest

# For Dynamic Parameters (LiDAR Mask)
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class State(Enum):
    SEARCHING = 0
    APPROACHING = 1
    POSITIONING = 2
    PICKING_UP = 3
    DELIVERING = 4
    OFFLOADING = 5
    CLEARING_SPACE = 6
    DECONTAMINATING = 7  # Go to cyan zone and decontaminate

class CollectPhase(Enum):
    ALIGN = 0
    APPROACH = 1
    TURN_AROUND = 2
    BACKUP = 3

class DecontaminatePhase(Enum):
    NAVIGATING = 0
    REVERSING = 1
    CALLING_SERVICE = 2

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # 1. PARAMETERS
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        
        self.robot_name = self.get_namespace().strip('/')
        if not self.robot_name: 
            self.robot_name = 'robot1'

        # 2. SETUP NAVIGATOR
        self.navigator = BasicNavigator()
        self.set_initial_pose()
        self.navigator.waitUntilNav2Active()

        # 3. SENSORS
        self.create_subscription(BarrelList, 'barrels', self.barrel_callback, 10)
        self.create_subscription(LaserScan, 'scan_filtered', self.scan_callback, 10)
        self.create_subscription(BarrelHolders, '/barrel_holders', self.holders_callback, 10)
        self.create_subscription(RadiationList, '/radiation_levels', self.radiation_callback, 10)

        # 4. PUBLISHERS
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # 5. SERVICE CLIENTS
        self.cb_group = ReentrantCallbackGroup()
        self.pickup_client = self.create_client(ItemRequest, '/pick_up_item', callback_group=self.cb_group)
        self.offload_client = self.create_client(ItemRequest, '/offload_item', callback_group=self.cb_group)
        self.decontaminate_client = self.create_client(ItemRequest, '/decontaminate', callback_group=self.cb_group)
        self.mask_client = self.create_client(SetParameters, f'/{self.robot_name}/dynamic_mask/set_parameters', callback_group=self.cb_group)
        
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Pickup Service not found!")

        # 6. DATA & STATE
        self.state = State.SEARCHING
        self.collect_phase = CollectPhase.ALIGN
        self.decontaminate_phase = DecontaminatePhase.NAVIGATING
        self.barrels = []
        self.holding_barrel = False
        self.radiation_level = 0  # Track radiation level
        self.search_enabled = False 
        self.phase_start_time = None
        self.service_future = None
        self.barrels_collected = 0 
        self.offload_start_time = None 
        self.forward_start_time = None
        self.decontaminate_start_time = None

        # Decontamination threshold
        self.DECONTAMINATION_THRESHOLD = 300

        # LiDAR Data
        self.front_dist = float('inf')
        self.left_dist = float('inf')
        self.right_dist = float('inf')
        self.back_dist = float('inf') 
        
        # 7. PATROL ROUTE
        self.waypoints = [
            {'x': 0.053, 'y': 7.213, 'name': 'Start Area'},
            {'x': 5.21, 'y': 5.17, 'name': 'Right Corridor Bottom'},
            {'x': 9.351, 'y': 4.7, 'name': 'Right Corridor Top'}, 
            {'x': 8.000, 'y': 9.041, 'name': 'Left Corridor Top'},
            {'x': 10.050, 'y': 14.850, 'name': 'Big Room Entrance'},
            {'x': 6.150, 'y': 14.811, 'name': 'Big Room Bottom Right'},
            {'x': 6.426, 'y': 19.251, 'name': 'Big Room Bottom Center'},
            {'x': 6.366, 'y': 23.279, 'name': 'Big Room Bottom Left'},
            {'x': 10.152, 'y': 23.138, 'name': 'Big Room Middle Left'},
            {'x': 14.433, 'y': 23.024, 'name': 'Big Room Top Left'},
            {'x': 14.264, 'y': 18.680, 'name': 'Big Room Top Middle'},
            {'x': 14.286, 'y': 14.876, 'name': 'Big Room Top Right'},
            {'x': 10.134, 'y': 19.612, 'name': 'Big Room Center'},
            {'x': 10.050, 'y': 14.850, 'name': 'Big Room Entrance (Exit)'},
        ]

        # Decontamination zone (cyan zone)
        self.decontamination_zone = {'x': 9.58, 'y': -0.33}
        
        self.current_wp_index = 1 
        self.nav_goal_sent = False
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Robot Controller Started")

    def set_initial_pose(self):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = 0.053
        pose.pose.position.y = 7.213
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self.navigator.setInitialPose(pose)

    def set_mask(self, enabled):
        """Enables mask with WIDE angles to hide the barrel"""
        req = SetParameters.Request()
        val_enabled = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enabled)
        
        val_start = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=100)
        val_end = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=260)

        req.parameters = [
            Parameter(name='mask_enabled', value=val_enabled),
            Parameter(name='ignore_sector_start', value=val_start),
            Parameter(name='ignore_sector_end', value=val_end)
        ]
        self.mask_client.call_async(req)
        self.get_logger().info(f"LiDAR Mask: {enabled}")

    def barrel_callback(self, msg):
        self.barrels = msg.data

    def scan_callback(self, msg):
        ranges = msg.ranges
        if not ranges: return
        n = len(ranges)
        
        front_slice = ranges[0:10] + ranges[-10:]
        left_idx = int(n / 4)
        left_slice = ranges[left_idx-15 : left_idx+15]
        back_idx = int(n / 2)
        back_slice = ranges[back_idx-15 : back_idx+15]
        right_idx = int(3 * n / 4)
        right_slice = ranges[right_idx-15 : right_idx+15]

        def get_min(slice_data):
            valid = [r for r in slice_data if msg.range_min < r < msg.range_max]
            return min(valid) if valid else float('inf')

        self.front_dist = get_min(front_slice)
        self.left_dist = get_min(left_slice)
        self.right_dist = get_min(right_slice)
        self.back_dist = get_min(back_slice)

    def holders_callback(self, msg):
        self.holding_barrel = False
        for h in msg.data:
            if h.robot_id == self.robot_name:
                self.holding_barrel = True
                break

    def radiation_callback(self, msg):
        """Callback for radiation levels."""
        for radiation in msg.data:
            if radiation.robot_id == self.robot_name:
                self.radiation_level = radiation.level
                break

    def get_best_barrel(self):
        if not self.barrels:
            return None
        
        # LOGIC 1: If we are SEARCHING, look for the closest/biggest one
        if self.state == State.SEARCHING:
            return max(self.barrels, key=lambda b: b.size)
        
        # LOGIC 2: If we are APPROACHING, Focus on the one in the CENTER!
        # This prevents switching to a neighbor just because it looks slightly bigger.
        elif self.state == State.APPROACHING:
            CAMERA_CENTER = 320
            # Find the barrel with the smallest X distance to the center
            return min(self.barrels, key=lambda b: abs(b.x - CAMERA_CENTER))
        
        # Default fallback
        return max(self.barrels, key=lambda b: b.size)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def elapsed(self):
        if not self.phase_start_time: return 0.0
        return (self.get_clock().now() - self.phase_start_time).nanoseconds / 1e9

    def should_decontaminate(self):
        """Check if robot needs decontamination."""
        return self.radiation_level >= self.DECONTAMINATION_THRESHOLD

    def control_loop(self):
        
        # ========================================================
        # STATE 1: SEARCHING
        # ========================================================
        if self.state == State.SEARCHING:
            if self.search_enabled:
                best_barrel = self.get_best_barrel()
                if best_barrel and not self.holding_barrel:
                    self.get_logger().info(f"👀 BARREL SPOTTED! Size: {best_barrel.size}")
                    self.navigator.cancelTask()
                    self.stop_robot()
                    self.state = State.APPROACHING
                    self.collect_phase = CollectPhase.ALIGN 
                    self.nav_goal_sent = False
                    return

            if not self.nav_goal_sent:
                wp = self.waypoints[self.current_wp_index]
                self.get_logger().info(f"Patrolling to: {wp['name']}")
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.navigator.get_clock().now().to_msg()
                goal.pose.position.x = wp['x']
                goal.pose.position.y = wp['y']
                
                # --- SPECIAL LOGIC FOR LEFT CORRIDOR ---
                if wp['name'] == 'Left Corridor Top':
                    # Face South (Down the corridor)
                    goal.pose.orientation.z = 1.0
                    goal.pose.orientation.w = 0.0
                else:
                    # Face East
                    goal.pose.orientation.z = 0.0
                    goal.pose.orientation.w = 1.0

                self.navigator.goToPose(goal)
                self.nav_goal_sent = True
            
            elif self.navigator.isTaskComplete():
                # --- THIS IS WHERE SEARCH GETS TURNED BACK ON ---
                if self.current_wp_index == 3:
                    self.search_enabled = True
                    self.get_logger().info("⚠️ SEARCH ACTIVATED ⚠️")
                
                self.current_wp_index += 1
                if self.current_wp_index >= len(self.waypoints):
                    self.current_wp_index = 3 
                self.nav_goal_sent = False

        # ========================================================
        # STATE 2: APPROACHING (Corrected)
        # ========================================================
        elif self.state == State.APPROACHING:
            target = self.get_best_barrel()
            
            # 1. Safety Check: If barrel disappears, stop and search again
            if not target:
                self.get_logger().warn("Lost barrel! Back to patrol.")
                self.state = State.SEARCHING
                return

            CAMERA_CENTER = 320
            STOP_DISTANCE = 0.55
            MIN_SIZE = 60000

            twist = Twist()
            error = target.x - CAMERA_CENTER

            # --- PHASE 1: ALIGN (Rotate in place) ---
            if self.collect_phase == CollectPhase.ALIGN:
                # Deadband: Only rotate if error is big (> 10)
                if abs(error) > 10:
                    twist.angular.z = -0.002 * error
                    twist.angular.z = max(-0.5, min(0.5, twist.angular.z))
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.collect_phase = CollectPhase.APPROACH

            # --- PHASE 2: APPROACH (Drive forward) ---
            elif self.collect_phase == CollectPhase.APPROACH:
                if self.front_dist < STOP_DISTANCE and target.size > MIN_SIZE:
                    self.stop_robot()
                    self.get_logger().info("✅ Reached Barrel! Starting positioning...")
                    self.state = State.POSITIONING
                    self.collect_phase = CollectPhase.TURN_AROUND
                    self.phase_start_time = self.get_clock().now()
                else:
                    twist.linear.x = 0.15 
                    
                    # Only steer if the error is significant (> 10 pixels)
                    if abs(error) > 10:
                        steer = -0.0015 * error
                    else:
                        steer = 0.0

                    # Wall Avoidance logic
                    if self.left_dist < 0.35: steer -= 0.3 
                    elif self.right_dist < 0.35: steer += 0.3 
                    
                    twist.angular.z = steer
                    self.cmd_vel_pub.publish(twist)

        # ========================================================
        # STATE 3: POSITIONING
        # ========================================================
        elif self.state == State.POSITIONING:
            t = self.elapsed()
            twist = Twist()

            if self.collect_phase == CollectPhase.TURN_AROUND:
                TURN_DURATION = 6.4 
                if t < TURN_DURATION:
                    twist.angular.z = 0.5
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.collect_phase = CollectPhase.BACKUP
                    self.phase_start_time = self.get_clock().now()
                    self.get_logger().info(f"Turn Complete. Backing up for 1.5s...")

            elif self.collect_phase == CollectPhase.BACKUP:
                BACKUP_TIME = 1.5 
                if t < BACKUP_TIME:
                    twist.linear.x = -0.15 
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.get_logger().info("Backup Finished. Attempting Pickup...")
                    self.state = State.PICKING_UP
                    self.service_future = None

        # ========================================================
        # STATE 4: PICKING UP
        # ========================================================
        elif self.state == State.PICKING_UP:
            if self.service_future is None:
                req = ItemRequest.Request()
                req.robot_id = self.robot_name
                self.service_future = self.pickup_client.call_async(req)
            
            elif self.service_future.done():
                try:
                    res = self.service_future.result()
                    if res.success:
                        self.get_logger().info("🎉 PICKUP SUCCESS!")
                        self.holding_barrel = True
                        self.set_mask(True) 
                        self.navigator.clearAllCostmaps() 
                        self.state = State.DELIVERING 
                        self.nav_goal_sent = False
                    else:
                        self.get_logger().warn(f"❌ Pickup Failed: {res.message}")
                        self.navigator.clearAllCostmaps()
                        self.state = State.SEARCHING 
                        self.nav_goal_sent = False
                except Exception as e:
                    self.get_logger().error(f"Service error: {e}")
                    self.state = State.SEARCHING
                
                self.service_future = None

        # ========================================================
        # STATE 5: DELIVERING (FACING WEST + SAFETY BUFFER)
        # ========================================================
        elif self.state == State.DELIVERING:
            if not self.nav_goal_sent:
                # --- ZONE CONFIGURATION ---
                SPACING_X = 0.7 
                SPACING_Y = 0.7
                ROW_LENGTH = 4  
                ZONE_CAPACITY = 16
                
                zones = [
                    {'name': 'Zone B', 'start_x': 11.7, 'start_y': -8.3},
                    {'name': 'Zone A', 'start_x': 11.7, 'start_y': -14.6}
                ]

                total_count = self.barrels_collected
                zone_index = (total_count // ZONE_CAPACITY) % len(zones)
                current_zone = zones[zone_index]
                local_index = total_count % ZONE_CAPACITY
                
                col = local_index % ROW_LENGTH 
                row = local_index // ROW_LENGTH 

                target_x = current_zone['start_x'] - (row * SPACING_X)
                target_y = current_zone['start_y'] + (col * SPACING_Y)

                self.get_logger().info(f"🚚 Barrel #{total_count + 1} -> {current_zone['name']} at ({target_x:.2f}, {target_y:.2f})")
                
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = self.navigator.get_clock().now().to_msg()
                goal.pose.position.x = target_x
                goal.pose.position.y = target_y
                goal.pose.orientation.z = 1.0
                goal.pose.orientation.w = 0.0
                
                self.navigator.goToPose(goal)
                self.nav_goal_sent = True
            
            elif self.navigator.isTaskComplete():
                if self.navigator.getResult() == TaskResult.SUCCEEDED:
                    self.get_logger().info("Arrived. Starting Reverse Park...")
                    self.state = State.OFFLOADING
                    self.offload_start_time = self.get_clock().now() 
                    self.service_future = None
                else:
                    self.get_logger().warn("Delivery Failed. Retrying...")
                    self.nav_goal_sent = False 

        # ========================================================
        # STATE 6: OFFLOADING (WITH REVERSE PARK)
        # ========================================================
        elif self.state == State.OFFLOADING:
            
            # 1. REVERSE MANEUVER
            t = (self.get_clock().now() - self.offload_start_time).nanoseconds / 1e9
            REVERSE_TIME = 1.8 
            
            if t < REVERSE_TIME:
                twist = Twist()
                twist.linear.x = -0.15 
                self.cmd_vel_pub.publish(twist)
                return 
            else:
                self.stop_robot()
            
            # 2. DROP BARREL
            if self.service_future is None:
                req = ItemRequest.Request()
                req.robot_id = self.robot_name
                self.service_future = self.offload_client.call_async(req)
            
            elif self.service_future.done():
                try:
                    res = self.service_future.result()
                    if res.success:
                        self.get_logger().info("📦 OFFLOAD SUCCESS! Driving forward to clear space...")
                        self.set_mask(False) 
                        self.holding_barrel = False
                        self.barrels_collected += 1
                        
                        self.state = State.CLEARING_SPACE
                        self.forward_start_time = self.get_clock().now()
                    else:
                        self.get_logger().warn("Offload Failed.")
                        self.service_future = None 
                except Exception as e:
                    self.get_logger().error(f"Offload error: {e}")
                self.service_future = None

        # ========================================================
        # STATE 7: CLEARING SPACE
        # ========================================================
        elif self.state == State.CLEARING_SPACE:
            
            t = (self.get_clock().now() - self.forward_start_time).nanoseconds / 1e9
            FORWARD_TIME = 1.0 

            if t < FORWARD_TIME:
                twist = Twist()
                twist.linear.x = 0.15
                self.cmd_vel_pub.publish(twist)
            else:
                self.stop_robot()
                self.get_logger().info("✅ Space cleared.")
                
                time.sleep(0.5)
                self.navigator.clearAllCostmaps()
                
                # Check if we need decontamination
                if self.should_decontaminate():
                    self.get_logger().info(f"☢️ RADIATION LEVEL: {self.radiation_level} >= {self.DECONTAMINATION_THRESHOLD}. Going to decontaminate!")
                    self.state = State.DECONTAMINATING
                    self.decontaminate_phase = DecontaminatePhase.NAVIGATING
                    self.nav_goal_sent = False
                else:
                    self.get_logger().info(f"Radiation level: {self.radiation_level}. Resuming patrol.")
                    self.state = State.SEARCHING 
                    self.nav_goal_sent = False
                    self.current_wp_index = 3 
                    self.search_enabled = False

        # ========================================================
        # STATE 8: DECONTAMINATING
        # ========================================================
        elif self.state == State.DECONTAMINATING:
            
            # --- PHASE 1: NAVIGATE TO CYAN ZONE ---
            if self.decontaminate_phase == DecontaminatePhase.NAVIGATING:
                if not self.nav_goal_sent:
                    self.get_logger().info(f"☢️ Navigating to decontamination zone at ({self.decontamination_zone['x']:.2f}, {self.decontamination_zone['y']:.2f})")
                    
                    goal = PoseStamped()
                    goal.header.frame_id = 'map'
                    goal.header.stamp = self.navigator.get_clock().now().to_msg()
                    goal.pose.position.x = self.decontamination_zone['x']
                    goal.pose.position.y = self.decontamination_zone['y']
                    goal.pose.orientation.z = 0.0
                    goal.pose.orientation.w = 1.0
                    
                    self.navigator.goToPose(goal)
                    self.nav_goal_sent = True
                
                elif self.navigator.isTaskComplete():
                    if self.navigator.getResult() == TaskResult.SUCCEEDED:
                        self.get_logger().info("☢️ Arrived at decontamination zone. Reversing into zone...")
                        self.decontaminate_phase = DecontaminatePhase.REVERSING
                        self.decontaminate_start_time = self.get_clock().now()
                    else:
                        self.get_logger().warn("Failed to reach decontamination zone. Retrying...")
                        self.nav_goal_sent = False
            
            # --- PHASE 2: REVERSE INTO ZONE ---
            elif self.decontaminate_phase == DecontaminatePhase.REVERSING:
                t = (self.get_clock().now() - self.decontaminate_start_time).nanoseconds / 1e9
                REVERSE_TIME = 1.5
                
                if t < REVERSE_TIME:
                    twist = Twist()
                    twist.linear.x = -0.15
                    self.cmd_vel_pub.publish(twist)
                else:
                    self.stop_robot()
                    self.get_logger().info("☢️ In position. Calling decontaminate service...")
                    self.decontaminate_phase = DecontaminatePhase.CALLING_SERVICE
                    self.service_future = None
            
            # --- PHASE 3: CALL DECONTAMINATE SERVICE ---
            elif self.decontaminate_phase == DecontaminatePhase.CALLING_SERVICE:
                if self.service_future is None:
                    if not self.decontaminate_client.wait_for_service(timeout_sec=0.5):
                        self.get_logger().warn("Decontaminate service not available, waiting...")
                        return
                    
                    req = ItemRequest.Request()
                    req.robot_id = self.robot_name
                    self.service_future = self.decontaminate_client.call_async(req)
                    self.get_logger().info(f"☢️ Decontaminate request sent for {self.robot_name}")
                
                elif self.service_future.done():
                    try:
                        res = self.service_future.result()
                        if res.success:
                            self.get_logger().info(f"✅ DECONTAMINATION SUCCESS! {res.message}")
                            self.get_logger().info(f"Radiation level now: {self.radiation_level}")
                        else:
                            self.get_logger().warn(f"❌ Decontamination failed: {res.message}")
                    except Exception as e:
                        self.get_logger().error(f"Decontaminate service error: {e}")
                    
                    self.service_future = None
                    
                    # Drive forward to clear the zone
                    self.get_logger().info("Driving forward to clear decontamination zone...")
                    self.forward_start_time = self.get_clock().now()
                    
                    # Clear costmaps and return to searching
                    time.sleep(0.5)
                    self.navigator.clearAllCostmaps()
                    
                    self.state = State.SEARCHING
                    self.nav_goal_sent = False
                    self.current_wp_index = 3
                    self.search_enabled = False
                    self.get_logger().info("Resuming patrol after decontamination.")

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