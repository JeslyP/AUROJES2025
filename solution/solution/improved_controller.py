#!/usr/bin/env python3
"""
AURO 2025 - Improved Robot Controller

This version correctly handles the pickup geometry requirement:
- Barrel must be BEHIND the robot (within ±15° of directly behind)
- Robot must be within 0.45m of the barrel

Strategy for pickup:
1. Approach barrel using camera until ~0.6m away
2. Drive PAST the barrel (continue forward)
3. Turn around 180°
4. The barrel is now BEHIND us - call pickup

This is more reliable than trying to reverse toward the barrel.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from assessment_interfaces.msg import ItemList, ZoneList, Item, Zone
from assessment_interfaces.srv import PickUpItem, OffloadItem, Decontaminate

import math
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


class State(Enum):
    INIT = auto()
    EXPLORE = auto()
    APPROACH_BARREL = auto()
    DRIVE_PAST = auto()
    TURN_AROUND = auto()
    FINAL_APPROACH = auto()
    PICKUP = auto()
    GO_DECONTAM = auto()
    DECONTAMINATE = auto()
    GO_DROPOFF = auto()
    DROPOFF = auto()
    RECOVERY = auto()


@dataclass
class BarrelTarget:
    x_offset: float = 0.0
    y_offset: float = 0.0
    size: float = 0.0
    color: str = ""
    world_x: float = 0.0
    world_y: float = 0.0


class ImprovedController(Node):
    """
    Improved robot controller with correct pickup geometry handling.
    """
    
    # Zone coordinates
    DECONTAM_ZONE = (7.5, 9.4)
    COLLECTION_A = (13.5, 9.4)
    COLLECTION_B = (19.5, 9.4)
    
    # Search waypoints - snake pattern for coverage
    WAYPOINTS = [
        (3.0, 3.0), (7.0, 3.0), (11.0, 3.0), (15.0, 3.0), (20.0, 3.0),
        (20.0, 6.0), (15.0, 6.0), (11.0, 6.0), (7.0, 6.0), (3.0, 6.0),
        (3.0, 9.0), (11.0, 9.0), (16.5, 9.0),
        (16.5, 12.0), (11.0, 12.0), (7.0, 12.0), (3.0, 12.0),
        (3.0, 15.0), (7.0, 15.0), (11.0, 15.0), (15.0, 15.0), (20.0, 15.0),
    ]
    
    # Geometry
    PICKUP_DISTANCE = 0.45
    APPROACH_STOP_DIST = 0.55
    DRIVE_PAST_DIST = 0.7
    
    # Speeds
    APPROACH_SPEED = 0.15
    TURN_SPEED = 0.4
    FINAL_APPROACH_SPEED = 0.08
    
    def __init__(self):
        super().__init__('improved_controller')
        
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.sub_group = ReentrantCallbackGroup()
        self.srv_group = MutuallyExclusiveCallbackGroup()
        
        # State
        self.state = State.INIT
        self.prev_state = None
        self.state_time = time.time()
        
        # Pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # Target tracking
        self.target: Optional[BarrelTarget] = None
        self.barrel_world_pos: Optional[Tuple[float, float]] = None
        
        # Detections
        self.barrels: List[Item] = []
        self.front_clear = True
        
        # Holding state
        self.holding = False
        self.holding_color = ""
        self.collected = 0
        
        # Navigation
        self.wp_idx = 0
        self.nav_active = False
        
        # Turn tracking
        self.turn_start_yaw = 0.0
        
        # Setup
        self._setup_ros()
        
        self.timer = self.create_timer(0.1, self._loop, callback_group=self.timer_group)
        self.get_logger().info('Improved Controller started')
        
    def _setup_ros(self):
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.create_subscription(Odometry, 'odom', self._odom_cb, 10, callback_group=self.sub_group)
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10, callback_group=self.sub_group)
        self.create_subscription(ItemList, 'items', self._items_cb, 10, callback_group=self.sub_group)
        
        self.pickup_cli = self.create_client(PickUpItem, 'pick_up_item', callback_group=self.srv_group)
        self.offload_cli = self.create_client(OffloadItem, 'offload_item', callback_group=self.srv_group)
        self.decontam_cli = self.create_client(Decontaminate, 'decontaminate', callback_group=self.srv_group)
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose', callback_group=self.sub_group)
        
    def _odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges)
        n = len(ranges)
        front = ranges[int(n*0.4):int(n*0.6)]
        front = front[np.isfinite(front)]
        self.front_clear = len(front) == 0 or np.min(front) > 0.30
        
    def _items_cb(self, msg: ItemList):
        self.barrels = list(msg.data)
        
    def _loop(self):
        if self.state != self.prev_state:
            self.get_logger().info(f'State: {self.prev_state} -> {self.state}')
            self.prev_state = self.state
            self.state_time = time.time()
            
        handlers = {
            State.INIT: self._init,
            State.EXPLORE: self._explore,
            State.APPROACH_BARREL: self._approach,
            State.DRIVE_PAST: self._drive_past,
            State.TURN_AROUND: self._turn_around,
            State.FINAL_APPROACH: self._final_approach,
            State.PICKUP: self._pickup,
            State.GO_DECONTAM: self._go_decontam,
            State.DECONTAMINATE: self._decontam,
            State.GO_DROPOFF: self._go_dropoff,
            State.DROPOFF: self._dropoff,
            State.RECOVERY: self._recovery,
        }
        handlers.get(self.state, lambda: None)()
        
    def _init(self):
        if self.nav_client.wait_for_server(timeout_sec=0.5):
            self.state = State.EXPLORE
            
    def _explore(self):
        # Look for barrels
        barrel = self._best_barrel()
        if barrel:
            self.target = barrel
            self._estimate_barrel_world_pos()
            self._cancel_nav()
            self.state = State.APPROACH_BARREL
            return
            
        if not self.nav_active:
            wp = self.WAYPOINTS[self.wp_idx]
            self._start_nav(wp[0], wp[1])
            self.wp_idx = (self.wp_idx + 1) % len(self.WAYPOINTS)
            
    def _approach(self):
        """Approach barrel head-on using camera."""
        if not self.target:
            self.state = State.EXPLORE
            return
            
        barrel = self._find_target()
        if not barrel:
            if time.time() - self.state_time > 4.0:
                self.target = None
                self.state = State.EXPLORE
            else:
                self._rotate(0.3)
            return
            
        self.target = barrel
        self._estimate_barrel_world_pos()
        
        dist = self._est_dist(barrel.size)
        
        cmd = Twist()
        
        # Center the barrel
        if abs(barrel.x_offset) > 0.03:
            cmd.angular.z = -0.8 * barrel.x_offset
            cmd.angular.z = np.clip(cmd.angular.z, -0.4, 0.4)
            
        # Approach
        if dist > self.APPROACH_STOP_DIST:
            if self.front_clear:
                cmd.linear.x = min(self.APPROACH_SPEED, dist * 0.25)
            self.cmd_pub.publish(cmd)
        else:
            # Close enough - now drive past
            self._stop()
            self.state = State.DRIVE_PAST
            
    def _drive_past(self):
        """Drive PAST the barrel so it ends up behind us."""
        elapsed = time.time() - self.state_time
        
        if elapsed < 2.0:  # Drive forward for ~2 seconds
            cmd = Twist()
            if self.front_clear:
                cmd.linear.x = 0.12
            self.cmd_pub.publish(cmd)
        else:
            self._stop()
            self.turn_start_yaw = self.yaw
            self.state = State.TURN_AROUND
            
    def _turn_around(self):
        """Turn 180 degrees so barrel is now behind us."""
        target_yaw = self._normalize(self.turn_start_yaw + math.pi)
        diff = self._normalize(target_yaw - self.yaw)
        
        if abs(diff) < 0.15:  # ~8.5 degrees
            self._stop()
            self.state = State.FINAL_APPROACH
        else:
            cmd = Twist()
            cmd.angular.z = self.TURN_SPEED * np.sign(diff)
            self.cmd_pub.publish(cmd)
            
    def _final_approach(self):
        """
        Final approach - barrel should be behind us now.
        Back up a tiny bit to ensure we're in pickup range.
        """
        elapsed = time.time() - self.state_time
        
        if elapsed < 1.5:  # Back up briefly
            cmd = Twist()
            cmd.linear.x = -self.FINAL_APPROACH_SPEED
            self.cmd_pub.publish(cmd)
        else:
            self._stop()
            self.state = State.PICKUP
            
    def _pickup(self):
        self._stop()
        
        if not self.pickup_cli.wait_for_service(timeout_sec=1.0):
            self.state = State.RECOVERY
            return
            
        req = PickUpItem.Request()
        future = self.pickup_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        
        if future.result() and future.result().success:
            self.get_logger().info('Pickup SUCCESS!')
            self.holding = True
            self.holding_color = self.target.color if self.target else ""
            self.collected += 1
            self.target = None
            
            if 'red' in self.holding_color.lower():
                self.state = State.GO_DECONTAM
            else:
                self.state = State.GO_DROPOFF
        else:
            self.get_logger().warn('Pickup FAILED - trying again')
            # Try adjusting position
            self.state = State.FINAL_APPROACH
            self.state_time = time.time()
            
    def _go_decontam(self):
        if not self.nav_active:
            self._start_nav(self.DECONTAM_ZONE[0], self.DECONTAM_ZONE[1])
        if self._in_zone(self.DECONTAM_ZONE):
            self._cancel_nav()
            self.state = State.DECONTAMINATE
            
    def _decontam(self):
        self._stop()
        if not self.decontam_cli.wait_for_service(timeout_sec=1.0):
            return
        req = Decontaminate.Request()
        future = self.decontam_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() and future.result().success:
            self.get_logger().info('Decontamination SUCCESS!')
            self.state = State.GO_DROPOFF
            
    def _go_dropoff(self):
        if not self.holding:
            self.state = State.EXPLORE
            return
        zone = self._closer_zone()
        if not self.nav_active:
            self._start_nav(zone[0], zone[1])
        if self._in_zone(zone):
            self._cancel_nav()
            self.state = State.DROPOFF
            
    def _dropoff(self):
        self._stop()
        if not self.offload_cli.wait_for_service(timeout_sec=1.0):
            return
        req = OffloadItem.Request()
        future = self.offload_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() and future.result().success:
            self.get_logger().info(f'Dropoff SUCCESS! Total: {self.collected}')
            self.holding = False
            self.holding_color = ""
            self.state = State.EXPLORE
        else:
            # Move and retry
            cmd = Twist()
            cmd.linear.x = 0.1
            self.cmd_pub.publish(cmd)
            time.sleep(0.5)
            self._stop()
            
    def _recovery(self):
        elapsed = time.time() - self.state_time
        if elapsed < 2.0:
            cmd = Twist()
            cmd.linear.x = -0.1
            self.cmd_pub.publish(cmd)
        elif elapsed < 4.0:
            cmd = Twist()
            cmd.angular.z = 0.4
            self.cmd_pub.publish(cmd)
        else:
            self._stop()
            self.target = None
            self.state = State.EXPLORE
            
    # Helpers
    def _best_barrel(self) -> Optional[BarrelTarget]:
        valid = [b for b in self.barrels if b.diameter > 0.01]
        if not valid:
            return None
        best = max(valid, key=lambda b: b.diameter)
        return BarrelTarget(best.x, best.y, best.diameter, best.colour)
        
    def _find_target(self) -> Optional[BarrelTarget]:
        if not self.target:
            return None
        matching = [b for b in self.barrels 
                   if b.colour.lower() == self.target.color.lower() and b.diameter > 0.01]
        if matching:
            best = max(matching, key=lambda b: b.diameter)
            return BarrelTarget(best.x, best.y, best.diameter, best.colour)
        return None
        
    def _estimate_barrel_world_pos(self):
        """Estimate barrel position in world coordinates."""
        if not self.target:
            return
        dist = self._est_dist(self.target.size)
        angle = self.yaw - self.target.x_offset * 0.5  # Approximate
        self.target.world_x = self.x + dist * math.cos(angle)
        self.target.world_y = self.y + dist * math.sin(angle)
        
    def _est_dist(self, size: float) -> float:
        if size <= 0:
            return 10.0
        return max(0.3, 0.18 / (size + 0.01))
        
    def _rotate(self, speed: float):
        cmd = Twist()
        cmd.angular.z = speed
        self.cmd_pub.publish(cmd)
        
    def _stop(self):
        self.cmd_pub.publish(Twist())
        
    def _in_zone(self, zone: Tuple[float, float], r: float = 1.0) -> bool:
        return math.sqrt((self.x - zone[0])**2 + (self.y - zone[1])**2) < r
        
    def _closer_zone(self) -> Tuple[float, float]:
        da = math.sqrt((self.x - self.COLLECTION_A[0])**2 + (self.y - self.COLLECTION_A[1])**2)
        db = math.sqrt((self.x - self.COLLECTION_B[0])**2 + (self.y - self.COLLECTION_B[1])**2)
        return self.COLLECTION_A if da < db else self.COLLECTION_B
        
    def _start_nav(self, x: float, y: float):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        self.nav_active = True
        self._nav_future = self.nav_client.send_goal_async(goal)
        self._nav_future.add_done_callback(self._nav_response)
        
    def _nav_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.nav_active = False
            return
        self._nav_result = gh.get_result_async()
        self._nav_result.add_done_callback(lambda f: setattr(self, 'nav_active', False))
        
    def _cancel_nav(self):
        self.nav_active = False
        
    @staticmethod
    def _normalize(a: float) -> float:
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    controller = ImprovedController()
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
