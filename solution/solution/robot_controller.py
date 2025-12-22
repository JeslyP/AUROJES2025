import sys
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import Twist

from assessment_interfaces.msg import BarrelList, ZoneList, BarrelHolders, RadiationList
from auro_interfaces.srv import ItemRequest


class ControllerState(Enum):
    SEARCH_BARREL = auto()
    APPROACH_BARREL = auto()
    PICKING_UP = auto()
    SEARCH_GREEN_ZONE = auto()
    APPROACH_GREEN_ZONE = auto()
    OFFLOADING = auto()
    SEARCH_CYAN_ZONE = auto()
    APPROACH_CYAN_ZONE = auto()
    DECONTAMINATING = auto()

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw', 0.0)
        # Autonomy defaults to on; set to false to keep manual-only.
        self.autonomy_enabled = self.declare_parameter('autonomy_enabled', True).get_parameter_value().bool_value

        self.initial_x = self.get_parameter('x').get_parameter_value().double_value
        self.initial_y = self.get_parameter('y').get_parameter_value().double_value
        self.initial_yaw = self.get_parameter('yaw').get_parameter_value().double_value

        namespace = (self.get_namespace() or '').strip()
        self.robot_id = namespace.lstrip('/') if namespace else 'robot1'

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.last_barrels: list = []
        self.last_zones: list = []
        self.holding_colour: Optional[int] = None
        self.radiation_level: int = 0

        # === TESTING: Teleop mode for manual robot control ===
        # Remove this section when done testing movement.
        self.teleop_twist: Optional[Twist] = None
        self.last_teleop_time = self.get_clock().now()
        self.create_subscription(Twist, 'teleop_twist', self._teleop_cb, 10)
        # === END TESTING ===

        self.create_subscription(BarrelList, 'barrels', self._barrels_cb, 10)
        self.create_subscription(ZoneList, 'zones', self._zones_cb, 10)
        self.create_subscription(BarrelHolders, '/barrel_holders', self._holders_cb, 10)
        self.create_subscription(RadiationList, '/radiation_levels', self._radiation_cb, 10)

        self.pickup_client = self.create_client(ItemRequest, '/pick_up_item')
        self.offload_client = self.create_client(ItemRequest, '/offload_item')
        self.decontaminate_client = self.create_client(ItemRequest, '/decontaminate')

        self.state = ControllerState.SEARCH_BARREL
        self.pending_future = None
        self.pending_action: Optional[ControllerState] = None
        self.last_service_call_time = self.get_clock().now()

        self._align_start_time = None

        # Search / exploration behaviour state
        self._search_start_time = None

        self.first_time = True
        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop)

    def _barrels_cb(self, msg: BarrelList):
        self.last_barrels = list(msg.data)

    def _zones_cb(self, msg: ZoneList):
        self.last_zones = list(msg.data)

    def _holders_cb(self, msg: BarrelHolders):
        holding = None
        for holder in msg.data:
            if holder.robot_id == self.robot_id:
                holding = holder.colour
                break
        self.holding_colour = holding

    def _radiation_cb(self, msg: RadiationList):
        level = 0
        for r in msg.data:
            if r.robot_id == self.robot_id:
                level = int(r.level)
                break
        self.radiation_level = level

    # === TESTING: Teleop callback ===
    # Remove this method when done testing movement.
    def _teleop_cb(self, msg: Twist):
        self.teleop_twist = msg
        self.last_teleop_time = self.get_clock().now()
    # === END TESTING ===

    def _publish_twist(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        try:
            self.cmd_vel_pub.publish(msg)
        except Exception:
            # During shutdown, the underlying rcl context may already be invalid.
            pass

    def _stop(self):
        self._publish_twist(0.0, 0.0)

    def _choose_barrel(self):
        if not self.last_barrels:
            return None
        blue = [b for b in self.last_barrels if int(b.colour) == 1]
        red = [b for b in self.last_barrels if int(b.colour) == 0]
        candidates = blue if blue else red
        return max(candidates, key=lambda b: float(b.size), default=None)

    def _choose_zone(self, zone_type: int):
        zones = [z for z in self.last_zones if int(z.zone) == int(zone_type)]
        if not zones:
            return None
        return max(zones, key=lambda z: float(z.size), default=None)

    def _drive_to_target(self, x_err: float, size: float):
        x_scale = 320.0
        k_ang = 1.2
        k_lin = 0.18

        ang = -k_ang * (float(x_err) / x_scale)
        ang = max(min(ang, 0.8), -0.8)

        # Prefer driving in an arc rather than turning in place forever.
        # When the target is far off-centre we still creep forward so the robot
        # actually explores new space and the vision target can re-enter view.
        forward_scale = max(0.0, 1.0 - min(abs(float(x_err)) / 260.0, 1.0))
        lin = k_lin * forward_scale

        # Add a small minimum speed for distant/small targets.
        if float(size) < 0.18:
            lin = max(lin, 0.06)
        elif float(size) < 0.26:
            lin = max(lin, 0.03)

        if float(size) > 0.35:
            lin *= 0.4

        self._publish_twist(lin, ang)

    def _wander_search(self):
        # Simple exploration: rotate to scan, then drive forward.
        if self._search_start_time is None:
            self._search_start_time = self.get_clock().now()

        elapsed = (self.get_clock().now() - self._search_start_time).nanoseconds / 1e9
        rotate_time = 3.0
        forward_time = 2.5
        cycle = rotate_time + forward_time
        phase = elapsed % cycle

        if phase < rotate_time:
            self._publish_twist(0.0, 0.55)
        else:
            self._publish_twist(0.14, 0.0)

    def _can_call_service(self) -> bool:
        now = self.get_clock().now()
        if (now - self.last_service_call_time).nanoseconds < int(1e9):
            return False
        self.last_service_call_time = now
        return True

    def _call_item_service(self, client, action_state: ControllerState):
        if not self._can_call_service():
            return False
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=0.0)
            return False

        req = ItemRequest.Request()
        req.robot_id = self.robot_id
        self.pending_future = client.call_async(req)
        self.pending_action = action_state
        return True

    def control_loop(self):

        if self.first_time:
            self.get_logger().info(f"Initial pose - x: {self.initial_x}, y: {self.initial_y}, yaw: {self.initial_yaw}. Ready to go.")
            self.get_logger().info(f"Robot id: {self.robot_id}")
            self.first_time = False

        # === TESTING: Check for active teleop input ===
        # Remove this section when done testing movement.
        now = self.get_clock().now()
        teleop_active = (self.teleop_twist is not None and 
                         (now - self.last_teleop_time).nanoseconds < int(0.5e9))  # 0.5s timeout
        
        if teleop_active:
            # Use teleop input instead of autonomous control
            self._publish_twist(self.teleop_twist.linear.x, self.teleop_twist.angular.z)
            return
        if not self.autonomy_enabled:
            # Autonomous mode disabled; hold position unless teleop input is active.
            self._stop()
            return
        # === END TESTING ===

        # Handle in-flight service call
        if self.pending_future is not None:
            if self.pending_future.done():
                try:
                    resp = self.pending_future.result()
                    if resp is not None:
                        if resp.success:
                            self.get_logger().info(resp.message)
                        else:
                            self.get_logger().warn(resp.message)
                except Exception as e:
                    self.get_logger().warn(f"Service call failed: {e}")

                finished_action = self.pending_action
                self.pending_future = None
                self.pending_action = None

                if finished_action == ControllerState.PICKING_UP:
                    # If it didn't attach, we will still see holding_colour=None
                    self.state = ControllerState.SEARCH_GREEN_ZONE if self.holding_colour is not None else ControllerState.SEARCH_BARREL
                elif finished_action == ControllerState.OFFLOADING:
                    # After offload, decide whether to decontaminate
                    if self.radiation_level > 0:
                        self.state = ControllerState.SEARCH_CYAN_ZONE
                    else:
                        self.state = ControllerState.SEARCH_BARREL
                elif finished_action == ControllerState.DECONTAMINATING:
                    self.state = ControllerState.SEARCH_BARREL
            else:
                self._stop()
            return

        # If something is attached, focus on delivery
        if self.holding_colour is not None and self.state in (ControllerState.SEARCH_BARREL, ControllerState.APPROACH_BARREL):
            self.state = ControllerState.SEARCH_GREEN_ZONE

        # --- State machine ---
        if self.state == ControllerState.SEARCH_BARREL:
            target = self._choose_barrel()
            if target is None:
                self._wander_search()
                return
            self._search_start_time = None
            self.state = ControllerState.APPROACH_BARREL

        if self.state == ControllerState.APPROACH_BARREL:
            target = self._choose_barrel()
            if target is None:
                self.state = ControllerState.SEARCH_BARREL
                return

            x_err = float(target.x)
            size = float(target.size)

            if abs(x_err) < 25.0 and size > 0.30:
                # Pickup expects the barrel behind the robot. We drive forward past the
                # barrel, stop briefly, then request pickup.
                if self._align_start_time is None:
                    self._align_start_time = self.get_clock().now()

                elapsed = (self.get_clock().now() - self._align_start_time).nanoseconds / 1e9
                if elapsed < 1.6:
                    self._publish_twist(0.16, 0.0)  # drive forward to put barrel behind
                    return
                if elapsed < 1.9:
                    self._stop()
                    return

                self._align_start_time = None
                self._stop()
                if self._call_item_service(self.pickup_client, ControllerState.PICKING_UP):
                    self.state = ControllerState.PICKING_UP
                return

            self._drive_to_target(x_err, size)
            return

        if self.state == ControllerState.SEARCH_GREEN_ZONE:
            target = self._choose_zone(zone_type=1)  # ZONE_GREEN
            if target is None:
                self._wander_search()
                return
            self._search_start_time = None
            self.state = ControllerState.APPROACH_GREEN_ZONE

        if self.state == ControllerState.APPROACH_GREEN_ZONE:
            target = self._choose_zone(zone_type=1)
            if target is None:
                self.state = ControllerState.SEARCH_GREEN_ZONE
                return

            x_err = float(target.x)
            size = float(target.size)

            if abs(x_err) < 30.0 and size > 0.28:
                # Barrel is behind; drive forward slightly into the zone before offloading.
                if self._align_start_time is None:
                    self._align_start_time = self.get_clock().now()

                elapsed = (self.get_clock().now() - self._align_start_time).nanoseconds / 1e9
                if elapsed < 1.5:
                    self._publish_twist(0.12, 0.0)
                    return

                self._align_start_time = None
                self._stop()
                if self._call_item_service(self.offload_client, ControllerState.OFFLOADING):
                    self.state = ControllerState.OFFLOADING
                return

            self._drive_to_target(x_err, size)
            return

        if self.state == ControllerState.SEARCH_CYAN_ZONE:
            target = self._choose_zone(zone_type=0)  # ZONE_CYAN
            if target is None:
                self._wander_search()
                return
            self._search_start_time = None
            self.state = ControllerState.APPROACH_CYAN_ZONE

        if self.state == ControllerState.APPROACH_CYAN_ZONE:
            target = self._choose_zone(zone_type=0)
            if target is None:
                self.state = ControllerState.SEARCH_CYAN_ZONE
                return

            x_err = float(target.x)
            size = float(target.size)

            if abs(x_err) < 30.0 and size > 0.28:
                self._stop()
                if self._call_item_service(self.decontaminate_client, ControllerState.DECONTAMINATING):
                    self.state = ControllerState.DECONTAMINATING
                return

            self._drive_to_target(x_err, size)
            return


    def destroy_node(self):
        self._stop()
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.ALL)

    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()