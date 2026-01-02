#!/usr/bin/env python3
"""
Visual Servoing Module for AURO 2025

This module handles precise visual servoing for barrel approach and pickup.
The key challenge is that the barrel must be BEHIND the robot for pickup
(within 0.45m and ±15° of directly behind).

Strategy:
1. Approach barrel head-on using camera
2. When close enough, execute a turn-and-reverse maneuver
3. Back up until barrel is within pickup range behind robot
"""

import math
import numpy as np
from typing import Tuple, Optional
from enum import Enum, auto


class ServoPhase(Enum):
    """Visual servoing phases for barrel pickup"""
    APPROACH = auto()       # Moving toward barrel while facing it
    TURNING = auto()        # Turning away from barrel
    REVERSING = auto()      # Backing up toward barrel
    ALIGNED = auto()        # Ready for pickup


class VisualServo:
    """
    Visual servoing controller for barrel approach and pickup alignment.
    
    The pickup geometry requires the barrel to be:
    - Within 0.45m of the robot
    - Within ±15° of directly BEHIND the robot
    
    Since the camera faces forward, we must:
    1. Approach the barrel until close (but not too close)
    2. Turn 180° to put the barrel behind us
    3. Fine-tune position by backing up
    """
    
    # Pickup constraints
    PICKUP_MAX_DISTANCE = 0.45  # meters
    PICKUP_MAX_ANGLE = 15.0     # degrees from directly behind
    
    # Approach parameters
    APPROACH_STOP_DISTANCE = 0.6  # Stop approaching when this close
    APPROACH_LINEAR_KP = 0.3
    APPROACH_ANGULAR_KP = 1.0
    
    # Turn parameters
    TURN_ANGULAR_SPEED = 0.4
    
    # Reverse parameters  
    REVERSE_SPEED = 0.1
    REVERSE_ANGULAR_KP = 0.5
    
    # Camera calibration (adjust based on actual camera)
    # Maps barrel apparent diameter to distance
    SIZE_TO_DISTANCE_K = 0.15  # Calibration constant
    
    def __init__(self):
        self.phase = ServoPhase.APPROACH
        self.turn_start_yaw: Optional[float] = None
        self.turn_target_yaw: Optional[float] = None
        self.last_barrel_direction = 0.0  # -1 for left, +1 for right
        
    def reset(self):
        """Reset servo state for new target"""
        self.phase = ServoPhase.APPROACH
        self.turn_start_yaw = None
        self.turn_target_yaw = None
        self.last_barrel_direction = 0.0
        
    def compute_control(
        self,
        barrel_x: float,          # Barrel x offset in image (normalized, -1 to 1)
        barrel_y: float,          # Barrel y offset in image (normalized)
        barrel_size: float,       # Barrel apparent size (larger = closer)
        robot_yaw: float,         # Current robot yaw (radians)
        barrel_visible: bool      # Is barrel currently visible?
    ) -> Tuple[float, float, bool]:
        """
        Compute velocity commands for visual servoing.
        
        Returns:
            (linear_vel, angular_vel, pickup_ready)
        """
        if self.phase == ServoPhase.APPROACH:
            return self._approach_control(barrel_x, barrel_y, barrel_size, 
                                         robot_yaw, barrel_visible)
        elif self.phase == ServoPhase.TURNING:
            return self._turn_control(robot_yaw)
        elif self.phase == ServoPhase.REVERSING:
            return self._reverse_control(barrel_visible, robot_yaw)
        else:  # ALIGNED
            return (0.0, 0.0, True)
            
    def _approach_control(
        self, 
        barrel_x: float,
        barrel_y: float,
        barrel_size: float,
        robot_yaw: float,
        barrel_visible: bool
    ) -> Tuple[float, float, bool]:
        """
        Approach phase: Drive toward barrel while keeping it centered.
        """
        if not barrel_visible:
            # Lost barrel - rotate to search
            return (0.0, 0.3, False)
            
        # Estimate distance from barrel size
        estimated_distance = self._estimate_distance(barrel_size)
        
        # Remember which side the barrel is on (for turn direction)
        if abs(barrel_x) > 0.1:
            self.last_barrel_direction = 1.0 if barrel_x > 0 else -1.0
            
        # Angular: keep barrel centered
        angular_vel = -self.APPROACH_ANGULAR_KP * barrel_x
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        
        # Linear: approach until close enough
        if estimated_distance > self.APPROACH_STOP_DISTANCE:
            linear_vel = self.APPROACH_LINEAR_KP * min(estimated_distance, 1.0)
            linear_vel = np.clip(linear_vel, 0.05, 0.2)
        else:
            # Close enough - start turn maneuver
            self.phase = ServoPhase.TURNING
            self.turn_start_yaw = robot_yaw
            # Turn direction: turn AWAY from barrel (so it ends up behind us)
            turn_direction = -self.last_barrel_direction  # Turn opposite to barrel
            self.turn_target_yaw = self._normalize_angle(
                robot_yaw + turn_direction * math.pi
            )
            return (0.0, 0.0, False)
            
        return (linear_vel, angular_vel, False)
        
    def _turn_control(self, robot_yaw: float) -> Tuple[float, float, bool]:
        """
        Turn phase: Rotate 180° to put barrel behind us.
        """
        if self.turn_target_yaw is None:
            self.phase = ServoPhase.APPROACH
            return (0.0, 0.0, False)
            
        # Calculate angle remaining
        angle_diff = self._normalize_angle(self.turn_target_yaw - robot_yaw)
        
        if abs(angle_diff) < 0.1:  # Within ~6 degrees
            # Turn complete - start reversing
            self.phase = ServoPhase.REVERSING
            return (0.0, 0.0, False)
            
        # Continue turning
        turn_speed = self.TURN_ANGULAR_SPEED * np.sign(angle_diff)
        return (0.0, turn_speed, False)
        
    def _reverse_control(
        self,
        barrel_visible: bool,
        robot_yaw: float
    ) -> Tuple[float, float, bool]:
        """
        Reverse phase: Back up toward the barrel (now behind us).
        
        At this point we can't see the barrel with the front camera,
        so we rely on our position estimate and back up carefully.
        """
        # Since barrel is behind us, we can't see it with front camera
        # Back up slowly for a fixed distance/time
        
        # For safety, just back up slowly
        linear_vel = -self.REVERSE_SPEED
        angular_vel = 0.0
        
        # After backing up sufficiently, we should be in position
        # This is a simplification - in practice you'd use other sensors
        # or track position more carefully
        
        # Signal ready for pickup attempt after a brief reverse
        # The actual success will be determined by the pickup service
        
        return (linear_vel, angular_vel, False)
        
    def signal_pickup_ready(self):
        """Call this after sufficient reversing to signal pickup attempt"""
        self.phase = ServoPhase.ALIGNED
        
    def _estimate_distance(self, barrel_size: float) -> float:
        """
        Estimate distance to barrel from its apparent size.
        
        This is approximate and should be calibrated for your camera.
        Larger apparent size = closer barrel.
        """
        if barrel_size <= 0:
            return float('inf')
            
        # Inverse relationship with some minimum
        # Tune SIZE_TO_DISTANCE_K based on actual camera characteristics
        return max(0.3, self.SIZE_TO_DISTANCE_K / barrel_size)
        
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class BarrelTracker:
    """
    Track barrel positions relative to robot using visual detections.
    
    Since the camera only faces forward, we need to estimate where
    barrels are when they're not visible (during turn/reverse maneuvers).
    """
    
    def __init__(self):
        self.tracked_barrels = {}  # id -> (rel_x, rel_y, color, last_seen_time)
        self.next_id = 0
        
    def update_from_detections(
        self,
        detections: list,  # List of Item messages
        robot_pose: Tuple[float, float, float],  # x, y, yaw
        current_time: float
    ):
        """Update tracked barrels from new visual detections"""
        # Convert image-space detections to robot-relative positions
        for det in detections:
            # Estimate position relative to robot
            # This is approximate - would need proper camera calibration
            distance = self._estimate_distance(det.diameter)
            angle = -det.x * 0.5  # Approximate angle from x offset
            
            rel_x = distance * math.cos(angle)
            rel_y = distance * math.sin(angle)
            
            # Try to match with existing tracked barrel
            matched = self._match_detection(rel_x, rel_y, det.colour)
            
            if matched is not None:
                self.tracked_barrels[matched] = (rel_x, rel_y, det.colour, current_time)
            else:
                # New barrel
                self.tracked_barrels[self.next_id] = (rel_x, rel_y, det.colour, current_time)
                self.next_id += 1
                
    def get_barrel_behind_robot(
        self,
        robot_pose: Tuple[float, float, float],
        max_distance: float = 0.5,
        max_angle_deg: float = 20.0
    ) -> Optional[Tuple[float, float, str]]:
        """
        Check if there's a barrel behind the robot within pickup range.
        
        Returns (rel_x, rel_y, color) if found, None otherwise.
        """
        for barrel_id, (rel_x, rel_y, color, _) in self.tracked_barrels.items():
            # Check if barrel is behind robot (negative x in robot frame)
            if rel_x > 0:
                continue  # In front
                
            distance = math.sqrt(rel_x**2 + rel_y**2)
            angle = math.degrees(math.atan2(abs(rel_y), abs(rel_x)))
            
            if distance <= max_distance and angle <= max_angle_deg:
                return (rel_x, rel_y, color)
                
        return None
        
    def _match_detection(self, rel_x: float, rel_y: float, color: str) -> Optional[int]:
        """Try to match a detection with an existing tracked barrel"""
        for barrel_id, (bx, by, bcolor, _) in self.tracked_barrels.items():
            if bcolor == color:
                dist = math.sqrt((rel_x - bx)**2 + (rel_y - by)**2)
                if dist < 0.5:  # Within 0.5m - likely same barrel
                    return barrel_id
        return None
        
    def _estimate_distance(self, barrel_size: float) -> float:
        """Estimate distance from barrel apparent size"""
        if barrel_size <= 0:
            return float('inf')
        return max(0.3, 0.15 / barrel_size)
