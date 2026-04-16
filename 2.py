import pygame
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

"""
Missile Evasion Game with Extended Kalman Filter (EKF) and Advanced Guidance
==============================================================================
A 2D game where missiles use different guidance strategies:
- BASIC: Simple pursuit without EKF
- EKF: Extended Kalman Filter with future prediction
- ADVANCED: EKF + True intercept guidance (solves interception geometry)

Features:
- Multiple missile types with different AI behaviors
- True intercept guidance solving for collision point
- Distance-adaptive measurement noise
- Optimized matrix operations for numerical stability
- Enhanced EKF initialization using first measurement
"""


# =============================================================================
# ENUMS
# =============================================================================

class MissileType(Enum):
    """Different missile guidance types."""
    BASIC = 0      # Simple pursuit
    EKF = 1        # Kalman filter with prediction
    ADVANCED = 2   # EKF + true intercept guidance


# =============================================================================
# CONSTANTS
# =============================================================================

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
YELLOW = (255, 255, 50)
CYAN = (50, 255, 255)
MAGENTA = (255, 50, 255)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
PURPLE = (200, 50, 200)

# Player settings
PLAYER_SIZE = 20
PLAYER_ACCELERATION = 0.5
PLAYER_MAX_SPEED = 5.0
PLAYER_FRICTION = 0.92

# Missile settings
MISSILE_SIZE = 10
MISSILE_BASE_SPEED = 2.0
MISSILE_ACCELERATION = 0.15
MISSILE_MAX_SPEED = 6.0
MISSILE_TURN_RATE = 0.08

# EKF Settings
PROCESS_NOISE_POSITION = 0.5  # Process noise for position
PROCESS_NOISE_VELOCITY = 0.1  # Process noise for velocity
MEASUREMENT_NOISE_RANGE_BASE = 20.0  # Base measurement noise for range (pixels)
MEASUREMENT_NOISE_ANGLE_BASE = 0.1   # Base measurement noise for angle (radians)

# Distance-based noise scaling
NOISE_DISTANCE_SCALE = 1000.0  # Reference distance for noise scaling
NOISE_DISTANCE_MAX = 800.0     # Max distance for noise calculation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_intercept_point(missile_pos: np.ndarray, target_pos: np.ndarray, 
                              target_vel: np.ndarray, missile_speed: float) \
                              -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    Calculate true intercept point where missile can reach target.
    
    Solves: ||target_pos + target_vel * t - missile_pos|| = missile_speed * t
    
    This is the classic "pursuit-evasion" problem. We solve for time t when
    the missile traveling at missile_speed can reach the target.
    
    Args:
        missile_pos: Current missile position [x, y]
        target_pos: Current target position [x, y]  
        target_vel: Target velocity [vx, vy]
        missile_speed: Speed of missile
        
    Returns:
        Tuple of (intercept_point, intercept_time) or (None, None) if no solution
    """
    # Vector from missile to target
    dx = target_pos[0] - missile_pos[0]
    dy = target_pos[1] - missile_pos[1]
    
    # Target velocity components
    tx = target_vel[0]
    ty = target_vel[1]
    
    # Solve quadratic: ||dx + tx*t, dy + ty*t|| = v*t
    # Expanding: (dx + tx*t)^2 + (dy + ty*t)^2 = (v*t)^2
    # (dx^2 + dy^2) + 2t(dx*tx + dy*ty) + t^2(tx^2 + ty^2) = v^2*t^2
    # t^2(tx^2 + ty^2 - v^2) + 2t(dx*tx + dy*ty) + (dx^2 + dy^2) = 0
    
    initial_distance = math.sqrt(dx*dx + dy*dy)
    
    # Coefficients for quadratic equation: a*t^2 + b*t + c = 0
    a = tx*tx + ty*ty - missile_speed*missile_speed
    b = 2.0 * (dx*tx + dy*ty)
    c = dx*dx + dy*dy
    
    # Handle edge cases
    if abs(a) < 1e-6:  # Nearly linear equation
        if abs(b) < 1e-6:
            return None, None
        t = -c / b
    else:
        # Quadratic formula
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            # No real solution - target moving away faster than missile can catch
            return None, None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        
        # Take smallest positive time (earliest intercept)
        valid_times = [t for t in [t1, t2] if t > 0.01]
        if not valid_times:
            return None, None
        
        t = min(valid_times)
    
    # Ensure valid intercept time
    if t < 0.01:
        return None, None
    
    # Cap intercept time to reasonable bounds
    t = min(t, 2.0)  # Don't look more than 2 seconds ahead
    
    # Calculate intercept point
    intercept_x = target_pos[0] + target_vel[0] * t
    intercept_y = target_pos[1] + target_vel[1] * t
    
    return (intercept_x, intercept_y), t


def get_distance_based_noise(distance: float, base_noise: float) -> float:
    """
    Calculate distance-based measurement noise.
    
    - Far away: higher noise (uncertainty increases with distance)
    - Close: lower noise
    
    Args:
        distance: Distance from missile to target
        base_noise: Base noise level
        
    Returns:
        Adjusted noise level
    """
    # Normalize distance to [0, 1]
    normalized_dist = min(distance / NOISE_DISTANCE_MAX, 1.0)
    
    # Scale noise: base * (1 + distance_factor)
    # At close range: 1x noise
    # At far range: 2x noise
    noise_multiplier = 1.0 + normalized_dist
    
    return base_noise * noise_multiplier


# =============================================================================
# EXTENDED KALMAN FILTER CLASS
# =============================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for tracking a 2D moving target.
    
    State vector: [px, py, vx, vy] - position and velocity
    Measurement: [range, bearing] - nonlinear sensor measurements
    
    Key improvements:
    - Deferred initialization: starts in "uninitialized" state
    - First measurement used to initialize position estimate
    - Distance-based adaptive measurement noise
    - Numerically stable matrix operations using np.linalg.solve
    """
    
    def __init__(self, missile_x: float, missile_y: float):
        """
        Initialize EKF in uninitialized state.
        
        Args:
            missile_x: Initial missile x position (for first measurement)
            missile_y: Initial missile y position (for first measurement)
        """
        # State dimension (4: px, py, vx, vy)
        self.state_dim = 4
        
        # Measurement dimension (2: range, bearing)
        self.meas_dim = 2
        
        # Start uninitialized - will be set on first measurement
        self.state = np.zeros(self.state_dim)
        self.initialized = False
        
        # Store missile position for initialization
        self.init_missile_x = missile_x
        self.init_missile_y = missile_y
        
        # State covariance matrix P
        # High uncertainty before initialization
        self.P = np.diag([500.0, 500.0, 50.0, 50.0])
        
        # Process noise covariance matrix Q (diagonal)
        self.Q = np.diag([
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_VELOCITY**2,
            PROCESS_NOISE_VELOCITY**2
        ])
        
        # Measurement noise covariance matrix R (diagonal, will be updated dynamically)
        self.R = np.diag([
            MEASUREMENT_NOISE_RANGE_BASE**2,
            MEASUREMENT_NOISE_ANGLE_BASE**2
        ])
        
        # Time step
        self.dt = 1.0 / FPS
        
        # Track measurement counter
        self.measurement_count = 0
    
    def initialize_from_measurement(self, measurement: np.ndarray, 
                                   missile_x: float, missile_y: float):
        """
        Initialize filter state from first measurement.
        
        Uses polar measurement (range, bearing) to estimate initial 
        Cartesian position. This is much better than starting at origin.
        
        Args:
            measurement: [range, bearing] in missile reference frame
            missile_x: Missile x position
            missile_y: Missile y position
        """
        if self.initialized:
            return
        
        range_meas, bearing_meas = measurement
        
        # Convert polar to Cartesian (in missile reference frame)
        # Then transform to world frame
        dx = range_meas * math.cos(bearing_meas)
        dy = range_meas * math.sin(bearing_meas)
        
        px = missile_x + dx
        py = missile_y + dy
        
        # Initialize state with estimated position, zero velocity
        self.state = np.array([px, py, 0.0, 0.0])
        
        # Reduce initial covariance significantly (we have first measurement)
        self.P = np.diag([100.0, 100.0, 30.0, 30.0])
        
        self.initialized = True
        self.measurement_count = 0
    
    def predict(self, dt: float = None):
        """
        Prediction step: propagate state and covariance forward.
        Uses constant velocity model.
        
        Args:
            dt: Time step in seconds
        """
        if not self.initialized:
            return
        
        if dt is not None:
            self.dt = dt
        
        # State transition matrix F (constant velocity model with time dt)
        # [px_new]   [1  0  dt  0 ] [px]
        # [py_new] = [0  1  0   dt] [py]
        # [vx_new]   [0  0  1   0 ] [vx]
        # [vy_new]   [0  0  0   1 ] [vy]
        
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        # Predict state: x = F @ x
        self.state = F @ self.state
        
        # Predict covariance: P = F @ P @ F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure covariance remains symmetric (numerical stability)
        self.P = (self.P + self.P.T) * 0.5
    
    def measure(self, true_px: float, true_py: float, 
                missile_x: float, missile_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noisy measurement of target position from missile perspective.
        
        Args:
            true_px: True target x position
            true_py: True target y position
            missile_x: Missile x position
            missile_y: Missile y position
            
        Returns:
            Tuple of (noisy_measurement, actual_measurement)
        """
        # Calculate true range and bearing (from missile to target)
        dx = true_px - missile_x
        dy = true_py - missile_y
        true_range = math.sqrt(dx**2 + dy**2)
        true_bearing = math.atan2(dy, dx)
        
        # Get distance-adaptive noise for this measurement
        range_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_RANGE_BASE)
        bearing_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_ANGLE_BASE)
        
        # Add Gaussian noise
        noisy_range = true_range + random.gauss(0, range_noise)
        noisy_bearing = true_bearing + random.gauss(0, bearing_noise)
        
        # Normalize bearing to [-pi, pi]
        noisy_bearing = math.atan2(math.sin(noisy_bearing), math.cos(noisy_bearing))
        
        # Ensure range is positive
        noisy_range = max(1.0, noisy_range)
        
        measurement = np.array([noisy_range, noisy_bearing])
        actual = np.array([true_range, true_bearing])
        
        return measurement, actual
    
    def update(self, measurement: np.ndarray, missile_x: float, missile_y: float):
        """
        Update step: incorporate measurement into state estimate.
        
        Uses numerically stable approach:
        - Replace inv(S) with solve(S, ...) 
        - Ensures covariance symmetry
        
        Args:
            measurement: Noisy measurement [range, bearing]
            missile_x: Missile x position
            missile_y: Missile y position
        """
        if not self.initialized:
            return
        
        # Measurement model h(x) converts state to [range, bearing]
        px, py, vx, vy = self.state
        
        # Predicted range and bearing from current estimate
        dx = px - missile_x
        dy = py - missile_y
        pred_range = math.sqrt(dx**2 + dy**2)
        pred_bearing = math.atan2(dy, dx)
        
        # Predicted measurement
        pred_measurement = np.array([pred_range, pred_bearing])
        
        # Calculate Jacobian H of measurement function (nonlinear measurement model)
        # h(x) = [sqrt(dx^2 + dy^2), atan2(dy, dx)]
        
        if pred_range > 0.001:
            # Partial derivatives of range
            dr_dpx = dx / pred_range
            dr_dpy = dy / pred_range
            
            # Partial derivatives of bearing
            # d(atan2(dy,dx))/dx = -dy / (dx^2 + dy^2)
            # d(atan2(dy,dx))/dy = dx / (dx^2 + dy^2)
            denom = dx**2 + dy**2
            db_dpx = -dy / denom
            db_dpy = dx / denom
        else:
            # Handle edge case - missile very close
            dr_dpx = dr_dpy = db_dpx = db_dpy = 0
        
        H = np.array([
            [dr_dpx, dr_dpy, 0, 0],
            [db_dpx, db_dpy, 0, 0]
        ], dtype=np.float64)
        
        # Innovation (measurement residual)
        y = measurement - pred_measurement
        
        # Normalize bearing innovation to [-pi, pi]
        y[1] = math.atan2(math.sin(y[1]), math.cos(y[1]))
        
        # Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Ensure S is symmetric for numerical stability
        S = (S + S.T) * 0.5
        
        # Kalman gain: K = P @ H^T @ S^(-1)
        # Use solve() for numerical stability instead of inv()
        try:
            # Solve S @ K_T = (P @ H^T)^T, then K = K_T^T
            # This is more stable than K = P @ H^T @ inv(S)
            HT_P_T = (H @ self.P).T  # (P @ H^T)^T
            K_T = np.linalg.solve(S, HT_P_T)  # More stable than S^(-1) @ ...
            K = K_T.T
        except np.linalg.LinAlgError:
            # If solve fails, use pseudo-inverse as fallback
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # Update state: x = x + K @ y
        self.state = self.state + K @ y
        
        # Update covariance: P = (I - K @ H) @ P (Joseph form is more stable)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P
        
        # Ensure covariance stays symmetric and positive definite
        self.P = (self.P + self.P.T) * 0.5
        
        # Add small regularization to maintain positive definiteness
        self.P += np.eye(self.state_dim) * 1e-6
    
    def get_estimated_position(self) -> Tuple[float, float]:
        """Get estimated target position."""
        return self.state[0], self.state[1]
    
    def get_estimated_velocity(self) -> Tuple[float, float]:
        """Get estimated target velocity."""
        return self.state[2], self.state[3]
    
    def get_position_covariance(self) -> np.ndarray:
        """Get 2x2 position covariance matrix (for uncertainty visualization)."""
        return self.P[0:2, 0:2]
    
    def reset(self):
        """Reset filter to uninitialized state."""
        self.state = np.zeros(self.state_dim)
        self.initialized = False
        self.P = np.diag([500.0, 500.0, 50.0, 50.0])
        self.measurement_count = 0


# =============================================================================
# PLAYER CLASS
# =============================================================================

class Player:
    """
    Player entity that can be controlled with keyboard.
    """
    
    def __init__(self, x: float, y: float):
        """Initialize player at given position."""
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.size = PLAYER_SIZE
        self.alive = True
        
        # For trail visualization
        self.position_history: List[Tuple[float, float]] = []
        self.max_history = 50
    
    def handle_input(self, keys: pygame.key.ScancodeWrapper):
        """Process keyboard input for movement."""
        # Apply acceleration based on input
        accel = PLAYER_ACCELERATION
        
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.vy -= accel
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.vy += accel
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.vx -= accel
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.vx += accel
        
        # Apply friction
        self.vx *= PLAYER_FRICTION
        self.vy *= PLAYER_FRICTION
        
        # Clamp to max speed
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER_MAX_SPEED:
            scale = PLAYER_MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale
    
    def update(self, dt: float):
        """Update player position."""
        if not self.alive:
            return
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Keep player on screen
        self.x = max(self.size, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
        
        # Update position history for trail
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def draw(self, screen: pygame.Surface, show_trail: bool = True):
        """Draw player on screen."""
        if not self.alive:
            return
        
        # Draw trail
        if show_trail and len(self.position_history) > 1:
            pygame.draw.lines(screen, GREEN, False, self.position_history, 2)
        
        # Draw player as triangle pointing in velocity direction
        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            angle = math.atan2(self.vy, self.vx)
        else:
            angle = -math.pi / 2
        
        # Triangle vertices
        tip = (
            self.x + self.size * math.cos(angle),
            self.y + self.size * math.sin(angle)
        )
        left = (
            self.x + self.size * 0.8 * math.cos(angle + 2.5),
            self.y + self.size * 0.8 * math.sin(angle + 2.5)
        )
        right = (
            self.x + self.size * 0.8 * math.cos(angle - 2.5),
            self.y + self.size * 0.8 * math.sin(angle - 2.5)
        )
        
        pygame.draw.polygon(screen, GREEN, [tip, left, right])
        
        # Draw velocity vector
        pygame.draw.line(screen, GREEN, (self.x, self.y), 
                        (self.x + self.vx * 10, self.y + self.vy * 10), 2)
    
    def get_position(self) -> Tuple[float, float]:
        """Get player position."""
        return self.x, self.y
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get player velocity."""
        return self.vx, self.vy
    
    def die(self):
        """Mark player as dead."""
        self.alive = False


# =============================================================================
# MISSILE CLASS
# =============================================================================

class Missile:
    """
    Missile with multiple guidance strategies.
    
    Types:
    - BASIC: Simple proportional navigation (no EKF)
    - EKF: Extended Kalman Filter with velocity prediction
    - ADVANCED: EKF + True intercept guidance (solves geometry)
    """
    
    def __init__(self, x: float, y: float, missile_type: MissileType = MissileType.EKF, 
                 speed_multiplier: float = 1.0):
        """
        Initialize missile with specific guidance type.
        
        Args:
            x: Initial x position
            y: Initial y position
            missile_type: Type of guidance (BASIC, EKF, ADVANCED)
            speed_multiplier: Speed scaling factor for difficulty
        """
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.speed = MISSILE_BASE_SPEED * speed_multiplier
        self.size = MISSILE_SIZE
        self.missile_type = missile_type
        
        # Initialize EKF if not using BASIC guidance
        if missile_type != MissileType.BASIC:
            self.ekf = ExtendedKalmanFilter(x, y)
        else:
            self.ekf = None
        
        # For trajectory visualization
        self.position_history: List[Tuple[float, float]] = []
        self.estimated_position_history: List[Tuple[float, float]] = []
        self.intercept_history: List[Tuple[float, float]] = []  # For ADVANCED missiles
        self.max_history = 100
        
        # Time since last measurement (for EKF and ADVANCED)
        self.measurement_interval = 0.1  # seconds
        self.time_since_measurement = 0.0
        
        # Prediction parameters
        self.prediction_time = 0.5  # seconds (for EKF type)
        
        # For BASIC guidance: smoothing / credibility
        self.basic_lead_time = 0.3  # How far ahead to aim
    
    def update(self, dt: float, player: 'Player'):
        """
        Update missile state based on guidance type.
        
        Args:
            dt: Time step
            player: Player object
        """
        if self.missile_type == MissileType.BASIC:
            self._update_basic(dt, player)
        elif self.missile_type == MissileType.EKF:
            self._update_ekf(dt, player)
        elif self.missile_type == MissileType.ADVANCED:
            self._update_advanced(dt, player)
    
    def _update_basic(self, dt: float, player: 'Player'):
        """
        BASIC guidance: Simple pursuit with basic lead.
        Chases player with simple velocity prediction (no EKF).
        """
        player_x, player_y = player.get_position()
        player_vx, player_vy = player.get_velocity()
        
        # Simple lead - aim ahead of player
        target_x = player_x + player_vx * self.basic_lead_time
        target_y = player_y + player_vy * self.basic_lead_time
        
        # Calculate direction to predicted target
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
        # Smooth turning
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0.1:
            current_angle = math.atan2(self.vy, self.vx)
        else:
            current_angle = target_angle
        
        # Normalize angle difference
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Apply turn rate limit
        turn_amount = max(-MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, angle_diff))
        new_angle = current_angle + turn_amount
        
        # Accelerate
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt, MISSILE_MAX_SPEED)
        
        # Update velocity and position
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        self.x += self.vx
        self.y += self.vy
        
        # Track position
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def _update_ekf(self, dt: float, player: 'Player'):
        """
        EKF guidance: Uses Kalman Filter to estimate player state and predict.
        """
        player_x, player_y = player.get_position()
        
        # EKF Prediction step
        self.ekf.predict(dt)
        
        # Sensor measurement at intervals
        self.time_since_measurement += dt
        if self.time_since_measurement >= self.measurement_interval:
            self.time_since_measurement = 0.0
            
            # Generate noisy measurement
            measurement, actual = self.ekf.measure(
                player_x, player_y, self.x, self.y
            )
            
            # Initialize EKF from first measurement if needed
            if not self.ekf.initialized:
                self.ekf.initialize_from_measurement(measurement, self.x, self.y)
            
            # EKF Update step
            self.ekf.update(measurement, self.x, self.y)
        
        # Get estimated player state
        est_px, est_py = self.ekf.get_estimated_position()
        est_vx, est_vy = self.ekf.get_estimated_velocity()
        
        # Predict future player position
        future_px = est_px + est_vx * self.prediction_time
        future_py = est_py + est_vy * self.prediction_time
        
        # Calculate direction to predicted position
        dx = future_px - self.x
        dy = future_py - self.y
        target_angle = math.atan2(dy, dx)
        
        # Smooth turning
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0.1:
            current_angle = math.atan2(self.vy, self.vx)
        else:
            current_angle = target_angle
        
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        turn_amount = max(-MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, angle_diff))
        new_angle = current_angle + turn_amount
        
        # Accelerate
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt, MISSILE_MAX_SPEED)
        
        # Update velocity and position
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        self.x += self.vx
        self.y += self.vy
        
        # Track trajectories
        self.position_history.append((self.x, self.y))
        if self.ekf.initialized:
            self.estimated_position_history.append((est_px, est_py))
        
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        if len(self.estimated_position_history) > self.max_history:
            self.estimated_position_history.pop(0)
    
    def _update_advanced(self, dt: float, player: 'Player'):
        """
        ADVANCED guidance: EKF + True intercept guidance.
        Solves for true intercept point where missile can reach target.
        """
        player_x, player_y = player.get_position()
        
        # EKF Prediction step
        self.ekf.predict(dt)
        
        # Sensor measurement at intervals
        self.time_since_measurement += dt
        if self.time_since_measurement >= self.measurement_interval:
            self.time_since_measurement = 0.0
            
            # Generate noisy measurement
            measurement, actual = self.ekf.measure(
                player_x, player_y, self.x, self.y
            )
            
            # Initialize EKF from first measurement if needed
            if not self.ekf.initialized:
                self.ekf.initialize_from_measurement(measurement, self.x, self.y)
            
            # EKF Update step
            self.ekf.update(measurement, self.x, self.y)
        
        # Get estimated player state (position and velocity)
        est_px, est_py = self.ekf.get_estimated_position()
        est_vx, est_vy = self.ekf.get_estimated_velocity()
        
        # Solve for true intercept point
        missile_pos = np.array([self.x, self.y])
        target_pos = np.array([est_px, est_py])
        target_vel = np.array([est_vx, est_vy])
        
        intercept_result = calculate_intercept_point(
            missile_pos, target_pos, target_vel, self.speed
        )
        
        # Use intercept point if valid, otherwise fall back to prediction
        if intercept_result[0] is not None:
            target_x, target_y = intercept_result[0]
            self.intercept_history.append((target_x, target_y))
        else:
            # Fallback: use simple prediction
            target_x = est_px + est_vx * self.prediction_time
            target_y = est_py + est_vy * self.prediction_time
        
        # Cap intercept history
        if len(self.intercept_history) > self.max_history:
            self.intercept_history.pop(0)
        
        # Calculate direction to intercept point
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
        # Smooth turning
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0.1:
            current_angle = math.atan2(self.vy, self.vx)
        else:
            current_angle = target_angle
        
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        turn_amount = max(-MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, angle_diff))
        new_angle = current_angle + turn_amount
        
        # Accelerate
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt, MISSILE_MAX_SPEED)
        
        # Update velocity and position
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        self.x += self.vx
        self.y += self.vy
        
        # Track trajectories
        self.position_history.append((self.x, self.y))
        if self.ekf.initialized:
            self.estimated_position_history.append((est_px, est_py))
        
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        if len(self.estimated_position_history) > self.max_history:
            self.estimated_position_history.pop(0)
    
    def check_collision(self, player: 'Player') -> bool:
        """Check if missile collides with player."""
        if not player.alive:
            return False
        
        px, py = player.get_position()
        dx = self.x - px
        dy = self.y - py
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance < (self.size + player.size * 0.7)
    
    def draw(self, screen: pygame.Surface, show_ekf_info: bool = True):
        """Draw missile and guidance information."""
        # Draw trajectory
        if len(self.position_history) > 1:
            color = RED if self.missile_type != MissileType.ADVANCED else PURPLE
            pygame.draw.lines(screen, color, False, self.position_history, 2)
        
        # Draw EKF information (for non-BASIC missiles)
        if show_ekf_info and self.ekf is not None and self.ekf.initialized:
            est_x, est_y = self.ekf.get_estimated_position()
            
            # Draw estimated position as cross
            cross_size = 8
            pygame.draw.line(screen, CYAN, 
                           (est_x - cross_size, est_y - cross_size),
                           (est_x + cross_size, est_y + cross_size), 2)
            pygame.draw.line(screen, CYAN,
                           (est_x + cross_size, est_y - cross_size),
                           (est_x - cross_size, est_y + cross_size), 2)
            
            # Draw estimated trajectory
            if len(self.estimated_position_history) > 1:
                pygame.draw.lines(screen, CYAN, False, 
                                self.estimated_position_history, 1)
            
            # Draw covariance ellipse
            cov = self.ekf.get_position_covariance()
            self._draw_covariance_ellipse(screen, est_x, est_y, cov)
        
        # Draw intercept points for ADVANCED missiles
        if show_ekf_info and self.missile_type == MissileType.ADVANCED:
            if len(self.intercept_history) > 1:
                pygame.draw.lines(screen, YELLOW, False, 
                                self.intercept_history, 1)
            if len(self.intercept_history) > 0:
                # Draw circle at predicted intercept
                ic_x, ic_y = self.intercept_history[-1]
                pygame.draw.circle(screen, YELLOW, (int(ic_x), int(ic_y)), 5, 1)
        
        # Draw missile body (color indicates type)
        if self.missile_type == MissileType.BASIC:
            color = RED
        elif self.missile_type == MissileType.EKF:
            color = BLUE
        else:  # ADVANCED
            color = PURPLE
        
        points = [
            (self.x, self.y - self.size),
            (self.x + self.size, self.y),
            (self.x, self.y + self.size),
            (self.x - self.size, self.y)
        ]
        pygame.draw.polygon(screen, color, points)
        
        # Draw velocity vector
        pygame.draw.line(screen, ORANGE, (self.x, self.y),
                        (self.x + self.vx * 15, self.y + self.vy * 15), 2)
    
    def _draw_covariance_ellipse(self, screen: pygame.Surface, 
                                  cx: float, cy: float, cov: np.ndarray):
        """Draw covariance ellipse (1-sigma and 2-sigma uncertainty)."""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
        except np.linalg.LinAlgError:
            return
        
        # Sort by eigenvalue (largest first)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        num_points = 30
        
        for n_std in [1, 2]:
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                
                # Point on ellipse in local coordinates
                local_x = n_std * 2.0 * math.sqrt(max(eigenvalues[0], 0)) * math.cos(angle)
                local_y = n_std * 2.0 * math.sqrt(max(eigenvalues[1], 0)) * math.sin(angle)
                
                # Rotate to world coordinates
                world_x = cx + eigenvectors[0, 0] * local_x + eigenvectors[1, 0] * local_y
                world_y = cy + eigenvectors[0, 1] * local_x + eigenvectors[1, 1] * local_y
                
                points.append((world_x, world_y))
            
            # Draw ellipse
            if len(points) > 2:
                pygame.draw.polygon(screen, CYAN, points, 1)
    
    def get_ekf_error(self, player: 'Player') -> Tuple[float, float]:
        """Calculate estimation error for debug display."""
        if self.ekf is None or not self.ekf.initialized:
            return 0, 0
        
        est_x, est_y = self.ekf.get_estimated_position()
        true_x, true_y = player.get_position()
        return est_x - true_x, est_y - true_y


# =============================================================================
# GAME CLASS
# =============================================================================

class Game:
    """
    Main game class managing game loop and state.
    """
    
    def __init__(self):
        """Initialize game."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Missile Evasion - EKF Demo")
        self.clock = pygame.time.Clock()
        
        self.running = True
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        
        # Create player
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        
        # Missiles list
        self.missiles: List[Missile] = []
        
        # Spawn timer
        self.spawn_interval = 5000  # ms
        self.last_spawn_time = pygame.time.get_ticks()
        
        # Difficulty scaling
        self.difficulty_timer = 0
        self.speed_increase_interval = 10000  # ms
        self.speed_multiplier = 1.0
        
        # Debug info
        self.show_debug = True
    
    def spawn_missile(self):
        """Spawn a new missile with random type distributed across difficulty."""
        # Choose random edge
        edge = random.randint(0, 3)
        
        if edge == 0:  # Top
            x = random.randint(0, SCREEN_WIDTH)
            y = -50
        elif edge == 1:  # Right
            x = SCREEN_WIDTH + 50
            y = random.randint(0, SCREEN_HEIGHT)
        elif edge == 2:  # Bottom
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT + 50
        else:  # Left
            x = -50
            y = random.randint(0, SCREEN_HEIGHT)
        
        # Choose missile type based on difficulty
        # Early game: mostly BASIC and EKF
        # Mid game: mix of all types
        # Late game: more ADVANCED missiles
        difficulty_factor = min(self.speed_multiplier / 3.0, 1.0)
        
        rand = random.random()
        if rand < 0.3:
            missile_type = MissileType.BASIC
        elif rand < 0.6 + difficulty_factor * 0.3:
            missile_type = MissileType.EKF
        else:
            missile_type = MissileType.ADVANCED
        
        missile = Missile(x, y, missile_type, self.speed_multiplier)
        self.missiles.append(missile)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r and self.game_over:
                    self.restart()
                elif event.key == pygame.K_SPACE and self.game_over:
                    self.restart()
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
    
    def update(self):
        """Update game state."""
        if self.game_over:
            return
        
        dt = self.clock.get_time() / 1000.0  # Convert to seconds
        
        # Handle player input
        keys = pygame.key.get_pressed()
        self.player.handle_input(keys)
        
        # Update player
        self.player.update(dt)
        
        # Spawn missiles
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_missile()
            self.last_spawn_time = current_time
        
        # Increase difficulty over time
        if current_time - self.difficulty_timer > self.speed_increase_interval:
            self.speed_multiplier += 0.1
            self.difficulty_timer = current_time
            # Also decrease spawn interval slightly
            self.spawn_interval = max(2000, self.spawn_interval - 200)
        
        # Update missiles
        for missile in self.missiles:
            missile.update(dt, self.player)
            
            # Check collision
            if missile.check_collision(self.player):
                self.player.die()
                self.game_over = True
        
        # Update score (survival time)
        self.score = (pygame.time.get_ticks() - self.start_time) // 100
    
    def draw(self):
        """Draw game to screen."""
        self.screen.fill(BLACK)
        
        # Draw grid (optional background)
        self._draw_grid()
        
        # Draw player
        self.player.draw(self.screen)
        
        # Draw missiles
        for missile in self.missiles:
            missile.draw(self.screen, self.show_debug)
        
        # Draw UI
        self._draw_ui()
        
        # Draw game over screen
        if self.game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw background grid."""
        grid_size = 50
        for x in range(0, SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (0, y), (SCREEN_WIDTH, y))
    
    def _draw_ui(self):
        """Draw UI elements with missile type info."""
        font = pygame.font.Font(None, 24)
        large_font = pygame.font.Font(None, 36)
        
        # Score
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Difficulty
        diff_text = font.render(f"Speed: {self.speed_multiplier:.1f}x", True, WHITE)
        self.screen.blit(diff_text, (10, 35))
        
        # Missile count with breakdown by type
        total_missiles = len(self.missiles)
        basic_count = sum(1 for m in self.missiles if m.missile_type == MissileType.BASIC)
        ekf_count = sum(1 for m in self.missiles if m.missile_type == MissileType.EKF)
        adv_count = sum(1 for m in self.missiles if m.missile_type == MissileType.ADVANCED)
        
        missile_text = font.render(f"Missiles: {total_missiles} (B:{basic_count} E:{ekf_count} A:{adv_count})", 
                                  True, WHITE)
        self.screen.blit(missile_text, (10, 60))
        
        # Missile type legend
        legend_y = 95
        pygame.draw.polygon(self.screen, RED, [
            (10, legend_y - 7),
            (17, legend_y),
            (10, legend_y + 7),
            (3, legend_y)
        ])
        legend_text = font.render("BASIC (Pursuit)", True, RED)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        legend_y += 25
        pygame.draw.polygon(self.screen, BLUE, [
            (10, legend_y - 7),
            (17, legend_y),
            (10, legend_y + 7),
            (3, legend_y)
        ])
        legend_text = font.render("EKF (Prediction)", True, BLUE)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        legend_y += 25
        pygame.draw.polygon(self.screen, PURPLE, [
            (10, legend_y - 7),
            (17, legend_y),
            (10, legend_y + 7),
            (3, legend_y)
        ])
        legend_text = font.render("ADVANCED (Intercept)", True, PURPLE)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        # Debug info
        if self.show_debug and not self.game_over and len(self.missiles) > 0:
            missile = self.missiles[0]
            error_x, error_y = missile.get_ekf_error(self.player)
            error_dist = math.sqrt(error_x**2 + error_y**2)
            
            debug_y = 265
            debug_lines = [
                f"First Missile Type: {missile.missile_type.name}",
                f"EKF Error X: {error_x:.1f}",
                f"EKF Error Y: {error_y:.1f}",
                f"EKF Error Dist: {error_dist:.1f}",
                f"Player Vel: ({self.player.vx:.1f}, {self.player.vy:.1f})",
            ]
            
            if missile.ekf is not None and missile.ekf.initialized:
                debug_lines.append(f"Est Vel: ({missile.ekf.state[2]:.1f}, {missile.ekf.state[3]:.1f})")
            
            for line in debug_lines:
                text = font.render(line, True, CYAN)
                self.screen.blit(text, (10, debug_y))
                debug_y += 25
        
        # Controls help
        help_text = font.render("WASD/Arrows: Move | D: Toggle Debug | ESC: Quit", 
                               True, GRAY)
        self.screen.blit(help_text, (10, SCREEN_HEIGHT - 30))
    
    def _draw_game_over(self):
        """Draw game over screen."""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        large_font = pygame.font.Font(None, 72)
        font = pygame.font.Font(None, 36)
        
        game_over_text = large_font.render("GAME OVER", True, RED)
        score_text = font.render(f"Final Score: {self.score}", True, WHITE)
        restart_text = font.render("Press SPACE or R to restart", True, YELLOW)
        
        # Center text
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
        self.screen.blit(game_over_text, text_rect)
        
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10))
        self.screen.blit(score_text, score_rect)
        
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60))
        self.screen.blit(restart_text, restart_rect)
    
    def restart(self):
        """Restart the game."""
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.missiles = []
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        self.last_spawn_time = pygame.time.get_ticks()
        self.difficulty_timer = pygame.time.get_ticks()
        self.speed_multiplier = 1.0
        self.spawn_interval = 5000
    
    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    game = Game()
    game.run()