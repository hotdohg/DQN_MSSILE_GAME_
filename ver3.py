import pygame
import numpy as np
import math
import random
import sys
from collections import deque
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim

"""
Advanced Real-Time DQN Missile Evasion Game
============================================
Hybrid system combining:
- Extended Kalman Filter (EKF) for state estimation
- True intercept guidance (baseline physics)
- Deep Q-Network (DQN) for learned behavior modulation
- Online training during gameplay (reinforcement learning)

Missiles learn in real-time how to better pursue the human player.
=====================================================================
3D Missile evasion game featuring:
- Three missile types: BASIC (pursuit), EKF (estimation), ADVANCED (intercept)
- Extended Kalman Filter for state estimation
- True intercept guidance with pursuit-evasion geometry
- Realistic fuel system with thrust multiplier
- Anti-orbiting fixes (closing velocity, dynamic turn rate)
- Distance-adaptive measurement noise
- Numerically stable matrix operations
"""


# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
YELLOW = (255, 255, 50)
CYAN = (50, 255, 255)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
PURPLE = (200, 50, 200)

# Player settings (Aircraft-like movement)
PLAYER_SIZE = 12
PLAYER_SPEED = 4.0  # Constant forward speed
PLAYER_TURN_RATE = 0.06  # Radians per frame
PLAYER_MAX_TURN_RATE = 0.12  # Maximum turn rate
PLAYER_INERTIA = 0.95  # Optional: slight smoothing for realism

# Missile settings
MISSILE_SIZE = 5
MISSILE_BASE_SPEED = 2.0
MISSILE_ACCELERATION = 0.15
MISSILE_MAX_SPEED = 6.0
MISSILE_TURN_RATE = 0.08

# Fuel system
MISSILE_MAX_FUEL = 12.0
MISSILE_BURN_TIME = 6.0
MISSILE_MIN_SPEED = 1.0

# EKF Settings
PROCESS_NOISE_POSITION = 0.5
PROCESS_NOISE_VELOCITY = 0.1
MEASUREMENT_NOISE_RANGE_BASE = 20.0
MEASUREMENT_NOISE_ANGLE_BASE = 0.1
NOISE_DISTANCE_MAX = 800.0

# Anti-orbiting
CLOSING_VELOCITY_THRESHOLD = -0.1
INTERCEPT_TIME_MAX = 1.5
CLOSE_RANGE_THRESHOLD = 80.0

# DQN Settings
DQN_STATE_SIZE = 9
DQN_ACTION_SIZE = 3  # turn_left, straight, turn_right
DQN_HIDDEN_SIZE = 128
REPLAY_BUFFER_SIZE = 10000
DQN_BATCH_SIZE = 64
DQN_LEARNING_RATE = 1e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY = 0.995
DQN_UPDATE_FREQ = 4  # train every N frames
DQN_TARGET_UPDATE_FREQ = 1000  # update target network every N frames

# Game settings
MAX_MISSILES = 20
MISSILE_SPAWN_INTERVAL = 3000  # ms

# Device settings (CPU or GPU)
# Options: 'auto' (auto-detect), 'cpu', 'cuda', 'gpu'
DEVICE_PREFERENCE = 'auto'  # <-- Change this to 'cpu' or 'cuda' to force a specific device


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_intercept_point(missile_pos: np.ndarray, target_pos: np.ndarray, 
                              target_vel: np.ndarray, missile_speed: float):
    """Calculate true intercept point using pursuit-evasion geometry."""
    dx = target_pos[0] - missile_pos[0]
    dy = target_pos[1] - missile_pos[1]
    
    tx = target_vel[0]
    ty = target_vel[1]
    
    a = tx*tx + ty*ty - missile_speed*missile_speed
    b = 2.0 * (dx*tx + dy*ty)
    c = dx*dx + dy*dy
    
    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return None, None
        t = -c / b
    else:
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None, None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        
        valid_times = [t for t in [t1, t2] if t > 0.01]
        if not valid_times:
            return None, None
        
        t = min(valid_times)
    
    if t < 0.01:
        return None, None
    
    t = min(t, INTERCEPT_TIME_MAX)
    
    intercept_x = target_pos[0] + target_vel[0] * t
    intercept_y = target_pos[1] + target_vel[1] * t
    
    return (intercept_x, intercept_y), t


def get_distance_based_noise(distance: float, base_noise: float) -> float:
    """Distance-adaptive measurement noise."""
    normalized_dist = min(distance / NOISE_DISTANCE_MAX, 1.0)
    noise_multiplier = 1.0 + normalized_dist
    return base_noise * noise_multiplier


def get_thrust_multiplier(fuel: float, max_fuel: float, age: float, burn_time: float) -> float:
    """Calculate thrust multiplier from fuel."""
    if age < burn_time:
        return 1.0
    fuel_ratio = max(0, fuel / max_fuel)
    return max(0, fuel_ratio ** 1.0)


# =============================================================================
# NEURAL NETWORK & DQN AGENT
# =============================================================================

class DQNNetwork(nn.Module):
    """Deep Q-Network for missile guidance adjustment."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer:
    """Experience replay buffer for training."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer."""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for missile guidance."""
    
    def __init__(self, state_size: int, action_size: int, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size, DQN_HIDDEN_SIZE).to(device)
        self.target_network = DQNNetwork(state_size, action_size, DQN_HIDDEN_SIZE).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=DQN_LEARNING_RATE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Epsilon-greedy
        self.epsilon = DQN_EPSILON_START
        self.total_steps = 0
        self.training_steps = 0
        
        # Stats
        self.avg_reward = 0
        self.hit_count = 0
        self.total_missiles = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax(dim=1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train on batch from replay buffer."""
        if len(self.replay_buffer) < DQN_BATCH_SIZE:
            return
        
        batch = self.replay_buffer.sample(DQN_BATCH_SIZE)
        if batch is None:
            return
        
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(dim=1)[0]
            target_q_values = rewards_t + DQN_GAMMA * next_q_values * (1 - dones_t)
        
        # Loss and optimization
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # Update target network
        if self.training_steps % DQN_TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(DQN_EPSILON_END, self.epsilon * DQN_EPSILON_DECAY)
    
    def update_stats(self, reward: float, hit: bool):
        """Update agent statistics."""
        self.avg_reward = 0.95 * self.avg_reward + 0.05 * reward
        if hit:
            self.hit_count += 1
        self.total_missiles += 1


# =============================================================================
# EXTENDED KALMAN FILTER
# =============================================================================

class ExtendedKalmanFilter:
    """EKF for missile state estimation."""
    
    def __init__(self, missile_x: float, missile_y: float):
        self.state_dim = 4
        self.meas_dim = 2
        
        self.state = np.zeros(self.state_dim)
        self.initialized = False
        
        self.P = np.diag([500.0, 500.0, 50.0, 50.0])
        self.Q = np.diag([
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_VELOCITY**2,
            PROCESS_NOISE_VELOCITY**2
        ])
        self.R = np.diag([
            MEASUREMENT_NOISE_RANGE_BASE**2,
            MEASUREMENT_NOISE_ANGLE_BASE**2
        ])
        
        self.dt = 1.0 / FPS
    
    def initialize_from_measurement(self, measurement: np.ndarray, 
                                   missile_x: float, missile_y: float):
        """Initialize from first measurement."""
        if self.initialized:
            return
        
        range_meas, bearing_meas = measurement
        dx = range_meas * math.cos(bearing_meas)
        dy = range_meas * math.sin(bearing_meas)
        
        px = missile_x + dx
        py = missile_y + dy
        
        self.state = np.array([px, py, 0.0, 0.0])
        self.P = np.diag([100.0, 100.0, 30.0, 30.0])
        self.initialized = True
    
    def predict(self, dt: float = None):
        """Prediction step."""
        if not self.initialized:
            return
        
        if dt is not None:
            self.dt = dt
        
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) * 0.5
    
    def measure(self, true_px: float, true_py: float, 
                missile_x: float, missile_y: float):
        """Generate noisy measurement."""
        dx = true_px - missile_x
        dy = true_py - missile_y
        true_range = math.sqrt(dx**2 + dy**2)
        true_bearing = math.atan2(dy, dx)
        
        range_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_RANGE_BASE)
        bearing_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_ANGLE_BASE)
        
        noisy_range = true_range + random.gauss(0, range_noise)
        noisy_bearing = true_bearing + random.gauss(0, bearing_noise)
        noisy_bearing = math.atan2(math.sin(noisy_bearing), math.cos(noisy_bearing))
        noisy_range = max(1.0, noisy_range)
        
        measurement = np.array([noisy_range, noisy_bearing])
        actual = np.array([true_range, true_bearing])
        
        return measurement, actual
    
    def update(self, measurement: np.ndarray, missile_x: float, missile_y: float):
        """Update step with numerically stable operations."""
        if not self.initialized:
            return
        
        px, py, vx, vy = self.state
        dx = px - missile_x
        dy = py - missile_y
        pred_range = math.sqrt(dx**2 + dy**2)
        pred_bearing = math.atan2(dy, dx)
        
        pred_measurement = np.array([pred_range, pred_bearing])
        
        if pred_range > 0.001:
            dr_dpx = dx / pred_range
            dr_dpy = dy / pred_range
            denom = dx**2 + dy**2
            db_dpx = -dy / denom
            db_dpy = dx / denom
        else:
            dr_dpx = dr_dpy = db_dpx = db_dpy = 0
        
        H = np.array([
            [dr_dpx, dr_dpy, 0, 0],
            [db_dpx, db_dpy, 0, 0]
        ], dtype=np.float64)
        
        y = measurement - pred_measurement
        y[1] = math.atan2(math.sin(y[1]), math.cos(y[1]))
        
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) * 0.5
        
        try:
            K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        self.state = self.state + K @ y
        
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P
        self.P = (self.P + self.P.T) * 0.5
        self.P += np.eye(self.state_dim) * 1e-6
    
    def get_estimated_position(self):
        return self.state[0], self.state[1]
    
    def get_estimated_velocity(self):
        return self.state[2], self.state[3]
    
    def get_error_estimate(self) -> float:
        """Return estimate of position uncertainty."""
        return (self.P[0, 0] + self.P[1, 1]) ** 0.5


# =============================================================================
# PLAYER CLASS
# =============================================================================

class Player:
    """Human-controlled player (aircraft-like movement)."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = PLAYER_SPEED  # Always moving forward initially
        self.vy = 0.0
        self.size = PLAYER_SIZE
        self.alive = True
        
        # Aircraft state
        self.heading = 0.0  # Heading angle in radians (0 = right, π/2 = down)
        self.target_heading = 0.0  # For smooth turning
        self.angular_velocity = 0.0  # For inertia
        
        self.position_history = []
        self.max_history = 50
    
    def handle_input(self, keys):
        """Process keyboard input for aircraft control."""
        turn_input = 0.0
        
        # Left/right turning only
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            turn_input -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            turn_input += 1.0
        
        # Apply turn rate
        if turn_input != 0:
            self.target_heading += turn_input * PLAYER_TURN_RATE
            # Normalize angle to [-π, π]
            self.target_heading = math.atan2(math.sin(self.target_heading), 
                                           math.cos(self.target_heading))
    
    def update(self, dt: float):
        """Update player position with aircraft physics."""
        if not self.alive:
            return
        
        # Smooth heading change (optional inertia)
        heading_diff = self.target_heading - self.heading
        # Normalize difference to [-π, π]
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi
        
        # Apply angular acceleration (inertia)
        angular_accel = heading_diff * 0.1  # Smooth turning
        self.angular_velocity += angular_accel
        self.angular_velocity *= PLAYER_INERTIA  # Dampen
        
        # Limit turn rate
        self.angular_velocity = max(-PLAYER_MAX_TURN_RATE, 
                                  min(PLAYER_MAX_TURN_RATE, self.angular_velocity))
        
        # Update heading
        self.heading += self.angular_velocity
        
        # Normalize heading
        self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
        
        # Update velocity based on heading (constant speed forward)
        self.vx = math.cos(self.heading) * PLAYER_SPEED
        self.vy = math.sin(self.heading) * PLAYER_SPEED
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Screen wrapping (arcade style)
        if self.x < 0:
            self.x = SCREEN_WIDTH
        elif self.x > SCREEN_WIDTH:
            self.x = 0
            
        if self.y < 0:
            self.y = SCREEN_HEIGHT
        elif self.y > SCREEN_HEIGHT:
            self.y = 0
        
        # Track position history
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def draw(self, screen: pygame.Surface):
        """Draw player aircraft."""
        if not self.alive:
            return
        
        # Draw trail
        if len(self.position_history) > 1:
            pygame.draw.lines(screen, GREEN, False, self.position_history, 2)
        
        # Draw aircraft as rotated triangle
        # Points: nose, left wing, right wing
        nose_x = self.x + math.cos(self.heading) * self.size
        nose_y = self.y + math.sin(self.heading) * self.size
        
        # Wings perpendicular to heading
        wing_angle = self.heading + math.pi / 2
        left_wing_x = self.x + math.cos(wing_angle) * self.size * 0.7
        left_wing_y = self.y + math.sin(wing_angle) * self.size * 0.7
        
        wing_angle = self.heading - math.pi / 2
        right_wing_x = self.x + math.cos(wing_angle) * self.size * 0.7
        right_wing_y = self.y + math.sin(wing_angle) * self.size * 0.7
        
        # Draw aircraft
        pygame.draw.polygon(screen, GREEN, [
            (nose_x, nose_y),
            (left_wing_x, left_wing_y),
            (right_wing_x, right_wing_y)
        ])
        
        # Draw direction indicator (optional)
        indicator_length = self.size * 1.5
        indicator_x = self.x + math.cos(self.heading) * indicator_length
        indicator_y = self.y + math.sin(self.heading) * indicator_length
        pygame.draw.line(screen, YELLOW, (self.x, self.y), 
                        (indicator_x, indicator_y), 2)
    
    def get_position(self):
        return self.x, self.y
    
    def get_velocity(self):
        return self.vx, self.vy
    
    def die(self):
        self.alive = False


# =============================================================================
# MISSILE CLASS (WITH DQN)
# =============================================================================

class Missile:
    """DQN-controlled missile with physics baseline."""
    
    def __init__(self, x: float, y: float, dqn_agent: DQNAgent, speed_multiplier: float = 1.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.speed = MISSILE_BASE_SPEED * speed_multiplier
        self.speed_multiplier = speed_multiplier
        self.size = MISSILE_SIZE
        
        # Fuel system
        self.max_fuel = MISSILE_MAX_FUEL
        self.fuel = self.max_fuel
        self.age = 0.0
        self.burn_time = MISSILE_BURN_TIME
        
        # EKF
        self.ekf = ExtendedKalmanFilter(x, y)
        
        # DQN
        self.dqn_agent = dqn_agent
        self.prev_state = None
        self.prev_action = None
        self.prev_distance = None
        self.episode_reward = 0.0
        
        # Measurement
        self.measurement_interval = 0.1
        self.time_since_measurement = 0.0
        
        # Guidance parameters
        self.prediction_time = 0.5
        self.heading_offset = 0.0  # Modified by DQN
        self.turn_rate_multiplier = 1.0  # Modified by DQN
        
        # History
        self.position_history = []
        self.max_history = 50
    
    def _get_state(self, player: Player) -> np.ndarray:
        """Create normalized state vector for DQN."""
        px, py = player.get_position()
        pvx, pvy = player.get_velocity()
        
        # Relative position and velocity
        dx = px - self.x
        dy = py -self.y
        dvx = pvx - self.vx
        dvy = pvy - self.vy
        
        distance = math.sqrt(dx**2 + dy**2)
        
        # Angle to target
        if distance > 0:
            angle_to_target = math.atan2(dy, dx)
        else:
            angle_to_target = 0
        
        # Current speed ratio
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        speed_ratio = current_speed / MISSILE_MAX_SPEED if MISSILE_MAX_SPEED > 0 else 0
        
        # Fuel ratio
        fuel_ratio = self.fuel / self.max_fuel
        
        # EKF error estimate
        ekf_error = self.ekf.get_error_estimate() if self.ekf.initialized else 100.0
        
        # Normalize state
        state = np.array([
            dx / 1000.0,
            dy / 1000.0,
            dvx / 10.0,
            dvy / 10.0,
            distance / 1000.0,
            angle_to_target / math.pi,
            speed_ratio,
            fuel_ratio,
            ekf_error / 100.0
        ], dtype=np.float32)
        
        return np.clip(state, -1.0, 1.0)
    
    def update(self, dt: float, player: Player) -> bool:
        """Update missile. Return True if should be removed."""
        # Update age and fuel
        self.age += dt
        self.fuel -= dt
        
        # EKF prediction
        self.ekf.predict(dt)
        
        # Get player info
        player_x, player_y = player.get_position()
        player_vx, player_vy = player.get_velocity()
        
        # Measurement
        self.time_since_measurement += dt
        if self.time_since_measurement >= self.measurement_interval:
            self.time_since_measurement = 0.0
            
            measurement, actual = self.ekf.measure(player_x, player_y, self.x, self.y)
            
            if not self.ekf.initialized:
                self.ekf.initialize_from_measurement(measurement, self.x, self.y)
            else:
                self.ekf.update(measurement, self.x, self.y)
        
        # Get estimated player state
        est_px, est_py = self.ekf.get_estimated_position()
        est_vx, est_vy = self.ekf.get_estimated_velocity()
        
        # Current state and DQN action
        current_state = self._get_state(player)
        current_distance = math.sqrt((player_x - self.x)**2 + (player_y - self.y)**2)
        
        # DQN action selection
        action = self.dqn_agent.select_action(current_state, training=True)
        
        # Calculate reward
        reward = 0.01  # small positive for staying alive
        
        if self.prev_distance is not None:
            reward += (self.prev_distance - current_distance) * 0.2  # reward closing distance
        
        # Store experience
        if self.prev_state is not None:
            done = False  # Missile doesn't die on failure
            self.dqn_agent.store_experience(self.prev_state, self.prev_action, 
                                           reward, current_state, done)
        
        self.prev_state = current_state
        self.prev_action = action
        self.prev_distance = current_distance
        self.episode_reward += reward
        
        # Apply DQN action modulation
        # Actions: 0=turn_left, 1=straight, 2=turn_right
        dqn_turn_bias = 0.0
        if action == 0:
            dqn_turn_bias = -0.2  # Turn left
        elif action == 2:
            dqn_turn_bias = 0.2   # Turn right
        
        # Guidance calculation
        close_range_dist = current_distance
        
        if close_range_dist < CLOSE_RANGE_THRESHOLD:
            # Close-range: direct pursuit
            target_x = est_px
            target_y = est_py
        else:
            # Calculate intercept
            missile_pos = np.array([self.x, self.y])
            target_pos = np.array([est_px, est_py])
            target_vel = np.array([est_vx, est_vy])
            
            current_speed = math.sqrt(self.vx**2 + self.vy**2)
            if current_speed < 0.1:
                current_speed = MISSILE_BASE_SPEED
            
            intercept_result = calculate_intercept_point(
                missile_pos, target_pos, target_vel, current_speed
            )
            
            if intercept_result[0] is not None:
                target_x, target_y = intercept_result[0]
            else:
                target_x = est_px + est_vx * self.prediction_time
                target_y = est_py + est_vy * self.prediction_time
        
        # Calculate desired heading
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
        # Add DQN bias
        target_angle += dqn_turn_bias
        
        # Current heading
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed > 0.1:
            current_angle = math.atan2(self.vy, self.vx)
        else:
            current_angle = target_angle
        
        # Turn towards target
        angle_diff = target_angle - current_angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Dynamic turn rate (slower = faster turn)
        speed_ratio = current_speed / MISSILE_BASE_SPEED if current_speed > 0 else 0
        turn_rate = MISSILE_TURN_RATE * (1 + (1 - speed_ratio) * 2)
        turn_rate *= self.turn_rate_multiplier
        
        turn_amount = max(-turn_rate, min(turn_rate, angle_diff))
        new_angle = current_angle + turn_amount
        
        # Thrust calculation
        thrust_mult = get_thrust_multiplier(self.fuel, self.max_fuel, self.age, self.burn_time)
        max_speed = MISSILE_MAX_SPEED * thrust_mult
        
        # Accelerate
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt * thrust_mult, max_speed)
        
        # Minimum speed clamp
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        current_speed = max(current_speed, MISSILE_MIN_SPEED)
        current_speed = min(current_speed, max_speed)
        
        # Update velocity
        self.vx = current_speed * math.cos(new_angle)
        self.vy = current_speed * math.sin(new_angle)
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Track position
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Check removal
        if self.fuel <= 0:
            # Store final experience
            if self.prev_state is not None:
                final_reward = self.episode_reward
                done = True
                self.dqn_agent.store_experience(self.prev_state, self.prev_action,
                                               final_reward, current_state, done)
            return True
        
        # Out of bounds
        margin = 200
        if (self.x < -margin or self.x > SCREEN_WIDTH + margin or
            self.y < -margin or self.y > SCREEN_HEIGHT + margin):
            return True
        
        return False
    
    def check_collision(self, player: Player) -> bool:
        """Check collision with player."""
        if not player.alive:
            return False
        
        px, py = player.get_position()
        distance = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        
        return distance < (self.size + player.size * 0.7)
    
    def draw(self, screen: pygame.Surface):
        """Draw missile with fuel-based coloring."""
        fuel_ratio = max(0, self.fuel / self.max_fuel)
        
        # Color based on fuel
        base_color = PURPLE
        dark_color = (80, 20, 80)
        color = tuple(int(dark_color[i] + (base_color[i] - dark_color[i]) * fuel_ratio) 
                     for i in range(3))
        
        # Trajectory
        if len(self.position_history) > 1:
            width = max(1, int(2 * fuel_ratio))
            pygame.draw.lines(screen, color, False, self.position_history, width)
        
        # Missile body
        points = [
            (self.x, self.y - self.size),
            (self.x + self.size, self.y),
            (self.x, self.y + self.size),
            (self.x - self.size, self.y)
        ]
        pygame.draw.polygon(screen, color, points)
        
        # Velocity
        pygame.draw.line(screen, ORANGE, (self.x, self.y),
                        (self.x + self.vx * 10, self.y + self.vy * 10), 2)
        
        # Fuel bar
        bar_width = 15
        bar_height = 3
        pygame.draw.rect(screen, GRAY, 
                        (self.x - bar_width//2, self.y - 15, bar_width, bar_height))
        fuel_color = GREEN if fuel_ratio > 0.3 else ORANGE if fuel_ratio > 0.1 else RED
        pygame.draw.rect(screen, fuel_color,
                        (self.x - bar_width//2, self.y - 15, 
                         bar_width * fuel_ratio, bar_height))


class MissileType(Enum):
    """Different missile guidance types."""
    BASIC = 0      # Simple pursuit
    EKF = 1        # Kalman filter with prediction
    ADVANCED = 2   # EKF + true intercept guidance


# =============================================================================
# GAME CLASS
# =============================================================================

class Game:
    """Main game loop and management."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("DQN Missile Evasion - Real-time Learning")
        self.clock = pygame.time.Clock()
        
        # Device for DQN - with command-line override support
        self.device = self._get_device()
        
        # AI agent
        self.dqn_agent = DQNAgent(DQN_STATE_SIZE, DQN_ACTION_SIZE, device=self.device)
        
        # Game state
        self.running = True
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        
        # Player
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        
        # Missiles
        self.missiles = []
        self.last_spawn_time = pygame.time.get_ticks()
        
        # Difficulty
        self.difficulty_timer = pygame.time.get_ticks()
        self.speed_multiplier = 1.0
        self.speed_increase_interval = 10000
        
        # Training
        self.frame_count = 0
        self.show_debug = True
    
    def _get_device(self):
        """Determine which device to use (CPU or GPU).
        Priority: command-line args > DEVICE_PREFERENCE constant > auto-detect
        """
        device_choice = DEVICE_PREFERENCE.lower()
        
        # Check for command-line argument
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            if arg in ('cpu', 'cuda', 'gpu'):
                device_choice = 'cuda' if arg in ('cuda', 'gpu') else 'cpu'
        
        # Determine device
        if device_choice == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choice in ('cuda', 'gpu'):
            if not torch.cuda.is_available():
                print("⚠️  CUDA not available, falling back to CPU")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda")
        else:  # 'cpu'
            device = torch.device("cpu")
        
        return device
        """Spawn a new AI-controlled missile."""
        if len(self.missiles) >= MAX_MISSILES:
            return
        
        edge = random.randint(0, 3)
        if edge == 0:
            x, y = random.randint(0, SCREEN_WIDTH), -50
        elif edge == 1:
            x, y = SCREEN_WIDTH + 50, random.randint(0, SCREEN_HEIGHT)
        elif edge == 2:
            x, y = random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT + 50
        else:
            x, y = -50, random.randint(0, SCREEN_HEIGHT)
        
        missile = Missile(x, y, self.dqn_agent, self.speed_multiplier)
        self.missiles.append(missile)
    
    def handle_events(self):
        """Handle input and events."""
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
        
        dt = self.clock.get_time() / 1000.0
        self.frame_count += 1
        
        # Player
        keys = pygame.key.get_pressed()
        self.player.handle_input(keys)
        self.player.update(dt)
        
        # Spawn missiles
        if self.frame_count % int((MISSILE_SPAWN_INTERVAL / 1000.0) * FPS) == 0:
            self.spawn_missile()
        
        # Increase difficulty
        current_time = pygame.time.get_ticks()
        if current_time - self.difficulty_timer > self.speed_increase_interval:
            self.speed_multiplier += 0.05
            self.difficulty_timer = current_time
        
        # Update missiles
        missiles_to_remove = []
        hit = False
        
        for i, missile in enumerate(self.missiles):
            should_remove = missile.update(dt, self.player)
            if should_remove:
                missiles_to_remove.append(i)
            
            if missile.check_collision(self.player):
                self.player.die()
                self.game_over = True
                hit = True
                # Give final reward
                missile.episode_reward += 200
                self.dqn_agent.update_stats(missile.episode_reward, True)
        
        # Remove dead missiles
        for i in reversed(missiles_to_remove):
            self.missiles.pop(i)
        
        # DQN training
        if self.frame_count % DQN_UPDATE_FREQ == 0:
            self.dqn_agent.train()
        
        # Update score
        self.score = (pygame.time.get_ticks() - self.start_time) // 100
    
    def spawn_missile(self):
        """Spawn missile with difficulty-scaled type distribution."""
        edge = random.randint(0, 3)
        
        if edge == 0:
            x = random.randint(0, SCREEN_WIDTH)
            y = -50
        elif edge == 1:
            x = SCREEN_WIDTH + 50
            y = random.randint(0, SCREEN_HEIGHT)
        elif edge == 2:
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT + 50
        else:
            x = -50
            y = random.randint(0, SCREEN_HEIGHT)
        
        difficulty_factor = min(self.speed_multiplier / 3.0, 1.0)
        
        rand = random.random()
        if rand < 0.3:
            missile_type = MissileType.BASIC
        elif rand < 0.6 + difficulty_factor * 0.3:
            missile_type = MissileType.EKF
        else:
            missile_type = MissileType.ADVANCED
        
        missile = Missile(x, y, self.dqn_agent, self.speed_multiplier)
        self.missiles.append(missile)
    
    def draw(self):
        """Render game."""
        self.screen.fill(BLACK)
        
        # Grid
        grid_size = 50
        for x in range(0, SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # Draw entities
        self.player.draw(self.screen)
        for missile in self.missiles:
            missile.draw(self.screen)
        
        # UI
        self._draw_ui()
        
        if self.game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_ui(self):
        """Draw UI elements."""
        font = pygame.font.Font(None, 24)
        
        # Score
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Speed multiplier
        speed_text = font.render(f"Speed: {self.speed_multiplier:.2f}x", True, WHITE)
        self.screen.blit(speed_text, (10, 35))
        
        # Missile count
        missile_text = font.render(f"Missiles: {len(self.missiles)}/{MAX_MISSILES}", True, WHITE)
        self.screen.blit(missile_text, (10, 60))
        
        # DQN info
        epsilon_text = font.render(f"ε: {self.dqn_agent.epsilon:.4f}", True, CYAN)
        self.screen.blit(epsilon_text, (10, 85))
        
        reward_text = font.render(f"Avg Reward: {self.dqn_agent.avg_reward:.3f}", True, YELLOW)
        self.screen.blit(reward_text, (10, 110))
        
        if self.dqn_agent.total_missiles > 0:
            hit_rate = (self.dqn_agent.hit_count / self.dqn_agent.total_missiles) * 100
            hit_text = font.render(f"Hit Rate: {hit_rate:.1f}% ({self.dqn_agent.hit_count}/{self.dqn_agent.total_missiles})", 
                                  True, RED)
            self.screen.blit(hit_text, (10, 135))
        
        training_text = font.render(f"Training Steps: {self.dqn_agent.training_steps}", True, BLUE)
        self.screen.blit(training_text, (10, 160))
        
        # Buffer usage
        buffer_usage = len(self.dqn_agent.replay_buffer) / REPLAY_BUFFER_SIZE
        buffer_text = font.render(f"Buffer: {int(buffer_usage * 100)}%", True, GRAY)
        self.screen.blit(buffer_text, (10, 185))
        
        # Help
        help_text = font.render("A/D: Turn | D: Debug | ESC: Quit", True, GRAY)
        self.screen.blit(help_text, (10, SCREEN_HEIGHT - 30))
    
    def _draw_game_over(self):
        """Draw game over screen."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        large_font = pygame.font.Font(None, 72)
        font = pygame.font.Font(None, 36)
        
        game_over_text = large_font.render("Jagded", True, RED)
        score_text = font.render(f"F_S: {self.score}", True, WHITE)
        learning_text = font.render(f"M_LF: {self.dqn_agent.total_missiles}", True, YELLOW)
        restart_text = font.render("R or S", True, YELLOW)
        
        self.screen.blit(game_over_text, 
                        game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 80)))
        self.screen.blit(score_text,
                        score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 10)))
        self.screen.blit(learning_text,
                        learning_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 30)))
        self.screen.blit(restart_text,
                        restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80)))
    
    def restart(self):
        """Restart game."""
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.missiles = []
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        self.frame_count = 0
    
    def run(self):
        """Main game loop."""
        print("Starting DQN Missile Evasion Game...")
        print(f"Using device: {self.device}")
        print(f"DQN Network: input={DQN_STATE_SIZE}, hidden={DQN_HIDDEN_SIZE}, actions={DQN_ACTION_SIZE}")
        print(f"Replay Buffer Size: {REPLAY_BUFFER_SIZE}")
        print(f"Training: epsilon {DQN_EPSILON_START} → {DQN_EPSILON_END}, decay {DQN_EPSILON_DECAY}")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        print(f"\nFinal Statistics:")
        print(f"Total Missiles Trained: {self.dqn_agent.total_missiles}")
        print(f"Total Hits: {self.dqn_agent.hit_count}")
        print(f"Hit Rate: {(self.dqn_agent.hit_count / max(1, self.dqn_agent.total_missiles)) * 100:.1f}%")
        print(f"Training Steps: {self.dqn_agent.training_steps}")
        
        pygame.quit()


if __name__ == "__main__":
    # Print device options
    print("DQN Missile Evasion Game")
    print("=" * 50)
    
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help', 'help'):
        print("\nUsage: python ver3.py [device]")
        print("\nDevice options:")
        print("  cpu        - Use CPU for computations")
        print("  cuda/gpu   - Use NVIDIA GPU for computations")
        print("  auto       - Auto-detect (CUDA if available, else CPU)")
        print("\nExamples:")
        print("  python ver3.py cpu    # Force CPU")
        print("  python ver3.py gpu    # Force GPU")
        print("  python ver3.py auto   # Auto-detect (default)")
        print("\nOr modify DEVICE_PREFERENCE constant in the code.")
        print("=" * 50)
        sys.exit(0)
    
    # Create and run game
    game = Game()
    game.run()
MISSILE_SIZE = 10
MISSILE_BASE_SPEED = 2.0
MISSILE_ACCELERATION = 0.15
MISSILE_MAX_SPEED = 6.0
MISSILE_TURN_RATE = 0.08

# Fuel system
MISSILE_MAX_FUEL = 15.0        # Maximum fuel in seconds
MISSILE_BURN_TIME = 8.0        # Full thrust duration
MISSILE_MIN_SPEED = 1.5        # Minimum speed when fuel depletes

# EKF Settings
PROCESS_NOISE_POSITION = 0.5
PROCESS_NOISE_VELOCITY = 0.1
MEASUREMENT_NOISE_RANGE_BASE = 20.0
MEASUREMENT_NOISE_ANGLE_BASE = 0.1

# Distance-based noise scaling
NOISE_DISTANCE_MAX = 800.0

# Anti-orbiting parameters
CLOSING_VELOCITY_THRESHOLD = -0.1  # Relative speed threshold
INTERCEPT_TIME_MAX = 1.5           # Max intercept lookahead
CLOSE_RANGE_THRESHOLD = 80.0       # Force direct pursuit at this distance


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_intercept_point(missile_pos: np.ndarray, target_pos: np.ndarray, 
                              target_vel: np.ndarray, missile_speed: float) \
                              -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    Calculate true intercept point where missile can reach target.
    
    Solves pursuit-evasion equation:
    ||target_pos + target_vel * t - missile_pos|| = missile_speed * t
    """
    dx = target_pos[0] - missile_pos[0]
    dy = target_pos[1] - missile_pos[1]
    
    tx = target_vel[0]
    ty = target_vel[1]
    
    # Quadratic coefficients
    a = tx*tx + ty*ty - missile_speed*missile_speed
    b = 2.0 * (dx*tx + dy*ty)
    c = dx*dx + dy*dy
    
    # Handle edge cases
    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return None, None
        t = -c / b
    else:
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None, None
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        
        valid_times = [t for t in [t1, t2] if t > 0.01]
        if not valid_times:
            return None, None
        
        t = min(valid_times)
    
    if t < 0.01:
        return None, None
    
    # Clamp intercept time
    t = min(t, INTERCEPT_TIME_MAX)
    
    intercept_x = target_pos[0] + target_vel[0] * t
    intercept_y = target_pos[1] + target_vel[1] * t
    
    return (intercept_x, intercept_y), t


def get_distance_based_noise(distance: float, base_noise: float) -> float:
    """Calculate distance-adaptive measurement noise."""
    normalized_dist = min(distance / NOISE_DISTANCE_MAX, 1.0)
    noise_multiplier = 1.0 + normalized_dist
    return base_noise * noise_multiplier


def get_thrust_multiplier(fuel: float, max_fuel: float, age: float, burn_time: float) -> float:
    """
    Calculate thrust multiplier based on fuel and age.
    
    Full thrust during burn phase, then gradual decay.
    Formula: (fuel / max_fuel) ^ 1.5
    """
    if age < burn_time:
        return 1.0  # Full thrust during burn
    
    fuel_ratio = max(0, fuel / max_fuel)
    thrust = fuel_ratio ** 1.5
    return max(0, thrust)


# =============================================================================
# EXTENDED KALMAN FILTER CLASS
# =============================================================================

class ExtendedKalmanFilter:
    """Extended Kalman Filter for tracking a 2D moving target."""
    
    def __init__(self, missile_x: float, missile_y: float):
        """Initialize EKF in uninitialized state."""
        self.state_dim = 4
        self.meas_dim = 2
        
        self.state = np.zeros(self.state_dim)
        self.initialized = False
        
        self.P = np.diag([500.0, 500.0, 50.0, 50.0])
        self.Q = np.diag([
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_POSITION**2,
            PROCESS_NOISE_VELOCITY**2,
            PROCESS_NOISE_VELOCITY**2
        ])
        self.R = np.diag([
            MEASUREMENT_NOISE_RANGE_BASE**2,
            MEASUREMENT_NOISE_ANGLE_BASE**2
        ])
        
        self.dt = 1.0 / FPS
    
    def initialize_from_measurement(self, measurement: np.ndarray, 
                                   missile_x: float, missile_y: float):
        """Initialize filter state from first measurement (polar → Cartesian)."""
        if self.initialized:
            return
        
        range_meas, bearing_meas = measurement
        
        dx = range_meas * math.cos(bearing_meas)
        dy = range_meas * math.sin(bearing_meas)
        
        px = missile_x + dx
        py = missile_y + dy
        
        self.state = np.array([px, py, 0.0, 0.0])
        self.P = np.diag([100.0, 100.0, 30.0, 30.0])
        self.initialized = True
    
    def predict(self, dt: float = None):
        """Prediction step with constant velocity model."""
        if not self.initialized:
            return
        
        if dt is not None:
            self.dt = dt
        
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) * 0.5
    
    def measure(self, true_px: float, true_py: float, 
                missile_x: float, missile_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate noisy measurement with distance-adaptive noise."""
        dx = true_px - missile_x
        dy = true_py - missile_y
        true_range = math.sqrt(dx**2 + dy**2)
        true_bearing = math.atan2(dy, dx)
        
        range_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_RANGE_BASE)
        bearing_noise = get_distance_based_noise(true_range, MEASUREMENT_NOISE_ANGLE_BASE)
        
        noisy_range = true_range + random.gauss(0, range_noise)
        noisy_bearing = true_bearing + random.gauss(0, bearing_noise)
        noisy_bearing = math.atan2(math.sin(noisy_bearing), math.cos(noisy_bearing))
        noisy_range = max(1.0, noisy_range)
        
        measurement = np.array([noisy_range, noisy_bearing])
        actual = np.array([true_range, true_bearing])
        
        return measurement, actual
    
    def update(self, measurement: np.ndarray, missile_x: float, missile_y: float):
        """Update step with numerically stable matrix operations."""
        if not self.initialized:
            return
        
        px, py, vx, vy = self.state
        dx = px - missile_x
        dy = py - missile_y
        pred_range = math.sqrt(dx**2 + dy**2)
        pred_bearing = math.atan2(dy, dx)
        
        pred_measurement = np.array([pred_range, pred_bearing])
        
        if pred_range > 0.001:
            dr_dpx = dx / pred_range
            dr_dpy = dy / pred_range
            denom = dx**2 + dy**2
            db_dpx = -dy / denom
            db_dpy = dx / denom
        else:
            dr_dpx = dr_dpy = db_dpx = db_dpy = 0
        
        H = np.array([
            [dr_dpx, dr_dpy, 0, 0],
            [db_dpx, db_dpy, 0, 0]
        ], dtype=np.float64)
        
        y = measurement - pred_measurement
        y[1] = math.atan2(math.sin(y[1]), math.cos(y[1]))
        
        S = H @ self.P @ H.T + self.R
        S = (S + S.T) * 0.5
        
        try:
            K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        self.state = self.state + K @ y
        
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P
        self.P = (self.P + self.P.T) * 0.5
        self.P += np.eye(self.state_dim) * 1e-6
    
    def get_estimated_position(self) -> Tuple[float, float]:
        return self.state[0], self.state[1]
    
    def get_estimated_velocity(self) -> Tuple[float, float]:
        return self.state[2], self.state[3]
    
    def get_position_covariance(self) -> np.ndarray:
        return self.P[0:2, 0:2]
    
    def reset(self):
        self.state = np.zeros(self.state_dim)
        self.initialized = False
        self.P = np.diag([500.0, 500.0, 50.0, 50.0])


# =============================================================================
# PLAYER CLASS
# =============================================================================

class Player:
    """Player entity controlled by keyboard."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.size = PLAYER_SIZE
        self.alive = True
        
        self.position_history: List[Tuple[float, float]] = []
        self.max_history = 50
    
    def handle_input(self, keys):
        accel = PLAYER_ACCELERATION
        
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.vy -= accel
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.vy += accel
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.vx -= accel
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.vx += accel
        
        self.vx *= PLAYER_FRICTION
        self.vy *= PLAYER_FRICTION
        
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > PLAYER_MAX_SPEED:
            scale = PLAYER_MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale
    
    def update(self, dt: float):
        if not self.alive:
            return
        
        self.x += self.vx
        self.y += self.vy
        
        self.x = max(self.size, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(self.size, min(SCREEN_HEIGHT - self.size, self.y))
        
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def draw(self, screen: pygame.Surface):
        if not self.alive:
            return
        
        if len(self.position_history) > 1:
            pygame.draw.lines(screen, GREEN, False, self.position_history, 2)
        
        if abs(self.vx) > 0.1 or abs(self.vy) > 0.1:
            angle = math.atan2(self.vy, self.vx)
        else:
            angle = -math.pi / 2
        
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
        pygame.draw.line(screen, GREEN, (self.x, self.y), 
                        (self.x + self.vx * 10, self.y + self.vy * 10), 2)
    
    def get_position(self) -> Tuple[float, float]:
        return self.x, self.y
    
    def get_velocity(self) -> Tuple[float, float]:
        return self.vx, self.vy
    
    def die(self):
        self.alive = False


# =============================================================================
# MISSILE CLASS
# =============================================================================

class Missile:
    """Missile with fuel system and multiple guidance strategies."""
    
    def __init__(self, x: float, y: float, missile_type: MissileType = MissileType.EKF, 
                 speed_multiplier: float = 1.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.speed = MISSILE_BASE_SPEED * speed_multiplier
        self.speed_multiplier = speed_multiplier
        self.size = MISSILE_SIZE
        self.missile_type = missile_type
        
        # Fuel system
        self.max_fuel = MISSILE_MAX_FUEL
        self.fuel = self.max_fuel
        self.age = 0.0
        self.burn_time = MISSILE_BURN_TIME
        
        if missile_type != MissileType.BASIC:
            self.ekf = ExtendedKalmanFilter(x, y)
        else:
            self.ekf = None
        
        self.position_history: List[Tuple[float, float]] = []
        self.estimated_position_history: List[Tuple[float, float]] = []
        self.intercept_history: List[Tuple[float, float]] = []
        self.max_history = 100
        
        self.measurement_interval = 0.1
        self.time_since_measurement = 0.0
        
        self.prediction_time = 0.5
        self.basic_lead_time = 0.3
    
    def update(self, dt: float, player: Player) -> bool:
        """
        Update missile state. Return True if missile should be removed.
        """
        # Update age and fuel
        self.age += dt
        self.fuel -= dt
        
        # Calculate thrust multiplier
        thrust_mult = get_thrust_multiplier(self.fuel, self.max_fuel, self.age, self.burn_time)
        current_max_speed = MISSILE_MAX_SPEED * thrust_mult
        
        # Applies guidance based on type
        if self.missile_type == MissileType.BASIC:
            self._update_basic(dt, player, thrust_mult)
        elif self.missile_type == MissileType.EKF:
            self._update_ekf(dt, player, thrust_mult)
        elif self.missile_type == MissileType.ADVANCED:
            self._update_advanced(dt, player, thrust_mult)
        
        # Apply speed limits (with minimum speed to prevent orbiting)
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        current_speed = max(current_speed, MISSILE_MIN_SPEED)
        current_speed = min(current_speed, current_max_speed)
        
        if current_speed > 0.01:
            scale = current_speed / math.sqrt(self.vx**2 + self.vy**2 + 1e-6)
            self.vx *= scale
            self.vy *= scale
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Track position
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Check removal conditions
        return self._should_remove(player)
    
    def _should_remove(self, player: Player) -> bool:
        """
        Determine if missile should be removed.
        - Fuel depleted
        - Speed too low
        - Closing velocity negative (moving away)
        - Out of bounds
        """
        if self.fuel <= 0:
            return True
        
        current_speed = math.sqrt(self.vx**2 + self.vy**2)
        if current_speed < 0.5:
            return True
        
        # Check if missile is moving away (closing velocity)
        px, py = player.get_position()
        dx = px - self.x
        dy = py - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0.1:
            direction_x = dx / distance
            direction_y = dy / distance
            closing_speed = self.vx * direction_x + self.vy * direction_y
            
            if closing_speed < CLOSING_VELOCITY_THRESHOLD:
                return True
        
        # Out of bounds with margin
        margin = 200
        if (self.x < -margin or self.x > SCREEN_WIDTH + margin or
            self.y < -margin or self.y > SCREEN_HEIGHT + margin):
            return True
        
        return False
    
    def _update_basic(self, dt: float, player: Player, thrust_mult: float):
        """BASIC guidance: simple pursuit with velocity lead."""
        player_x, player_y = player.get_position()
        player_vx, player_vy = player.get_velocity()
        
        target_x = player_x + player_vx * self.basic_lead_time
        target_y = player_y + player_vy * self.basic_lead_time
        
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
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
        
        # Dynamic turn rate (slower missiles turn faster)
        speed_ratio = current_speed / MISSILE_BASE_SPEED if current_speed > 0 else 0
        turn_rate = MISSILE_TURN_RATE * (1 + (1 - speed_ratio) * 2)
        
        turn_amount = max(-turn_rate, min(turn_rate, angle_diff))
        new_angle = current_angle + turn_amount
        
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt * thrust_mult, MISSILE_MAX_SPEED * thrust_mult)
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
    
    def _update_ekf(self, dt: float, player: Player, thrust_mult: float):
        """EKF guidance: Kalman filter with velocity prediction."""
        player_x, player_y = player.get_position()
        
        self.ekf.predict(dt)
        
        self.time_since_measurement += dt
        if self.time_since_measurement >= self.measurement_interval:
            self.time_since_measurement = 0.0
            
            measurement, actual = self.ekf.measure(
                player_x, player_y, self.x, self.y
            )
            
            if not self.ekf.initialized:
                self.ekf.initialize_from_measurement(measurement, self.x, self.y)
            
            self.ekf.update(measurement, self.x, self.y)
        
        est_px, est_py = self.ekf.get_estimated_position()
        est_vx, est_vy = self.ekf.get_estimated_velocity()
        
        # Check if should switch to close-range pursuit
        close_range_dist = math.sqrt((est_px - self.x)**2 + (est_py - self.y)**2)
        
        if close_range_dist < CLOSE_RANGE_THRESHOLD:
            future_px = est_px
            future_py = est_py
        else:
            future_px = est_px + est_vx * self.prediction_time
            future_py = est_py + est_vy * self.prediction_time
        
        dx = future_px - self.x
        dy = future_py - self.y
        target_angle = math.atan2(dy, dx)
        
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
        
        speed_ratio = current_speed / MISSILE_BASE_SPEED if current_speed > 0 else 0
        turn_rate = MISSILE_TURN_RATE * (1 + (1 - speed_ratio) * 2)
        
        turn_amount = max(-turn_rate, min(turn_rate, angle_diff))
        new_angle = current_angle + turn_amount
        
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt * thrust_mult, MISSILE_MAX_SPEED * thrust_mult)
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        
        if self.ekf.initialized:
            self.estimated_position_history.append((est_px, est_py))
            if len(self.estimated_position_history) > self.max_history:
                self.estimated_position_history.pop(0)
    
    def _update_advanced(self, dt: float, player: Player, thrust_mult: float):
        """ADVANCED guidance: EKF + true intercept."""
        player_x, player_y = player.get_position()
        
        self.ekf.predict(dt)
        
        self.time_since_measurement += dt
        if self.time_since_measurement >= self.measurement_interval:
            self.time_since_measurement = 0.0
            
            measurement, actual = self.ekf.measure(
                player_x, player_y, self.x, self.y
            )
            
            if not self.ekf.initialized:
                self.ekf.initialize_from_measurement(measurement, self.x, self.y)
            
            self.ekf.update(measurement, self.x, self.y)
        
        est_px, est_py = self.ekf.get_estimated_position()
        est_vx, est_vy = self.ekf.get_estimated_velocity()
        
        # Check close range
        close_range_dist = math.sqrt((est_px - self.x)**2 + (est_py - self.y)**2)
        
        if close_range_dist < CLOSE_RANGE_THRESHOLD:
            target_x = est_px
            target_y = est_py
        else:
            missile_pos = np.array([self.x, self.y])
            target_pos = np.array([est_px, est_py])
            target_vel = np.array([est_vx, est_vy])
            
            current_speed = math.sqrt(self.vx**2 + self.vy**2)
            if current_speed < 0.1:
                current_speed = MISSILE_BASE_SPEED
            
            intercept_result = calculate_intercept_point(
                missile_pos, target_pos, target_vel, current_speed
            )
            
            if intercept_result[0] is not None:
                target_x, target_y = intercept_result[0]
                self.intercept_history.append((target_x, target_y))
            else:
                target_x = est_px + est_vx * self.prediction_time
                target_y = est_py + est_vy * self.prediction_time
            
            if len(self.intercept_history) > self.max_history:
                self.intercept_history.pop(0)
        
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        
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
        
        speed_ratio = current_speed / MISSILE_BASE_SPEED if current_speed > 0 else 0
        turn_rate = MISSILE_TURN_RATE * (1 + (1 - speed_ratio) * 2)
        
        turn_amount = max(-turn_rate, min(turn_rate, angle_diff))
        new_angle = current_angle + turn_amount
        
        self.speed = min(self.speed + MISSILE_ACCELERATION * dt * thrust_mult, MISSILE_MAX_SPEED * thrust_mult)
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        
        if self.ekf.initialized:
            self.estimated_position_history.append((est_px, est_py))
            if len(self.estimated_position_history) > self.max_history:
                self.estimated_position_history.pop(0)
    
    def check_collision(self, player: Player) -> bool:
        if not player.alive:
            return False
        
        px, py = player.get_position()
        dx = self.x - px
        dy = self.y - py
        distance = math.sqrt(dx**2 + dy**2)
        
        return distance < (self.size + player.size * 0.7)
    
    def draw(self, screen: pygame.Surface, show_ekf_info: bool = True):
        """Draw missile with fuel-based visualization."""
        # Determine color based on type and fuel
        fuel_ratio = max(0, self.fuel / self.max_fuel)
        
        if self.missile_type == MissileType.BASIC:
            base_color = RED
            dark_color = DARK_RED
        elif self.missile_type == MissileType.EKF:
            base_color = BLUE
            dark_color = DARK_BLUE
        else:
            base_color = PURPLE
            dark_color = DARK_PURPLE
        
        # Blend color based on fuel
        color = tuple(int(dark_color[i] + (base_color[i] - dark_color[i]) * fuel_ratio) 
                     for i in range(3))
        
        # Draw trajectory with fuel-based width
        if len(self.position_history) > 1:
            width = max(1, int(3 * fuel_ratio))
            pygame.draw.lines(screen, color, False, self.position_history, width)
        
        # Draw EKF information
        if show_ekf_info and self.ekf is not None and self.ekf.initialized:
            est_x, est_y = self.ekf.get_estimated_position()
            
            cross_size = 8
            pygame.draw.line(screen, CYAN, 
                           (est_x - cross_size, est_y - cross_size),
                           (est_x + cross_size, est_y + cross_size), 2)
            pygame.draw.line(screen, CYAN,
                           (est_x + cross_size, est_y - cross_size),
                           (est_x - cross_size, est_y + cross_size), 2)
            
            if len(self.estimated_position_history) > 1:
                pygame.draw.lines(screen, CYAN, False, 
                                self.estimated_position_history, 1)
            
            cov = self.ekf.get_position_covariance()
            self._draw_covariance_ellipse(screen, est_x, est_y, cov)
        
        # Draw intercept points for ADVANCED missiles
        if show_ekf_info and self.missile_type == MissileType.ADVANCED:
            if len(self.intercept_history) > 1:
                pygame.draw.lines(screen, YELLOW, False, 
                                self.intercept_history, 1)
            if len(self.intercept_history) > 0:
                ic_x, ic_y = self.intercept_history[-1]
                pygame.draw.circle(screen, YELLOW, (int(ic_x), int(ic_y)), 5, 1)
        
        # Draw missile body
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
        
        # Draw fuel bar
        self._draw_fuel_bar(screen)
    
    def _draw_fuel_bar(self, screen: pygame.Surface):
        """Draw small fuel indicator above missile."""
        bar_width = 15
        bar_height = 3
        fuel_ratio = max(0, self.fuel / self.max_fuel)
        
        # Background
        pygame.draw.rect(screen, GRAY, 
                        (self.x - bar_width//2, self.y - 15, bar_width, bar_height))
        
        # Fuel level
        fuel_color = GREEN if fuel_ratio > 0.3 else ORANGE if fuel_ratio > 0.1 else RED
        pygame.draw.rect(screen, fuel_color,
                        (self.x - bar_width//2, self.y - 15, 
                         bar_width * fuel_ratio, bar_height))
    
    def _draw_covariance_ellipse(self, screen: pygame.Surface, 
                                  cx: float, cy: float, cov: np.ndarray):
        """Draw covariance uncertainty ellipses."""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
        except np.linalg.LinAlgError:
            return
        
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        num_points = 30
        
        for n_std in [1, 2]:
            points = []
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                
                local_x = n_std * 2.0 * math.sqrt(max(eigenvalues[0], 0)) * math.cos(angle)
                local_y = n_std * 2.0 * math.sqrt(max(eigenvalues[1], 0)) * math.sin(angle)
                
                world_x = cx + eigenvectors[0, 0] * local_x + eigenvectors[1, 0] * local_y
                world_y = cy + eigenvectors[0, 1] * local_x + eigenvectors[1, 1] * local_y
                
                points.append((world_x, world_y))
            
            if len(points) > 2:
                pygame.draw.polygon(screen, CYAN, points, 1)
    
    def get_ekf_error(self, player: Player) -> Tuple[float, float]:
        """Calculate EKF estimation error."""
        if self.ekf is None or not self.ekf.initialized:
            return 0, 0
        
        est_x, est_y = self.ekf.get_estimated_position()
        true_x, true_y = player.get_position()
        return est_x - true_x, est_y - true_y


# =============================================================================
# GAME CLASS
# =============================================================================

class Game:
    """Main game class managing game loop and state."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Missile Evasion - Advanced EKF")
        self.clock = pygame.time.Clock()
        
        self.running = True
        self.game_over = False
        self.score = 0
        self.start_time = pygame.time.get_ticks()
        
        self.player = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.missiles: List[Missile] = []
        
        self.spawn_interval = 5000
        self.last_spawn_time = pygame.time.get_ticks()
        
        self.difficulty_timer = 0
        self.speed_increase_interval = 10000
        self.speed_multiplier = 1.0
        
        self.show_debug = True
    
    def spawn_missile(self):
        """Spawn missile with difficulty-scaled type distribution."""
        edge = random.randint(0, 3)
        
        if edge == 0:
            x = random.randint(0, SCREEN_WIDTH)
            y = -50
        elif edge == 1:
            x = SCREEN_WIDTH + 50
            y = random.randint(0, SCREEN_HEIGHT)
        elif edge == 2:
            x = random.randint(0, SCREEN_WIDTH)
            y = SCREEN_HEIGHT + 50
        else:
            x = -50
            y = random.randint(0, SCREEN_HEIGHT)
        
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
        if self.game_over:
            return
        
        dt = self.clock.get_time() / 1000.0
        
        keys = pygame.key.get_pressed()
        self.player.handle_input(keys)
        self.player.update(dt)
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_missile()
            self.last_spawn_time = current_time
        
        if current_time - self.difficulty_timer > self.speed_increase_interval:
            self.speed_multiplier += 0.1
            self.difficulty_timer = current_time
            self.spawn_interval = max(2000, self.spawn_interval - 200)
        
        # Update missiles and remove those that should be deleted
        missiles_to_remove = []
        for i, missile in enumerate(self.missiles):
            should_remove = missile.update(dt, self.player)
            if should_remove:
                missiles_to_remove.append(i)
            
            if missile.check_collision(self.player):
                self.player.die()
                self.game_over = True
        
        # Remove missiles in reverse order to maintain indices
        for i in reversed(missiles_to_remove):
            self.missiles.pop(i)
        
        self.score = (pygame.time.get_ticks() - self.start_time) // 100
    
    def draw(self):
        self.screen.fill(BLACK)
        
        self._draw_grid()
        self.player.draw(self.screen)
        
        for missile in self.missiles:
            missile.draw(self.screen, self.show_debug)
        
        self._draw_ui()
        
        if self.game_over:
            self._draw_game_over()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        grid_size = 50
        for x in range(0, SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, DARK_GRAY, (0, y), (SCREEN_WIDTH, y))
    
    def _draw_ui(self):
        font = pygame.font.Font(None, 24)
        
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        diff_text = font.render(f"Speed: {self.speed_multiplier:.1f}x", True, WHITE)
        self.screen.blit(diff_text, (10, 35))
        
        total_missiles = len(self.missiles)
        basic_count = sum(1 for m in self.missiles if m.missile_type == MissileType.BASIC)
        ekf_count = sum(1 for m in self.missiles if m.missile_type == MissileType.EKF)
        adv_count = sum(1 for m in self.missiles if m.missile_type == MissileType.ADVANCED)
        
        missile_text = font.render(f"Missiles: {total_missiles} (B:{basic_count} E:{ekf_count} A:{adv_count})", 
                                  True, WHITE)
        self.screen.blit(missile_text, (10, 60))
        
        legend_y = 95
        pygame.draw.polygon(self.screen, RED, [
            (10, legend_y - 7), (17, legend_y), (10, legend_y + 7), (3, legend_y)
        ])
        legend_text = font.render("BASIC (Pursuit)", True, RED)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        legend_y += 25
        pygame.draw.polygon(self.screen, BLUE, [
            (10, legend_y - 7), (17, legend_y), (10, legend_y + 7), (3, legend_y)
        ])
        legend_text = font.render("EKF (Prediction)", True, BLUE)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        legend_y += 25
        pygame.draw.polygon(self.screen, PURPLE, [
            (10, legend_y - 7), (17, legend_y), (10, legend_y + 7), (3, legend_y)
        ])
        legend_text = font.render("ADVANCED (Intercept)", True, PURPLE)
        self.screen.blit(legend_text, (25, legend_y - 10))
        
        if self.show_debug and not self.game_over and len(self.missiles) > 0:
            missile = self.missiles[0]
            error_x, error_y = missile.get_ekf_error(self.player)
            error_dist = math.sqrt(error_x**2 + error_y**2)
            
            debug_y = 265
            debug_lines = [
                f"First Missile: {missile.missile_type.name}",
                f"Fuel: {missile.fuel:.1f}s Age: {missile.age:.1f}s",
                f"EKF Error: {error_dist:.1f}px",
                f"Player Vel: ({self.player.vx:.1f}, {self.player.vy:.1f})",
            ]
            
            if missile.ekf is not None and missile.ekf.initialized:
                debug_lines.append(f"Est Vel: ({missile.ekf.state[2]:.1f}, {missile.ekf.state[3]:.1f})")
            
            for line in debug_lines:
                text = font.render(line, True, CYAN)
                self.screen.blit(text, (10, debug_y))
                debug_y += 25
        
        help_text = font.render("WASD/Arrows: Move | D: Debug | ESC: Quit", True, GRAY)
        self.screen.blit(help_text, (10, SCREEN_HEIGHT - 30))
    
    def _draw_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        large_font = pygame.font.Font(None, 72)
        font = pygame.font.Font(None, 36)
        
        game_over_text = large_font.render("GAME OVER", True, RED)
        score_text = font.render(f"Final Score: {self.score}", True, WHITE)
        restart_text = font.render("Press SPACE or R to restart", True, YELLOW)
        
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
        self.screen.blit(game_over_text, text_rect)
        
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10))
        self.screen.blit(score_text, score_rect)
        
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 60))
        self.screen.blit(restart_text, restart_rect)
    
    def restart(self):
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
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()



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
            # Calculate K = (P @ H^T) @ solve(S, I)
            # This is numerically more stable than direct inversion
            # Shape: (4x2) @ (2x2) = (4x2) ✓
            K = self.P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
            
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