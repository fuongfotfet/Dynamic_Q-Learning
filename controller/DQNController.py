import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import heapq
import math
from collections import deque
from controller.Controller import Controller, action_space, decision_movement, remap_keys, convertphi, convertdeltaphi, convertdeltad, angle


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture adapted from DQN-main system
    Input: 29 dimensions (5x5 matrix + grid distance-to-goal + dynamic obstacle features)
    Output: 8 actions (8 directions, no stay)
    """
    def __init__(self, input_dim=29, output_dim=8):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Value stream V(s) - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage stream A(s,a) - estimates action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        # Initialize weights after all submodules are defined
        self._init_weights()

    def forward(self, x):
        # Extract features
        features = self.feature(x)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine using Dueling DQN formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        if len(advantage.shape) == 1:  # Single sample
            q_values = value + advantage - advantage.mean(dim=0, keepdim=True)
        else:  # Batch
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

    def _init_weights(self):
        # Kaiming init for ReLU networks and zeros for bias
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Make the last layers small so initial Q-values are near zero but not identical
        last_value = self.value_stream[-1]
        last_adv = self.advantage_stream[-1]
        nn.init.uniform_(last_value.weight, -1e-3, 1e-3)
        nn.init.zeros_(last_value.bias)
        nn.init.uniform_(last_adv.weight, -1e-3, 1e-3)
        nn.init.zeros_(last_adv.bias)


class DQNLearning(Controller):
    """
    DQN Controller adapted for CQL system
    Maintains CQL interface while using DQN internally
    """
    
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles=None):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles)
        
        # DQN Parameters
        # 5x5 matrix (25) + grid-distance-to-goal (1) + dynamic obstacle triplet (c_phi, c_deltaphi, c_deltad) (3)
        self.state_dim = 5 * 5 + 1 + 3  # = 29 dimensions
        self.action_dim = 8  # CQL action space: 8 directions + stay
        
        # Device setup - prioritize MPS on macOS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Neural Networks
        self.q_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        # DQN Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate (will be managed by CQL system)
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.batch_size = 64
        
        # Experience Replay
        self.memory = deque(maxlen=20000)
        self.target_update_freq = 200
        self.step_count = 0
        
        # Action mapping: DQN action indices to CQL action keys
        self.action_mapping = [
            "right_1",      # 0: (1, 0)   -> East
            "down-right_1", # 1: (1, 1)   -> Southeast  
            "down_1",       # 2: (0, 1)   -> South
            "down-left_1",  # 3: (-1, 1)  -> Southwest
            "left_1",       # 4: (-1, 0)  -> West
            "up-left_1",    # 5: (-1, -1) -> Northwest
            "up_1",         # 6: (0, -1)  -> North
            "up-right_1",   # 7: (1, -1)  -> Northeast
            # "stay"          # 8: (0, 0)   -> Stay
        ]
        
        # Training state
        self.is_training = True
        self.episode_experiences = []  # Store experiences for current episode
        
        # State tracking for training
        self.previous_state = None
        self.previous_action = None
        self.previous_distance = None
        self.current_state = None
        self.current_action = None
        
        # Training frequency tracking
        self.total_simulation_calls = 0
        self.actual_training_calls = 0
        
        # Anti-trap mechanism
        self.position_history = {}
        self.position_memory_size = 100
        self.position_history_list = []
        
        # Curiosity mechanism
        self.visit_counts = {}
        self.curiosity_factor = 0.1
        
        # Initialize distance to goal matrix (like CombinedQL)
        self._initialize_distance_matrix()
        
        # dist_to_goal is set in _initialize_distance_matrix() 
        
        # Max distance (in pixels) for normalization; set during _initialize_distance_matrix
        self.max_distance = getattr(self, 'max_distance', 1.0)

        # Expert prior (A*/greedy cost-to-go) guidance parameters
        self.p_prior = 0.9
        self.p_prior_decay = 0.998
        self.p_prior_min = 0.0
        
        print(f"DQN Controller initialized - Device: {self.device}")
    
    def _initialize_distance_matrix(self):
        """
        Initialize distance to goal matrix using Dijkstra algorithm
        Similar to CombinedQL implementation
        """
        # Build static occupancy grid
        M = int(self.env_size / self.cell_size)
        grid = [[0]*M for _ in range(M)]
        
        for obs in self.static_obstacles:
            x1, x2, y1, y2 = obs.return_coordinate()
            i_min = max(0, int((x1 - self.env_padding)//self.cell_size))
            i_max = min(M-1, int((x2 - self.env_padding)//self.cell_size))
            j_min = max(0, int((y1 - self.env_padding)//self.cell_size))
            j_max = min(M-1, int((y2 - self.env_padding)//self.cell_size))
            for ii in range(i_min, i_max+1):
                for jj in range(j_min, j_max+1):
                    grid[jj][ii] = 1
        
        # Compute distances using Dijkstra from goal
        goal_i = int((self.goal[0] - self.env_padding)//self.cell_size)
        goal_j = int((self.goal[1] - self.env_padding)//self.cell_size)
        
        self.dist_to_goal = self._compute_distances(grid, goal_i, goal_j)
        
        # Compute max finite distance (convert to pixel units) for normalization
        finite_values = [v for row in self.dist_to_goal for v in row if math.isfinite(v)]
        if len(finite_values) > 0:
            self.max_distance = max(finite_values) * self.cell_size
        else:
            # Fallback to environment diagonal in pixels
            diag_cells = math.sqrt(2) * (int(self.env_size / self.cell_size) - 1)
            self.max_distance = max(diag_cells * self.cell_size, 1.0)
    
    def _compute_distances(self, grid, goal_i, goal_j):
        """
        Compute shortest distances from all cells to goal using Dijkstra
        """
        M = len(grid)
        INF = float('inf')
        dist = [[INF]*M for _ in range(M)]
        dist[goal_j][goal_i] = 0.0
        
        # Priority queue: (current_cost, i, j)
        heap = [(0.0, goal_i, goal_j)]
        
        # 8-directional movement with costs
        neighbours = [
            ( 1,  0, 1.0), (-1,  0, 1.0),
            ( 0,  1, 1.0), ( 0, -1, 1.0),
            ( 1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
            (-1,  1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]
        
        while heap:
            cost, i, j = heapq.heappop(heap)
            if cost > dist[j][i]:
                continue
            for di, dj, w in neighbours:
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < M and grid[nj][ni] == 0:
                    new_cost = cost + w
                    if new_cost < dist[nj][ni]:
                        dist[nj][ni] = new_cost
                        heapq.heappush(heap, (new_cost, ni, nj))
        return dist
    
    def _normalize_distance_to_goal(self, distance_in_pixels: float) -> float:
        """Normalize distance-to-goal into [0,1]. If unreachable/NaN, return 1.0."""
        if not np.isfinite(distance_in_pixels) or self.max_distance <= 0:
            return 1.0
        norm = distance_in_pixels / self.max_distance
        # Clamp to [0,1]
        if norm < 0.0:
            return 0.0
        if norm > 1.0:
            return 1.0
        return float(norm)

    def _cell_blocked(self, grid_x: int, grid_y: int) -> bool:
        """Return True if grid cell is out of bounds or inside any static obstacle."""
        grid_width = grid_height = int(self.env_size / self.cell_size)
        if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
            return True
        pixel_x = self.env_padding + grid_x * self.cell_size
        pixel_y = self.env_padding + grid_y * self.cell_size
        for obs in self.static_obstacles:
            x1, x2, y1, y2 = obs.return_coordinate()
            if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
                return True
        return False

    def _expert_greedy_action_idx(self, rb) -> int:
        """Pick action that greedily minimizes Dijkstra distance-to-goal among valid neighbors."""
        i, j = self.convertState(rb)
        M = int(self.env_size / self.cell_size)
        candidates = [
            ( 1,  0, 0), ( 1,  1, 1), ( 0,  1, 2), (-1,  1, 3),
            (-1,  0, 4), (-1, -1, 5), ( 0, -1, 6), ( 1, -1, 7),
        ]
        best = None  # (dist, action_idx)
        for dx, dy, a_idx in candidates:
            ni, nj = i + dx, j + dy
            if 0 <= ni < M and 0 <= nj < M and not self._cell_blocked(ni, nj):
                d = self.dist_to_goal[nj][ni] if math.isfinite(self.dist_to_goal[nj][ni]) else float('inf')
                if best is None or d < best[0]:
                    best = (d, a_idx)
        return best[1] if best is not None else 0  # fallback: right_1
    
    def get_state_representation(self, rb, obstacle_position=None):
        """
        Convert robot position to DQN state representation
        Returns: (5x5 matrix, grid_distance_to_goal, (c_phi, c_deltaphi, c_deltad))
        """
        # Get robot grid position
        robot_grid_x, robot_grid_y = self.convertState(rb)
        # print(robot_grid_x, robot_grid_y)

        goal_grid_x = math.ceil((self.goal[0] - self.env_padding) // self.cell_size)
        goal_grid_y = math.ceil((self.goal[1] - self.env_padding) // self.cell_size)
        
        # Create 5x5 state matrix
        state_matrix = np.zeros((5, 5), dtype=int)
        
        # Grid dimensions (assuming 32x32 based on 512/16)
        grid_width = grid_height = int(self.env_size / self.cell_size)
        
        # Fill 5x5 matrix around robot
        for i in range(-2, 3):  # -2, -1, 0, 1, 2
            for j in range(-2, 3):
                grid_x = robot_grid_x + i
                grid_y = robot_grid_y + j
                state_idx_x = i + 2
                state_idx_y = j + 2
                
                # Check boundaries
                if not (0 < grid_x <= grid_width and 0 < grid_y <= grid_height):
                    state_matrix[state_idx_y, state_idx_x] = 1
                    continue

                if (grid_x, grid_y) == (goal_grid_x, goal_grid_y):
                    state_matrix[state_idx_y, state_idx_x] = 3
                    continue
                
                # Check static obstacles
                pixel_x = self.env_padding + grid_x * self.cell_size
                pixel_y = self.env_padding + grid_y * self.cell_size
                
                for obs in self.static_obstacles:
                    x1, x2, y1, y2 = obs.return_coordinate()
                    if x1 + 2 <= pixel_x - 2 <= x2 and y1 + 2 <= pixel_y - 2 <= y2:
                        state_matrix[state_idx_y, state_idx_x] = 1
                        break
        
        # Mark robot position (center of 5x5 matrix)
        state_matrix[2, 2] = 2
        # Compute grid-based distance-to-goal using Dijkstra distance matrix
        if 0 <= robot_grid_x < len(self.dist_to_goal[0]) and 0 <= robot_grid_y < len(self.dist_to_goal):
            d = self.dist_to_goal[robot_grid_y][robot_grid_x]
            grid_distance = d if np.isfinite(d) else -1.0
        else:
            grid_distance = -1.0
        
        # Dynamic obstacle features (c_phi, c_deltaphi, c_deltad)
        # Default when no obstacle in vision: (-1, -1, -1)
        c_phi = -10
        c_deltaphi = -10
        c_deltad = -10
        
        if obstacle_position is not None and isinstance(obstacle_position, (list, tuple)) and len(obstacle_position) == 2:
            try:
                obstacle_before = obstacle_position[0]
                obstacle_after = obstacle_position[1]
                # Distances
                distance_to_obstacle = math.hypot(rb.pos[0] - obstacle_before[0], rb.pos[1] - obstacle_before[1])
                distance_to_obstacle_next = math.hypot(rb.pos[0] - obstacle_after[0], rb.pos[1] - obstacle_after[1])
                # Direction to goal (robot nextPosition, consistent with CombinedQL)
                rb_direction = rb.nextPosition(self.goal)
                phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0], obstacle_before[1] - rb.pos[1])
                phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0], obstacle_after[1] - rb.pos[1])
                # Discretize
                c_phi = convertphi(phi / np.pi * 180)
                c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
                c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))
            except Exception:
                c_phi, c_deltaphi, c_deltad = -10, -10, -10
        
        # print(state_matrix, grid_distance, (c_phi, c_deltaphi, c_deltad))
        return state_matrix, grid_distance, (c_phi, c_deltaphi, c_deltad)
    
    def makeDecision(self, rb) -> tuple:
        """
        CQL interface: Make decision for robot movement
        Returns: movement tuple (dx, dy)
        """
        # Get DQN state representation (matrix + grid distance + obstacle triplet)
        state_matrix, grid_distance, obstacle_triplet = self.get_state_representation(rb)
        
        # Convert to 29D neural network input
        state_flat = state_matrix.flatten().astype(np.float32)
        combined_state = np.concatenate([
            state_flat,
            np.array([grid_distance], dtype=np.float32),
            np.array(list(obstacle_triplet), dtype=np.float32)
        ])
        state_tensor = torch.from_numpy(combined_state).to(self.device)
        
        # Configure network mode
        if self.is_training:
            self.q_network.train()
        else:
            self.q_network.eval()
        
        # Compute Q-values once (for debug and greedy selection)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Prior-guided epsilon-greedy (two-stage sampling)
        action_idx = None
        if self.is_training:
            r = random.random()
            if r < self.p_prior:
                action_idx = self._expert_greedy_action_idx(rb)
            else:
                r2 = random.random()
                if r2 < self.epsilon:
                    action_idx = random.randint(0, self.action_dim - 1)
        if action_idx is None:
            action_idx = q_values.argmax().item()
        
        # Map DQN action to CQL action
        action_key = self.action_mapping[action_idx]
        
        # Store current state and action for experience replay (if training)
        if self.is_training:
            # Store full components for replay building later
            self.current_action = action_idx
        
        return decision_movement[action_key]
    
    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        """
        CQL interface: Make decision when obstacles are detected
        For DQN, we use the same decision logic but could incorporate obstacle info
        """
        # Incorporate obstacle info into state when available
        # Build state with obstacle features
        state_matrix, grid_distance, obstacle_triplet = self.get_state_representation(rb, obstacle_position)
        state_flat = state_matrix.flatten().astype(np.float32)
        combined_state = np.concatenate([
            state_flat,
            np.array([grid_distance], dtype=np.float32),
            np.array(list(obstacle_triplet), dtype=np.float32)
        ])
        state_tensor = torch.from_numpy(combined_state).to(self.device)
        
        # Q-values for greedy selection unless prior/random kicks in
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        action_idx = None
        if self.is_training:
            r = random.random()
            if r < self.p_prior:
                action_idx = self._expert_greedy_action_idx(rb)
            else:
                r2 = random.random()
                if r2 < self.epsilon:
                    action_idx = random.randint(0, self.action_dim - 1)
        if action_idx is None:
            action_idx = q_values.argmax().item()
        
        action_key = self.action_mapping[action_idx]
        if self.is_training:
            self.current_action = action_idx
        return decision_movement[action_key]
    
    def store_experience(self, state, action_idx, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        # State and next_state come as tuples: (matrix, grid_distance, (c_phi, c_deltaphi, c_deltad))
        if isinstance(state, tuple):
            # support both 2-tuple and 3-tuple for backward compatibility
            if len(state) == 3:
                state_matrix, state_grid_dist, obs_triplet = state
            else:
                state_matrix, state_grid_dist = state
                obs_triplet = (-10, -10, -10)
        else:
            state_matrix, state_grid_dist, obs_triplet = state, 0.0, (-10, -10, -10)
        if isinstance(next_state, tuple):
            if len(next_state) == 3:
                next_state_matrix, next_state_grid_dist, next_obs_triplet = next_state
            else:
                next_state_matrix, next_state_grid_dist = next_state
                next_obs_triplet = (-10, -10, -10)
        else:
            next_state_matrix, next_state_grid_dist, next_obs_triplet = next_state, 0.0, (-10, -10, -10)

        state_vec = np.concatenate([
            state_matrix.flatten().astype(np.float32),
            np.array([float(state_grid_dist)], dtype=np.float32),
            np.array(list(obs_triplet), dtype=np.float32)
        ])
        next_state_vec = np.concatenate([
            next_state_matrix.flatten().astype(np.float32),
            np.array([float(next_state_grid_dist)], dtype=np.float32),
            np.array(list(next_obs_triplet), dtype=np.float32)
        ])

        state_tensor = torch.from_numpy(state_vec).to(self.device)
        next_state_tensor = torch.from_numpy(next_state_vec).to(self.device)
        action = torch.LongTensor([action_idx]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        self.memory.append((state_tensor, action, reward_tensor, next_state_tensor, done_tensor))
    
    def train_step(self):
        """
        Perform one training step using experience replay
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards).squeeze()
        next_states = torch.stack(next_states)
        dones = torch.stack(dones).squeeze()
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Target Q-values using Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, rb, obstacles_list, decision, done, reached_goal, prev_distance=None):
        """
        Calculate reward for current state
        """
        # position = self.convertState(rb)
        
        # # Update position history for anti-trap mechanism
        # if position in self.position_history:
        #     self.position_history[position] += 1
        # else:
        #     self.position_history[position] = 1
        
        # self.position_history_list.append(position)
        
        # if len(self.position_history_list) > self.position_memory_size:
        #     old_pos = self.position_history_list.pop(0)
        #     self.position_history[old_pos] -= 1
        #     if self.position_history[old_pos] <= 0:
        #         del self.position_history[old_pos]
        
        # # Repetition penalty
        # repetition_penalty = min(-2 * (self.position_history[position] - 1), 0)
        
        # # Curiosity reward
        # if position in self.visit_counts:
        #     self.visit_counts[position] += 1
        # else:
        #     self.visit_counts[position] = 1
        
        # curiosity_reward = self.curiosity_factor / max(1, self.visit_counts[position]**0.5)
        
        # Base rewards
        if reached_goal:
            # return 100 + curiosity_reward
            return 15
        
        if done:  # Collision
            # print("oh shit")
            return -50

        
        # Progress reward using CQL distance calculation
        current_state = self.convertState(rb)
        current_distance = self.calculateDistanceToGoal(current_state)

        state_matrix, _, obstacle_triplet = self.get_state_representation(rb)
        c_phi, c_deltaphi, c_deltad = obstacle_triplet

        dynamic_obstacle_penalty = 0
        if c_deltad == 1:  # Obstacle moving away from robot
            dynamic_obstacle_penalty = 3  # Penalty for obstacle moving away
        elif c_deltad == -1:  # Obstacle moving towards robot
            dynamic_obstacle_penalty = -3   # Reward for obstacle moving towards
        elif c_deltad == 0:  # Obstacle maintaining distance
            dynamic_obstacle_penalty = 0   # No penalty

        
        # local_reward = 0.0
        # inner_cells = [(1,1),(1,2),(1,3),(2,1),(2,3),(3,1),(3,2),(3,3)]
        # outer_cells = []
        # for dx in range(-2, 3):
        #     for dy in range(-2, 3):
        #         if dx == 0 and dy == 0:
        #             continue
        #         if max(abs(dx), abs(dy)) == 2:
        #             outer_cells.append((2 + dx, 2 + dy))
        # lambda3 = -1.0  # phạt vật cản (ngoài < trong)
        # lambda4 = 1.0   # thưởng khi thấy đích ở vòng trong
        # local_reward = 0.0
        # if any(state_matrix[y][x] == 1 for (x, y) in outer_cells):
        #     local_reward += lambda3
        # if any(state_matrix[y][x] == 1 for (x, y) in inner_cells):
        #     local_reward += 2 * lambda3
        # if any(state_matrix[y][x] == 3 for (x, y) in inner_cells):
        #     local_reward += lambda4        
        weight = 1 if decision == 0 or decision == 2 or decision == 4 or decision == 6 else 1 / np.sqrt(2)
        # Progress reward for getting closer to goal
        if prev_distance is not None:
            if not np.isfinite(prev_distance) or not np.isfinite(current_distance):
                progress_reward = -10
            else:
                progress_reward = ((prev_distance - current_distance) / np.abs(prev_distance - current_distance + 1e-6)) * weight * 10
                # progress_reward = (prev_distance - current_distance) * 10
                # print(prev_distance, current_distance, progress_reward)
            # step_penalty = -0.1
            
            # Additional penalty for moving away from goal
            # if current_distance > prev_distance:
            #     progress_reward -= 5
            
            # total_reward = progress_reward + step_penalty + repetition_penalty + curiosity_reward
            total_reward = progress_reward + dynamic_obstacle_penalty 
            # print(total_reward)
            return total_reward
        
        # If no previous distance available
        # return -0.1 - (current_distance * 0.001) + repetition_penalty + curiosity_reward
        return -0.1 - (current_distance * 0.001) + dynamic_obstacle_penalty
    
    def process_simulation_step(self, rb, obstacles_list, goal_reached, collision_occurred):
        """
        Main integration point with CQL simulation loop
        Handles complete DQN training cycle - ONLY when action is completed
        """
        self.total_simulation_calls += 1
        
        if not self.is_training:
            return  # Skip training if in test mode
        
        # Handle terminal events immediately, even if mid-step
        if collision_occurred or goal_reached:
            # Get current state at termination
            current_state = self.get_state_representation(rb)
            # Use pixel-based distance for reward consistency
            current_distance = self.calculateDistanceToGoal(self.convertState(rb))
            
            if self.previous_state is not None and self.previous_action is not None:
                reward = self.calculate_reward(
                    rb, obstacles_list, self.current_action,
                    collision_occurred, goal_reached,
                    self.previous_distance
                )
                done = True
                self.store_experience(
                    self.previous_state,
                    self.current_action,
                    reward,
                    current_state,
                    done
                )
                self.train_step()
            
            # Reset episode tracking after terminal
            self.previous_state = None
            self.previous_action = None
            self.previous_distance = None
            self.update_epsilon()
            if self.p_prior > self.p_prior_min:
                self.p_prior = max(self.p_prior * self.p_prior_decay, self.p_prior_min)
            return
        
        # CRITICAL: Only train when robot has completed an action (8 steps)
        # Note: In the simulation loop, makeDecision() runs BEFORE move(), and process_simulation_step()
        # is called AFTER move(). That means when an action just finished, currentStep == numsOfSteps.
        # Checking for 0 would never trigger here because move() increments it to 1 immediately.
        if rb.currentStep != rb.numsOfSteps:
            return  # Skip training during intermediate steps
        
        self.actual_training_calls += 1

        # Get current state (now that action is completed)
        current_state = self.get_state_representation(rb)
        current_distance = self.calculateDistanceToGoal(self.convertState(rb))
        
        # If we have a previous state, create experience and train
        if self.previous_state is not None and self.previous_action is not None:
            # Calculate reward
            reward = self.calculate_reward(
                rb, obstacles_list, 
                self.current_action,
                collision_occurred, goal_reached, 
                self.previous_distance
            )
            
            # Determine if episode is done
            done = collision_occurred or goal_reached
            
            # Store experience (previous_state/current_state already include grid distance)
            self.store_experience(
                self.previous_state, 
                self.current_action, 
                reward,
                current_state, 
                done
            )
            
            # Train the network
            self.train_step()
            
            # If episode ended, don't update previous state
            if done:
                self.previous_state = None
                self.previous_action = None
                self.previous_distance = None
                self.update_epsilon()
                # Decay expert prior after each episode
                if self.p_prior > self.p_prior_min:
                    self.p_prior = max(self.p_prior * self.p_prior_decay, self.p_prior_min)
                return
        
        # Update previous state for next step (only when action is completed)
        self.previous_state = current_state
        self.previous_action = self.current_action  # Set by makeDecision
        self.previous_distance = current_distance
    
    def setCollision(self, rb):
        """
        CQL interface: Called when collision occurs
        NOTE: Training is now handled by process_simulation_step()
        """
        pass  # Training handled in process_simulation_step
    
    def setSuccess(self):
        """
        CQL interface: Called when goal is reached
        NOTE: Training is now handled by process_simulation_step()
        """
        # Epsilon decay is now handled in reset() method
        pass
    
    def reset(self):
        """
        CQL interface: Reset for new episode
        """
        # Clear episode-specific data
        self.episode_experiences = []
        
        # Reset position tracking
        self.position_history = {}
        self.position_history_list = []
        
        # Reset state tracking for new episode
        self.previous_state = None
        self.previous_action = None
        self.previous_distance = None
        self.current_state = None
        self.current_action = None
        
        
        # Reset counters for new episode
        self.total_simulation_calls = 0
        self.actual_training_calls = 0
        self.p_prior = 0.9
        self.p_prior_decay = 0.998
        self.p_prior_min = 0.0
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.batch_size = 64
        self.memory.clear()
        self.target_update_freq = 100
        self.step_count = 0
    
    def outputPolicy(self, scenario, current_map, run_index):
        """
        CQL interface: Save trained model
        """
        # Save neural network weights
        model_path = f"policy/{scenario}/{current_map}/DQN/{run_index}/dqn_model.pth"
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, model_path)
        
        print(f"DQN model saved to {model_path}")
        
        # Also save a dummy policy file for compatibility with CQL system
        policy_path = f"policy/{scenario}/{current_map}/DQN/{run_index}/policy.json"
        dummy_policy = [{"key": [0, 0], "value": "stay"}]  # Dummy entry
        
        with open(policy_path, "w") as f:
            json.dump(dummy_policy, f)
    
    def update_epsilon(self):
        """
        Update epsilon after each epoch (called from Simulation.py)
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load_model(self, model_path):
        """
        Load trained DQN model for testing
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.0)  # Set to 0 for testing
            self.step_count = checkpoint.get('step_count', 0)
            
            # Set to evaluation mode
            self.q_network.eval()
            self.target_network.eval()
            self.is_training = False
            
            print(f"DQN model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def set_training_mode(self, is_training=True):
        """
        Set training/testing mode
        """
        self.is_training = is_training
        if is_training:
            self.q_network.train()
        else:
            self.q_network.eval()
            self.epsilon = 0.0  # No exploration during testing


# Alias for compatibility with CQL naming convention
QLearning = DQNLearning
