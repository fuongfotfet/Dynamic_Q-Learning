import numpy as np
from controller.Controller import Controller
import random
import heapq
import math

def compute_distances(grid, goal_i, goal_j):
    M = len(grid)
    INF = float('inf')
    dist = [[INF]*M for _ in range(M)]
    dist[goal_j][goal_i] = 0.0

    # mỗi entry: (current_cost, i, j)
    heap = [(0.0, goal_i, goal_j)]
    # các hướng và cost tương ứng
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

# A* initializer ---
def astar(grid, start, goal):
    M = len(grid)
    INF = M*M + 1
    g = [[INF]*M for _ in range(M)]
    visited = [[False]*M for _ in range(M)]
    si, sj = start
    gi, gj = goal
    def h(i, j):
        return max(abs(i-gi), abs(j-gj))
    g[sj][si] = 0
    heap = [(h(si, sj), 0, si, sj)]
    while heap:
        f, cost, i, j = heapq.heappop(heap)
        if visited[j][i]:
            continue
        visited[j][i] = True
        if (i, j) == (gi, gj):
            return cost
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < M and 0 <= nj < M and grid[nj][ni] == 0 and not visited[nj][ni]:
                new_cost = cost + 1
                if new_cost < g[nj][ni]:
                    g[nj][ni] = new_cost
                    heapq.heappush(heap, (new_cost + h(ni,nj), new_cost, ni, nj))
    return INF


class DynamicWindowApproach(Controller):
    def __init__(self, goal, cell_size, env_size, env_padding, max_speed_ratio=1.0, max_turn_rate=np.pi/4, static_obstacles=None):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles=static_obstacles)
        self.max_speed_ratio = max_speed_ratio
        self.max_turn_rate = max_turn_rate
        self.goal = goal
        self.static_obstacles = static_obstacles if static_obstacles else []
        self.dt = 0.1  # Time step for simulation
        self.predict_time = 0.8  # How far ahead to predict (8 steps * 0.1)
        
        # Anti-stuck mechanism
        self.position_history = []  # Track recent positions
        self.stuck_counter = 0
        self.max_history_size = 20  # Keep last 20 positions
        self.stuck_threshold = 15  # If robot stays in same area for 15 decisions, it's stuck
        self.exploration_boost = 0.0  # Boost exploration when stuck
        self.last_decision = (0, 0)  # Track last decision to avoid repetition
        
        # Initialize distance to goal using pathfinding
        self._initialize_distance_to_goal()

    def _initialize_distance_to_goal(self):
        """Initialize distance to goal using A*/Dijkstra like in CombinedQL and DualQL"""
        # 1) Build static occupancy grid
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

        # 2) Compute distances from each cell to goal using Dijkstra
        goal_i = int((self.goal[0] - self.env_padding)//self.cell_size)
        goal_j = int((self.goal[1] - self.env_padding)//self.cell_size)
        
        # Ensure goal is within bounds
        goal_i = max(0, min(M-1, goal_i))
        goal_j = max(0, min(M-1, goal_j))
        
        self.dist_to_goal = compute_distances(grid, goal_i, goal_j)
        
        print(f"DWA: Initialized distance grid. Goal at ({goal_i}, {goal_j}). Grid size: {M}x{M}")

    def _get_optimal_heading(self, rb):
        """Calculate optimal robot heading based on gradient of distance field"""
        # Get current position in grid coordinates
        current_i = int((rb.pos[0] - self.env_padding) / self.cell_size)
        current_j = int((rb.pos[1] - self.env_padding) / self.cell_size)
        
        # Ensure within bounds
        M = len(self.dist_to_goal)
        current_i = max(0, min(M-1, current_i))
        current_j = max(0, min(M-1, current_j))
        
        # Find the best neighboring direction (gradient descent on distance field)
        best_direction = None
        best_distance = self.dist_to_goal[current_j][current_i]
        
        # Check all 8 neighbors
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # cardinal
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal
        ]
        
        for di, dj in directions:
            ni, nj = current_i + di, current_j + dj
            if 0 <= ni < M and 0 <= nj < M:
                neighbor_dist = self.dist_to_goal[nj][ni]
                if neighbor_dist < best_distance:
                    best_distance = neighbor_dist
                    best_direction = (di, dj)
        
        # If found a better direction, use it
        if best_direction is not None:
            # Convert grid direction to angle
            dx, dy = best_direction
            return np.arctan2(dy, dx)
        else:
            # Fallback: direct heading to goal
            goal_dx = self.goal[0] - rb.pos[0]
            goal_dy = self.goal[1] - rb.pos[1]
            return np.arctan2(goal_dy, goal_dx)

    def makeDecision(self, rb) -> tuple:
        # Make decision with no dynamic obstacles
        return self.makeObstacleDecision(rb, [])

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        current_pos = (rb.pos[0], rb.pos[1])
        
        # Update position history
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
        
        # Check if robot is stuck (moving in small area)
        is_stuck = self._detect_stuck_behavior(current_pos)
        
        # If stuck, increase exploration and try different strategies
        if is_stuck:
            self.stuck_counter += 1
            self.exploration_boost = min(self.stuck_counter * 0.1, 1.0)  # Gradually increase exploration
            print(f"Robot stuck detected! Counter: {self.stuck_counter}, Exploration boost: {self.exploration_boost}")
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)  # Gradually decrease counter
            self.exploration_boost = max(0, self.exploration_boost - 0.05)
        
        # Generate feasible velocity commands
        # Include both forward and backward movement
        valid_speeds = np.linspace(-self.max_speed_ratio * 0.5, self.max_speed_ratio, num=8)  # Allow backing up
        valid_turn_rates = np.linspace(-self.max_turn_rate, self.max_turn_rate, num=11)  # More turning options

        # If very stuck, add random exploration
        if self.stuck_counter > 10:
            # Add some random velocity commands for exploration
            random_speeds = [random.uniform(-0.5, 1.0) for _ in range(3)]
            random_turns = [random.uniform(-self.max_turn_rate, self.max_turn_rate) for _ in range(3)]
            valid_speeds = np.concatenate([valid_speeds, random_speeds])
            valid_turn_rates = np.concatenate([valid_turn_rates, random_turns])

        # Initialize best commands
        best_cmd = (0, 0)
        best_score = float('-inf')
        robot_pose = (rb.pos[0], rb.pos[1])
        
        # Get robot heading based on gradient of distance field (like in QL algorithms)
        robot_heading = self._get_optimal_heading(rb)

        # Iterate through feasible velocity commands
        for v in valid_speeds:
            for w in valid_turn_rates:
                # Skip if this is exactly the same as last decision (avoid repetition)
                if self.stuck_counter > 5 and abs(v - self.last_decision[0]) < 0.1 and abs(w - self.last_decision[1]) < 0.1:
                    continue
                
                # Simulate motion
                simulated_trajectory = self.simulate_motion(robot_pose, robot_heading, v, w)

                # Evaluate trajectory with stuck awareness
                score = self.evaluate_trajectory(simulated_trajectory, obstacle_position, rb.vision * 0.3, is_stuck)

                # Add exploration bonus if stuck
                if is_stuck:
                    exploration_bonus = self.exploration_boost * random.uniform(0.5, 1.5)
                    score += exploration_bonus

                # Update best commands if score is better
                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)
        
        # Debug: print if all trajectories are rejected
        if best_score == float('-inf'):
            print(f"Warning: All trajectories rejected at pos {robot_pose}")
            # Enhanced fallback strategy
            if self.stuck_counter > 20:
                # Very stuck - try drastic measures
                best_cmd = (random.uniform(-0.5, 0.5), random.uniform(-self.max_turn_rate, self.max_turn_rate))
            else:
                # Fallback: try simple goal-directed movement with very small steps
                goal_distance = self.calculateDistanceToGoal(self.convertState(rb))
                if goal_distance > self.cell_size:
                    best_cmd = (0.2, 0)  # Very small forward movement
                else:
                    # Near goal, try different directions
                    best_cmd = (0.3, self.max_turn_rate * 0.5)  # Turn and move

        # Store last decision
        self.last_decision = best_cmd

        # Calculate direction based on best command
        # Use the velocity and angular velocity to compute movement
        v, w = best_cmd
        if v == 0:
            return (0, 0)  # No movement
        
        # Calculate the direction of movement
        final_heading = robot_heading + w * self.dt
        
        # Convert to cell units (not pixels) and scale appropriately
        # Since this direction will be used for 8 steps, we need to scale it down
        direction = (v * np.cos(final_heading), 
                    v * np.sin(final_heading))

        return direction

    def _detect_stuck_behavior(self, current_pos):
        """Detect if robot is stuck in repetitive movement patterns"""
        if len(self.position_history) < self.stuck_threshold:
            return False
        
        # Check if robot has been moving in a small area
        recent_positions = self.position_history[-self.stuck_threshold:]
        
        # Calculate the area covered by recent positions
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # If robot has been moving in area smaller than 2 cells, it's stuck
        stuck_area_threshold = self.cell_size * 2
        if x_range < stuck_area_threshold and y_range < stuck_area_threshold:
            return True
        
        # Check for oscillation pattern (moving back and forth)
        if len(recent_positions) >= 6:
            # Check if robot is oscillating between similar positions
            pos_diffs = []
            for i in range(1, len(recent_positions)):
                diff = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                pos_diffs.append(diff)
            
            # If movement distances are very small, robot is stuck
            avg_movement = np.mean(pos_diffs)
            if avg_movement < self.cell_size * 0.5:
                return True
        
        return False

    def simulate_motion(self, pose, heading, v, w):
        # Simulate robot motion given current pose, heading and velocity commands
        trajectory = []
        x, y, theta = pose[0], pose[1], heading
        
        # Simulate trajectory for predict_time duration
        steps = int(self.predict_time / self.dt)
        for _ in range(steps):
            # Update position and heading
            # Scale movement to match robot's 8-step system
            # v = 1.0 means 1 cell movement over 8 steps
            step_distance = (v * self.cell_size) / 8.0
            x += step_distance * np.cos(theta)
            y += step_distance * np.sin(theta)
            theta += w * self.dt
            trajectory.append([x, y])
        
        return trajectory

    def evaluate_trajectory(self, trajectory, dynamic_obstacles, safety_distance, is_stuck=False):
        if not trajectory:
            return float('-inf')
        
        # Score based on final position to goal using pathfinding distance
        final_pos = trajectory[-1]
        
        # Convert final position to grid coordinates
        final_i = int((final_pos[0] - self.env_padding) / self.cell_size)
        final_j = int((final_pos[1] - self.env_padding) / self.cell_size)
        
        # Get pathfinding distance using calculateDistanceToGoal from Controller
        final_state = (final_i, final_j)
        distance_to_goal = self.calculateDistanceToGoal(final_state)
        
        # Increase goal attraction when stuck
        goal_weight = 15.0 if is_stuck else 10.0
        if distance_to_goal == float('inf'):
            score = -1000  # Unreachable position
        else:
            score = goal_weight / (distance_to_goal + 1)  # Reward getting closer to goal
        
        # Add bonus for unexplored areas when stuck
        if is_stuck and len(self.position_history) > 5:
            # Bonus for moving away from recent positions
            recent_positions = self.position_history[-5:]
            min_dist_to_recent = min([np.linalg.norm(np.array(final_pos) - np.array(pos)) for pos in recent_positions])
            if min_dist_to_recent > self.cell_size:
                score += 5.0  # Bonus for exploration
        
        # Robot radius for collision detection
        robot_radius = 8  # From Robot.py
        
        # Check collision with static obstacles
        for point in trajectory:
            # Check boundaries with robot radius
            if (point[0] < self.env_padding + robot_radius or 
                point[0] > self.env_padding + self.env_size - robot_radius or
                point[1] < self.env_padding + robot_radius or 
                point[1] > self.env_padding + self.env_size - robot_radius):
                return float('-inf')  # Out of bounds
            
            # Check static obstacles with robot radius
            for obs in self.static_obstacles:
                obs_x1, obs_x2, obs_y1, obs_y2 = obs.return_coordinate()
                # Check if robot circle intersects with obstacle rectangle
                closest_x = max(obs_x1, min(point[0], obs_x2))
                closest_y = max(obs_y1, min(point[1], obs_y2))
                dist_to_static = np.sqrt((closest_x - point[0])**2 + (closest_y - point[1])**2)
                
                if dist_to_static < robot_radius:
                    return float('-inf')  # Collision with static obstacle
                
                # Reduced penalty for getting close to static obstacles when stuck
                penalty_weight = 1.0 if is_stuck else 2.0
                if dist_to_static < safety_distance:
                    score -= penalty_weight / (dist_to_static + 1)  # Reduced penalty when stuck
        
        # Check collision with dynamic obstacles
        for obstacle in dynamic_obstacles:
            for point in trajectory:
                # Assuming obstacle format: [x, y, width, height]
                if len(obstacle) >= 4:
                    closest_x = max(obstacle[0] - obstacle[2] / 2, min(point[0], obstacle[0] + obstacle[2] / 2))
                    closest_y = max(obstacle[1] - obstacle[3] / 2, min(point[1], obstacle[1] + obstacle[3] / 2))
                    dist_to_obstacle = np.sqrt((closest_x - point[0])**2 + (closest_y - point[1])**2)
                    
                    if dist_to_obstacle < robot_radius:
                        return float('-inf')  # Collision
                    
                    if dist_to_obstacle < safety_distance:
                        score -= 5.0 / (dist_to_obstacle + 1)
        
        return score
