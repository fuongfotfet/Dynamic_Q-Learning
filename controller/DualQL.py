from collections import deque
import json
import time
import numpy as np
import random
from controller.Controller import (Controller, action_space, decision_movement, convertphi, convertdeltaphi,
                                   convertdeltad, angle, remap_keys, find_octant)
import heapq, math

def compute_distances(grid, goal_i, goal_j):
    M = len(grid)
    INF = float('inf')
    dist = [[INF]*M for _ in range(M)]
    dist[goal_j][goal_i] = 0.0
    heap = [(0.0, goal_i, goal_j)]
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


# Hyperparameters
GAMMA = 0.8  # 0.8 to 0.9

EPSILON = 0.3
EPSILON_DECAY = 0.95

EPSILON_OBS = 0.8
EPSILON_OBS_DECAY = 0.99

ALPHA = 0.9  # 0.2 to 0.9
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -5
successReward = 15

class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles=None):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles=static_obstacles)
        self.Qtable = {}
        self.obstacleQtable = {}
        self.obstaclePolicy = {}
        self.obstacleEpisodeDecisions = []
        self.episodeDecisions = []
        self.sumOfRewards = []
        self.averageReward = []
        self.isObstacleDecisionMade = False
        self.safe_radius_cells = 2  
        self.reset()

    def reset(self) -> None:
        global EPSILON, EPSILON_OBS
        EPSILON = 0.3
        EPSILON_OBS = 0.8

        self.episodeDecisions.clear()
        self.obstacleEpisodeDecisions.clear()
        self.sumOfRewards.clear()
        self.averageReward.clear()

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

        # 2) Precompute dist via A* from each free cell to goal
        goal_i = int((self.goal[0] - self.env_padding)//self.cell_size)
        goal_j = int((self.goal[1] - self.env_padding)//self.cell_size)
        self.dist_to_goal = compute_distances(grid, goal_i, goal_j)

        dist = [[float('inf')]*M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                if grid[j][i] == 0:
                    dist[j][i] = astar(grid, (i, j), (goal_i, goal_j))

        # 3) Initialize policy & Qtable to drive around static walls
        for i in range(M):
            for j in range(M):
                if grid[j][i] == 1:
                    continue  
                best_act = None
                best_d = dist[j][i]
                for (di, dj), action in [((1,0),"right_1"),((-1,0),"left_1"),
                                        ((0,1),"down_1"),((0,-1),"up_1"),
                                        ((1,1),"down-right_1"),((1,-1),"up-right_1"),
                                        ((-1,1),"down-left_1"),((-1,-1),"up-left_1")]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < M and 0 <= nj < M and grid[nj][ni] == 0 and dist[nj][ni] < best_d:
                        best_d = dist[nj][ni]
                        best_act = action
                if best_act is None:
                    cx = self.env_padding + self.cell_size/2 + i*self.cell_size
                    cy = self.env_padding + self.cell_size/2 + j*self.cell_size
                    dx, dy = self.goal[0] - cx, self.goal[1] - cy
                    ratio = np.tan(np.pi / 8)
                    if abs(dy) > ratio * abs(dx):
                        best_act = "down_1" if dy > 0 else "up_1"
                        if abs(dx) > ratio * abs(dy):
                            best_act = "down-right_1" if dy > 0 and dx > 0 else "down-left_1" if dy > 0 else "up-right_1" if dx > 0 else "up-left_1"
                    else:
                        best_act = "right_1" if dx > 0 else "left_1"

                ini_val = 1.0 / dist[j][i] if 0 < dist[j][i] < float('inf') else 0.0

                self.policy[(i, j)] = best_act
                for act in action_space:
                    self.Qtable[(i, j, act)] = ini_val
        
        for i in range(M):
            for j in range(M):
                if (i, j) not in self.policy:
                    self.policy[(i, j)] = "up_1" 

                for act in action_space:
                    if (i, j, act) not in self.Qtable:
                        self.Qtable[(i, j, act)] = 0.0

        # Initialize ObstacleQtable and obstaclePolicy
        for c_phi in range(3):                    
            for c_deltaphi in range(-2, 3):       
                for c_deltad in range(-1, 2):     
                    for goal_direction in range(8):          
                        for nearest_obs_octant in range(8):  
                            for nearest_obs_dist_bin in range(3):  
                                if goal_direction == 0:
                                    default = "right_1"
                                elif goal_direction == 1:
                                    default = "down-right_1"
                                elif goal_direction == 2:
                                    default = "down_1"
                                elif goal_direction == 3:
                                    default = "down-left_1"
                                elif goal_direction == 4:
                                    default = "left_1"
                                elif goal_direction == 5:
                                    default = "up-left_1"
                                elif goal_direction == 6:
                                    default = "up_1"
                                else:  
                                    default = "up-right_1"

                                self.obstaclePolicy[(c_phi, c_deltaphi, c_deltad, goal_direction, nearest_obs_octant, nearest_obs_dist_bin)] = default

                                for action in action_space:
                                    self.obstacleQtable[(c_phi, c_deltaphi, c_deltad, goal_direction, nearest_obs_octant, nearest_obs_dist_bin, action)] = 0.0

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self, rb) -> None:
        global EPSILON_OBS
        EPSILON_OBS *= EPSILON_OBS_DECAY
        # Add collision discount to the "normal" decision if it made the decision which caused the collision
        if not self.isObstacleDecisionMade:
            if len(self.episodeDecisions) > 0:
                state, decision, reward = self.episodeDecisions[-1]
                reward += collisionDiscount

                self.episodeDecisions.pop()
                self.episodeDecisions.append((state, decision, reward))
                self.episodeDecisions.append((self.convertState(rb), "", 0))
            if len(self.episodeDecisions) >= 3:
                for i in range(-3, -1):
                    s, a, r = self.episodeDecisions[i]
                    ns    = self.episodeDecisions[i+1][0]
                    self.updateQtable(s, a, r, ns)
                    self.updatePolicy(s)
            else:
                for i in range(len(self.episodeDecisions)-1):
                    s, a, r = self.episodeDecisions[i]
                    ns    = self.episodeDecisions[i+1][0]
                    self.updateQtable(s, a, r, ns)
                    self.updatePolicy(s)
        # Else add collision discount to the obstacle decision
        else:
            if len(self.obstacleEpisodeDecisions) > 0:
                state, decision, reward = self.obstacleEpisodeDecisions[-1]
                reward += collisionDiscount - 500  # Add more if the obstacle Q-table made the collision
 
                self.obstacleEpisodeDecisions.pop()
                self.obstacleEpisodeDecisions.append((state, decision, reward))
                self.obstacleEpisodeDecisions.append(((0, 0, 0, 0, 0, 2), "", 0))

            if len(self.obstacleEpisodeDecisions) >= 3:
                for i in range(-3, -1):
                    s, a, r = self.obstacleEpisodeDecisions[i]
                    ns    = self.obstacleEpisodeDecisions[i+1][0]
                    self.updateObstacleQtable(s, a, r, ns)
                    self.updateObstaclePolicy(s)
            else:
                for i in range(len(self.obstacleEpisodeDecisions)-1):
                    s, a, r = self.obstacleEpisodeDecisions[i]
                    ns    = self.obstacleEpisodeDecisions[i+1][0]
                    self.updateObstacleQtable(s, a, r, ns)
                    self.updateObstaclePolicy(s)

        # Update Qtable and policy for both Q-Learning after collision
        self.isObstacleDecisionMade = False
        self.updateAll()
        self.isObstacleDecisionMade = True
        self.updateAll()

        self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()
        self.obstacleEpisodeDecisions.clear()

    # Add success reward to the last decision if the robot has reached the goal
    def setSuccess(self) -> None:
        # Decay epsilon
        global EPSILON
        EPSILON *= EPSILON_DECAY

        # Always add success reward to the last "normal" decision if success
        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += successReward

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))

            goal_pos = (int((self.goal[0] - self.env_padding) / self.cell_size),
                        int((self.goal[1] - self.env_padding) / self.cell_size))
            self.episodeDecisions.append((goal_pos, "", 0))

        # Add success reward to the last obstacle decision if it makes the final decision
        # In other words, the reward have not been added by func makeDecision
        if self.isObstacleDecisionMade:
            if len(self.obstacleEpisodeDecisions) > 0:
                state, decision, reward = self.obstacleEpisodeDecisions[-1]
                reward += successReward

                self.obstacleEpisodeDecisions.pop()
                self.obstacleEpisodeDecisions.append((state, decision, reward))
                self.obstacleEpisodeDecisions.append(((0, 0, 0, 0, 0, 2), "", 0))

        # Update Qtable and policy for both Q-Learning after success
        self.isObstacleDecisionMade = False
        self.updateAll()
        self.isObstacleDecisionMade = True
        self.updateAll()

        self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()
        self.obstacleEpisodeDecisions.clear()

    # Calculate the sum of rewards and the average reward of the episode after the episode ends
    def calculateReward(self) -> None:
        sumOfReward = 0
        for episodeDecision in self.episodeDecisions:
            sumOfReward += episodeDecision[2]

        for obstacleEpisodeDecision in self.obstacleEpisodeDecisions:
            sumOfReward += obstacleEpisodeDecision[2]

        self.sumOfRewards.append(sumOfReward)
        self.averageReward.append(sumOfReward / (len(self.episodeDecisions) + len(self.obstacleEpisodeDecisions) + 1e-6))

    # Output policy to json file
    def outputPolicy(self, scenario, current_map, run_index) -> None:
        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/obstaclePolicy.json", "w") as outfile:
            json.dump(remap_keys(self.obstaclePolicy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state) -> None:
        # Optimal value of next state
        optimalQnext = max([self.Qtable[(next_state[0], next_state[1], action)] for action in action_space])

        # Update Qtable
        self.Qtable[(state[0], state[1], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

    def updatePolicy(self, state) -> None:
        # Update policy
        bestAction = max(action_space,
                         key=lambda action: self.Qtable[(state[0], state[1], action)])

        self.policy[state] = bestAction

    def updateObstacleQtable(self, state, decision, reward, next_state) -> None:
        # Optimal value of next state
        optimalQnext = max([self.obstacleQtable[(next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5], action)] for action in action_space])

        # Update Qtable
        self.obstacleQtable[(state[0], state[1], state[2], state[3], state[4], state[5], decision)] = (1 - ALPHA) * self.obstacleQtable[(state[0], state[1], state[2], state[3], state[4], state[5], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

    def updateObstaclePolicy(self, state) -> None:
        # state là 6-tuple: (c_phi, c_deltaphi, c_deltad, goal_direction, nearest_obs_octant, nearest_obs_dist_bin)
        bestAction = max(action_space,key=lambda action: self.obstacleQtable[(state[0], state[1], state[2], state[3], state[4], state[5], action)])
        self.obstaclePolicy[state] = bestAction

    def updateAll(self) -> None:
        # If the last decision is a "normal" decision, update the Qtable and policy of the "normal" decision
        if not self.isObstacleDecisionMade:
            if len(self.episodeDecisions) >= 2:
                state, decision, reward = self.episodeDecisions[-2]
                next_state = self.episodeDecisions[-1][0]

                # Update Qtable
                self.updateQtable(state, decision, reward, next_state)

                # Update policy
                self.updatePolicy(state)
        # Else update the Qtable and policy of the obstacle decision
        else:
            if len(self.obstacleEpisodeDecisions) >= 2:
                state, decision, reward = self.obstacleEpisodeDecisions[-2]
                next_state = self.obstacleEpisodeDecisions[-1][0]

                # Update Qtable
                self.updateObstacleQtable(state, decision, reward, next_state)

                # Update policy
                self.updateObstaclePolicy(state)

    def makeDecision(self, rb) -> tuple:
        # Add reward to obstacle decision after successfully avoiding the obstacle
        if self.isObstacleDecisionMade and len(self.obstacleEpisodeDecisions) > 0:
            self.obstacleEpisodeDecisions[-1] = (
                self.obstacleEpisodeDecisions[-1][0], self.obstacleEpisodeDecisions[-1][1],
                self.obstacleEpisodeDecisions[-1][2] + successReward)

        # Update Qtable and policy for the last decision
        self.updateAll()

        # Reset the obstacle decision flag
        self.isObstacleDecisionMade = False

        state = self.convertState(rb)

        # Epsilon greedy
        # Randomly choose an action
        if random.random() < EPSILON:
            decision = random.choice(action_space)
        # Choose the best action
        else:
            decision = self.policy[state]

        # Calculate reward
        distance = self.calculateDistanceToGoal(state)

        movement = decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self.calculateDistanceToGoal(next_state)

        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)

        # Reward is the change in distance to the goal (negative reward if the robot is moving away from the goal)
        if not np.isfinite(distance) or not np.isfinite(next_distance):
            reward = -100 
        else:
            reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight

        # Add to episode decisions
        self.episodeDecisions.append((state, decision, reward))

        return decision_movement[decision]

    
    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        self.updateAll()
        self.isObstacleDecisionMade = True

        obstacle_before = obstacle_position[0]
        obstacle_after  = obstacle_position[1]

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0],
                        obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0],
                        obstacle_after[1] - rb.pos[1])
        
        # Convert to state
        c_phi      = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)
        c_deltad = convertdeltad(distance_to_obstacle_next - distance_to_obstacle)
        goal_direction = find_octant(rb.pos[0], rb.pos[1], self.goal)

        # Find nearest static wall point
        min_dist_static = float('inf')
        nearest_obs_octant = 0
        nearest_obs_dist_bin = 2

        for obs in self.static_obstacles:
            x1, x2, y1, y2 = obs.return_coordinate()
            closest_x = min(max(rb.pos[0], x1), x2)
            closest_y = min(max(rb.pos[1], y1), y2)
            d = math.hypot(rb.pos[0] - closest_x, rb.pos[1] - closest_y)
            if d < min_dist_static:
                min_dist_static = d
                nearest_point = (closest_x, closest_y)

        # Angle to wall
        nearest_obs_octant = find_octant(rb.pos[0], rb.pos[1], nearest_point)

        # Distance to goal
        r = self.safe_radius_cells * self.cell_size
        if min_dist_static <= r:
            nearest_obs_dist_bin = 0
        elif min_dist_static <= 2 * r:
            nearest_obs_dist_bin = 1
        else:
            nearest_obs_dist_bin = 2

        state = (c_phi, c_deltaphi, c_deltad, goal_direction, nearest_obs_octant, nearest_obs_dist_bin)

        # Epsilon‐greedy 
        if random.random() < EPSILON_OBS:
            decision = random.choice(action_space)
        else:
            decision = self.obstaclePolicy[state]

        # Distance to goal
        movement = decision_movement[decision]
        curr_pos = self.convertState(rb)
        distance = self.calculateDistanceToGoal(curr_pos)
        next_state = (curr_pos[0] + movement[0], curr_pos[1] + movement[1])   
        next_distance = self.calculateDistanceToGoal(next_state)

        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1.0
        else:
            weight = 1.0 / math.sqrt(2)

        # Reward shaping
        if not np.isfinite(distance) or not np.isfinite(next_distance):
            reward = -100  
        else:
            reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight

        self.obstacleEpisodeDecisions.append((state, decision, reward))

        return decision_movement[decision]
