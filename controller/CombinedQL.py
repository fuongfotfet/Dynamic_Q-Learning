import json
import numpy as np
import random
from controller.Controller import (Controller, action_space, decision_movement, convertphi, convertdeltaphi,
                                   convertdeltad, angle, remap_keys)
import heapq  
from collections import deque
import math

def get_opposite_action(action):
    mapping = {
        "up_1": "down_1", "down_1": "up_1",
        "left_1": "right_1", "right_1": "left_1",
        "up-left_1": "down-right_1", "down-right_1": "up-left_1",
        "up-right_1": "down-left_1", "down-left_1": "up-right_1"
    }
    return mapping.get(action, None)

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


# Hyperparameters
GAMMA = 0.8  # 0.8 to 0.9

EPSILON = 0.5
EPSILON_DECAY = 0.9

ALPHA = 0.9  # 0.2 to 0.9
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -50
successReward = 15


class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles=None):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles=static_obstacles)
        self.goal = goal
        self.Qtable = {}
        self.episodeDecisions = []
        self.sumOfRewards = []
        self.averageReward = []

        self.reset()

    def reset(self) -> None:
        global EPSILON
        EPSILON = 0.5

        self.episodeDecisions.clear()
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
        dist = [[float('inf')]*M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                if grid[j][i] == 0:
                    dist[j][i] = astar(grid, (i, j), (goal_i, goal_j))
        
        self.dist_to_goal = compute_distances(grid, goal_i, goal_j)

        # 3) Initialize policy & Qtable to drive around static walls
        for i in range(M):
            for j in range(M):
                if grid[j][i] == 1:
                    best_act = "up_1" 
                # find neighbor that decreases dist
                best_act = None
                best_d = dist[j][i]
                for (di, dj), action in [((1,0),"right_1"),((-1,0),"left_1"),
                                        ((0,1),"down_1"),((0,-1),"up_1"),
                                        ((1,1),"down-right_1"),((1,-1),"up-right_1"),
                                        ((-1,1),"down-left_1"),((-1,-1),"up-left_1")]:
                    ni, nj = i+di, j+dj
                    if 0<=ni<M and 0<=nj<M and grid[nj][ni]==0 and dist[nj][ni]<best_d:
                        best_d = dist[nj][ni]
                        best_act = action
                # fallback: direct-to-goal if no neighbor improves
                if best_act is None:
                    cx = self.env_padding + self.cell_size/2 + i*self.cell_size
                    cy = self.env_padding + self.cell_size/2 + j*self.cell_size
                    dx, dy = self.goal[0]-cx, self.goal[1]-cy
                    ratio = np.tan(np.pi/8)
                    dec = ""
                    if abs(dy)>ratio*abs(dx):
                        dec += "down" if dy>0 else "up"
                        if abs(dx)>ratio*abs(dy):
                            dec += "-right" if dx>0 else "-left"
                    else:
                        dec += "right" if dx>0 else "left"
                    best_act = dec + "_1"

                # initial Q-value = 1/dist (or 0 if unreachable)
                ini_val = 1.0/dist[j][i] if 0<dist[j][i]<float('inf') else 0.0

                # assign to all static‐state Q & policy
                for phi in range(3):
                    for dphi in range(-2,3):
                        for dd in range(-1,2):
                            st = (i,j,phi,dphi,dd)
                            self.policy[st] = best_act
                            for act in action_space:
                                self.Qtable[st+(act,)] = ini_val
                # no‐obstacle state
                st0 = (i,j,-10,-10,-10)
                self.policy[st0] = best_act
                for act in action_space:
                    self.Qtable[st0+(act,)] = ini_val

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += collisionDiscount

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))

            next_state = self.convertState(rb)
            self.episodeDecisions.append(((next_state[0], next_state[1], -10, -10, -10), "", 0))

        # Experience Replay
        if len(self.episodeDecisions) >= 3:
            for i in range(-3, -1):
                s, a, r = self.episodeDecisions[i]
                ns = self.episodeDecisions[i + 1][0]
                self.updateQtable(s, a, r, ns)
                self.updatePolicy(s)
        else:
            self.updateAll()

        self.calculateReward()

        self.episodeDecisions.clear()

    # Add success reward to the last decision if the robot has reached the goal
    def setSuccess(self) -> None:
        # Decay epsilon
        global EPSILON
        EPSILON *= EPSILON_DECAY

        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += successReward

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))

            goal_pos = (int((self.goal[0] - self.env_padding) / self.cell_size),
                        int((self.goal[1] - self.env_padding) / self.cell_size))
            self.episodeDecisions.append(((goal_pos[0], goal_pos[1], -10, -10, -10), "", 0))

            # Update Qtable and policy after success
            self.updateAll()
            self.calculateReward()

        # Clear episode decisions
        self.episodeDecisions.clear()

    # Calculate the sum of rewards and average reward of the episode after the episode ends
    def calculateReward(self) -> None:
        sumOfReward = 0
        for episodeDecision in self.episodeDecisions:
            sumOfReward += episodeDecision[2]

        self.sumOfRewards.append(sumOfReward)
        self.averageReward.append(sumOfReward / (len(self.episodeDecisions) + 1e-6))

    # Out put policy to json file
    def outputPolicy(self, scenario, current_map, run_index) -> None:
        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/CombinedQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state) -> None:
        # Optimal value of next state
        optimalQnext = max([self.Qtable[(next_state[0], next_state[1], next_state[2], next_state[3],
                                         next_state[4], action)] for action in action_space])

        # Update Qtable
        self.Qtable[(state[0], state[1], state[2], state[3], state[4], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], state[2], state[3], state[4], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

    def updatePolicy(self, state) -> None:
        # Update policy
        bestAction = max(action_space, key=lambda action: self.Qtable[(state[0], state[1], state[2], state[3], state[4], action)])

        self.policy[state] = bestAction

    def updateAll(self):
        for i in range(len(self.episodeDecisions) - 1):
            state, decision, reward = self.episodeDecisions[i]
            next_state = self.episodeDecisions[i + 1][0]
            self.updateQtable(state, decision, reward, next_state)
            self.updatePolicy(state)

    def makeDecision(self, rb) -> tuple:
        self.updateAll()

        state = self.convertState(rb)
        state = (state[0], state[1], -10, -10, -10)

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
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0],
                    obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0],
                         obstacle_after[1] - rb.pos[1])

        # Convert to state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))

        state = self.convertState(rb)
        state = (state[0], state[1], c_phi, c_deltaphi, c_deltad)
        
        # No cycle 
        last_action = self.episodeDecisions[-1][1] if self.episodeDecisions else None
        forbidden = get_opposite_action(last_action)
        candidates = [a for a in action_space if a != forbidden] if forbidden else action_space

        decision = max(candidates, key=lambda a: self.Qtable[state + (a,)])

        # Goal reward
        distance = self.calculateDistanceToGoal(state)
        movement = decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self.calculateDistanceToGoal(next_state)

        weight = 1 if decision in ["up_1", "down_1", "left_1", "right_1"] else 1 / np.sqrt(2)
        # Reward is the change in distance to the goal
        if not np.isfinite(distance) or not np.isfinite(next_distance):
            reward = -100 
        else:
            reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight
            print(reward)

        self.episodeDecisions.append((state, decision, reward))
        return decision_movement[decision]