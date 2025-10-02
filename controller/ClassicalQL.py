import json
import numpy as np
import random
from controller.Controller import Controller, action_space, remap_keys, decision_movement
import heapq
import math

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

# This is the implementation of the Q-Learning algorithm
# Jiang, Qi. "Path planning method of mobile robot based on Q-learning."
# Journal of Physics: Conference Series. Vol. 2181. No. 1. IOP Publishing, 2022.

# Hyperparameters
GAMMA = 0.9  # 0.8 to 0.9

EPSILON = 0.9
EPSILON_DECAY = 0.95

ALPHA = 0.9  # 0.2 to 0.9
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -0.3
successReward = 1


class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles=None):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles=static_obstacles)
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

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) > 0:
            state, decision, reward = self.episodeDecisions[-1]
            reward += collisionDiscount

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))
            self.episodeDecisions.append((self.convertState(rb), "", 0))

            # Update Qtable and policy after collision
            self.updateAll()
            self.calculateReward()

        # Clear episode decisions
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
            self.episodeDecisions.append((goal_pos, "", 0))

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
        with open(f"policy/{scenario}/{current_map}/ClassicalQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/ClassicalQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/ClassicalQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state) -> None:
        # Optimal value of next state
        optimalQnext = max([self.Qtable[(next_state[0], next_state[1], action)] for action in action_space])

        # Update Qtable
        self.Qtable[(state[0], state[1], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

    def updatePolicy(self, state) -> None:
        # Update policy
        bestAction = max(action_space, key=lambda action: self.Qtable[(state[0], state[1], action)])

        self.policy[state] = bestAction

    def updateAll(self) -> None:
        if len(self.episodeDecisions) >= 2:
            state, decision, reward = self.episodeDecisions[-2]
            next_state = self.episodeDecisions[-1][0]

            # Update Qtable
            self.updateQtable(state, decision, reward, next_state)

            # Update policy
            self.updatePolicy(state)

    def makeDecision(self, rb) -> tuple:
        self.updateAll()

        state = self.convertState(rb)

        # Epsilon greedy
        # Randomly choose an action
        if random.random() < EPSILON:
            decision = random.choice(action_space)
        # Choose the best action
        else:
            decision = self.policy[state]

        # Add to episode decisions
        # Reward is -1 to minimize the number of steps
        self.episodeDecisions.append((state, decision, 0))

        return decision_movement[decision]
