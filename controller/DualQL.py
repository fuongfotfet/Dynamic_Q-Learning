import json
import numpy as np
import random
from Controller import Controller, action_space, decision_movement, convertphi, convertdeltaphi, convertdeltad, angle, \
    remap_keys, find_octant

# Hyperparameters
GAMMA = 0.9  # 0.8 to 0.9

EPSILON = 0.5
EPSILON_DECAY = 0.95

ALPHA = 0.8  # 0.2 to 0.8
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -5
successReward = 15

OBSTACLE_REWARD_FACTOR = 0.2


class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal)
        self.Qtable = {}
        self.obstacleQtable = {}
        self.obstaclePolicy = {}
        self.obstacleEpisodeDecisions = []
        self.episodeDecisions = []
        self.policyChanged = []
        self.sumOfRewards = []
        self.averageReward = []
        self.isObstacleDecisionMade = False
        self.hasCollided = False

        self.reset()

    def reset(self):
        global EPSILON
        EPSILON = 0.5

        self.episodeDecisions.clear()
        self.obstacleEpisodeDecisions.clear()
        self.policyChanged.clear()
        self.sumOfRewards.clear()
        self.averageReward.clear()
        self.hasCollided = False

        # Initialize Qtable and policy
        for i in range(int(self.env_size / self.cell_size)):
            for j in range(int(self.env_size / self.cell_size)):
                # Initialize policy to always go to the goal, wherever the robot is
                cell_center = (self.env_padding + self.cell_size / 2 + i * self.cell_size, self.env_padding + self.cell_size / 2 + j * self.cell_size)
                direction = (self.goal[0] - cell_center[0], self.goal[1] - cell_center[1])
                decision = ""

                # A 90-degree region is divided into 3 parts
                # For example, 0 - 90: right, up-right, up
                ratio = np.tan(np.pi / 8)
                if abs(direction[1]) > ratio * abs(direction[0]):
                    if direction[1] > 0:
                        decision += "down"
                    else:
                        decision += "up"

                    if abs(direction[0]) > ratio * abs(direction[1]):
                        if direction[0] > 0:
                            decision += "-right"
                        else:
                            decision += "-left"
                else:
                    if direction[0] > 0:
                        decision += "right"
                    else:
                        decision += "left"

                # If the robot is too far from the goal, add 2 to the decision, else add 1
                distance = self.calculateDistanceToGoal((i, j))
                decision += "_1"

                # Initialize Qtable, value is higher if the robot is closer to the goal
                if distance > 0:
                    ini_value = self.cell_size / distance
                else:
                    ini_value = 0

                self.policy[(i, j)] = decision

                for action in action_space:
                    self.Qtable[(i, j, action)] = ini_value

        # Initialize ObstacleQtable and obstaclePolicy
        for phi in range(3):
            for delta_phi in range(-2, 3):  # DeltaPhi: C, LC, U, LA, A
                for delta_d in range(-1, 2):  # DeltaD: C, U, A
                    for goal_direction in range(8):
                        if goal_direction == 0:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "right_1"
                        elif goal_direction == 1:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "down-right_1"
                        elif goal_direction == 2:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "down_1"
                        elif goal_direction == 3:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "down-left_1"
                        elif goal_direction == 4:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "left_1"
                        elif goal_direction == 5:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "up-left_1"
                        elif goal_direction == 6:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "up_1"
                        elif goal_direction == 7:
                            self.obstaclePolicy[(phi, delta_phi, delta_d, goal_direction)] = "up-right_1"

                        for action in action_space:
                            self.obstacleQtable[(phi, delta_phi, delta_d, goal_direction, action)] = 0

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self):
        if not self.isObstacleDecisionMade:
            if len(self.episodeDecisions) == 0:
                return

            self.hasCollided = True

            state, decision, reward = self.episodeDecisions[-1]
            reward += collisionDiscount

            self.episodeDecisions.pop()
            self.episodeDecisions.append((state, decision, reward))
        else:
            if len(self.obstacleEpisodeDecisions) == 0:
                return

            self.hasCollided = True

            state, decision, reward = self.obstacleEpisodeDecisions[-1]
            reward += collisionDiscount

            self.obstacleEpisodeDecisions.pop()
            self.obstacleEpisodeDecisions.append((state, decision, reward))

    # Out put policy to json file
    def outputPolicy(self, scenario, current_map, run_index):
        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/obstaclePolicy.json", "w") as outfile:
            json.dump(remap_keys(self.obstaclePolicy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/policyChange.txt", "w") as outfile:
            outfile.write(str(self.policyChanged))

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/DualQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state):
        # Optimal value of next state
        optimalQnext = max(
            [self.Qtable[(next_state[0], next_state[1], action)] for action
             in action_space])

        prevQ = self.Qtable[(state[0], state[1], decision)]
        # Update Qtable
        self.Qtable[(state[0], state[1], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

        # Calculate change in Q value
        return abs(self.Qtable[(state[0], state[1], decision)] - prevQ)

    def updatePolicy(self, state):
        # Update policy
        bestAction = max(action_space,
                         key=lambda action: self.Qtable[(state[0], state[1], action)])

        self.policy[state] = bestAction

    def updateObstacleQtable(self, state, decision, reward, next_state):
        # Optimal value of next state
        optimalQnext = max(
            [self.obstacleQtable[(next_state[0], next_state[1], next_state[2], next_state[3], action)] for action
             in action_space])

        prevQ = self.obstacleQtable[(state[0], state[1], state[2], next_state[3], decision)]
        # Update Qtable
        self.obstacleQtable[(state[0], state[1], state[2], next_state[3], decision)] = (1 - ALPHA) * self.obstacleQtable[
            (state[0], state[1], state[2], next_state[3], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

        # Calculate change in Q value
        return abs(self.obstacleQtable[(state[0], state[1], state[2], next_state[3], decision)] - prevQ)

    def updateObstaclePolicy(self, state):
        # Update policy
        bestAction = max(action_space,
                         key=lambda action: self.obstacleQtable[(state[0], state[1], state[2], state[3], action)])

        self.obstaclePolicy[state] = bestAction

    def updateAll(self, rb):
        global EPSILON

        # Add reward after success
        if not self.hasCollided and len(self.episodeDecisions) > 0:
            self.episodeDecisions[-1] = (
                self.episodeDecisions[-1][0], self.episodeDecisions[-1][1],
                self.episodeDecisions[-1][2] + successReward)

        # Add the last state to the episode decisions
        last_state = self.convertState(rb)
        self.episodeDecisions.append((last_state, "", 0))
        self.obstacleEpisodeDecisions.append(((0, 0, 0, 0), "", 0))

        QvalueChange = 0
        totalReward = 0
        # Update Qtable and policy
        for i in range(len(self.episodeDecisions) - 2, -1, -1):
            state, decision, reward = self.episodeDecisions[i]
            next_state = self.episodeDecisions[i + 1][0]

            totalReward += reward

            # Update Qtable
            QvalueChange += self.updateQtable(state, decision, reward, next_state)

            # Update policy
            self.updatePolicy(state)

        # Update obstacle Qtable and policy
        for i in range(len(self.obstacleEpisodeDecisions) - 2, -1, -1):
            state, decision, reward = self.obstacleEpisodeDecisions[i]
            next_state = self.obstacleEpisodeDecisions[i + 1][0]

            totalReward += reward

            # Update Qtable
            QvalueChange += self.updateObstacleQtable(state, decision, reward, next_state)

            # Update policy
            self.updateObstaclePolicy(state)

        # Calculate policy value change
        self.policyChanged.append(QvalueChange)
        self.sumOfRewards.append(totalReward)
        self.averageReward.append(totalReward / (len(self.episodeDecisions) + len(self.obstacleEpisodeDecisions)))

        # Clear episode decisions
        self.episodeDecisions.clear()
        self.obstacleEpisodeDecisions.clear()

        # Dynamic epsilon
        if not self.hasCollided:
            EPSILON *= EPSILON_DECAY

        self.hasCollided = False

    def makeDecision(self, rb):
        # Add reward to obstacle decision after successfully avoiding the obstacle
        if self.isObstacleDecisionMade and len(self.obstacleEpisodeDecisions) > 0:
            self.obstacleEpisodeDecisions[-1] = (
                self.obstacleEpisodeDecisions[-1][0], self.obstacleEpisodeDecisions[-1][1],
                self.obstacleEpisodeDecisions[-1][2] + successReward)

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
        reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight

        # Add to episode decisions
        self.episodeDecisions.append((state, decision, reward))

        return decision

    def makeObstacleDecision(self, rb, obstacle_position):
        self.isObstacleDecisionMade = True

        # Get the position of the obstacle before and after moving
        # Then calculate the relative position of the obstacle to the robot
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        distance_to_obstacle = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_to_obstacle_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0], obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0], obstacle_after[1] - rb.pos[1])

        # Convert to state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_to_obstacle_next - distance_to_obstacle))

        # Find the octant of the goal
        goal_direction = find_octant(rb.pos[0], rb.pos[1], self.goal)

        state = (c_phi, c_deltaphi, c_deltad, goal_direction)

        # Epsilon greedy
        # Randomly choose an action
        if random.random() < EPSILON:
            decision = random.choice(action_space)
        # Choose the best action
        else:
            decision = self.obstaclePolicy[state]

        # Calculate reward
        distance = self.calculateDistanceToGoal(state)

        movement = decision_movement[decision]
        next_state = (state[0] + movement[0], state[1] + movement[1])
        next_distance = self.calculateDistanceToGoal(next_state)

        distance_to_obstacle_after_movement = np.sqrt((rb.pos[0] + movement[0] * self.cell_size - obstacle_after[0]) ** 2
                                                      + (rb.pos[1] + movement[1] * self.cell_size - obstacle_after[1]) ** 2)

        if decision in ["up_1", "down_1", "left_1", "right_1"]:
            weight = 1
        else:
            weight = 1 / np.sqrt(2)

        # Reward is the change in distance to the goal (negative reward if the robot is moving away from the goal)
        # plus the change in distance to the obstacle (negative reward if the robot is moving closer to the obstacle)
        reward = ((distance - next_distance) / np.abs(distance - next_distance + 1e-6)) * weight
        + OBSTACLE_REWARD_FACTOR * (distance_to_obstacle_after_movement - distance_to_obstacle) / np.abs(distance_to_obstacle_after_movement - distance_to_obstacle + 1e-6) * weight

        # Add to episode decisions
        self.obstacleEpisodeDecisions.append((state, decision, reward))

        return decision
