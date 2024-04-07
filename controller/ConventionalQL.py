import json
import numpy as np
import random
from controller.Controller import Controller, action_space, remap_keys

# Hyperparameters
GAMMA = 0.9  # 0.8 to 0.9

EPSILON = 0.5
EPSILON_DECAY = 0.95

ALPHA = 0.9  # 0.2 to 0.9
LEARNING_RATE_DECAY = 1.0

collisionDiscount = -5
successReward = 15


class QLearning(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal):
        # Initialize Qtable and policy
        super().__init__(cell_size, env_size, env_padding, goal)
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

        # Initialize Qtable and policy
        for i in range(int(self.env_size / self.cell_size)):
            for j in range(int(self.env_size / self.cell_size)):
                # Initialize policy to always go to the goal, wherever the robot is
                cell_center = (self.env_padding + self.cell_size / 2 + i * self.cell_size,
                               self.env_padding + self.cell_size / 2 + j * self.cell_size)
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
                decision += "_1"

                self.policy[(i, j)] = decision

                # Initialize Qtable with 0
                for action in action_space:
                    self.Qtable[(i, j, action)] = 0

    # Add collision discount to the last decision if the robot has collided with an obstacle
    def setCollision(self, rb) -> None:
        if len(self.episodeDecisions) == 0:
            return

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

        if len(self.episodeDecisions) == 0:
            return

        state, decision, reward = self.episodeDecisions[-1]
        reward += successReward

        self.episodeDecisions.pop()
        self.episodeDecisions.append((state, decision, reward))

        goal_pos = (int((self.goal[0] - self.env_padding) / self.cell_size), int((self.goal[1] - self.env_padding) / self.cell_size))
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
        self.averageReward.append(sumOfReward / len(self.episodeDecisions))

    # Out put policy to json file
    def outputPolicy(self, scenario, current_map, run_index) -> None:
        with open(f"policy/{scenario}/{current_map}/ConventionalQL/{run_index}/policy.json", "w") as outfile:
            json.dump(remap_keys(self.policy), outfile, indent=2)

        with open(f"policy/{scenario}/{current_map}/ConventionalQL/{run_index}/sumOfRewards.txt", "w") as outfile:
            outfile.write(str(self.sumOfRewards))

        with open(f"policy/{scenario}/{current_map}/ConventionalQL/{run_index}/averageReward.txt", "w") as outfile:
            outfile.write(str(self.averageReward))

    def updateQtable(self, state, decision, reward, next_state) -> float:
        # Optimal value of next state
        optimalQnext = max([self.Qtable[(next_state[0], next_state[1], action)] for action in action_space])

        prevQ = self.Qtable[(state[0], state[1], decision)]
        # Update Qtable
        self.Qtable[(state[0], state[1], decision)] = (1 - ALPHA) * self.Qtable[
            (state[0], state[1], decision)] + ALPHA * (reward + GAMMA * optimalQnext)

        # Calculate change in Q value
        return abs(self.Qtable[(state[0], state[1], decision)] - prevQ)

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

    def makeDecision(self, rb) -> str:
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
        self.episodeDecisions.append((state, decision, -1))

        return decision
