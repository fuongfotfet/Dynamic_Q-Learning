import json
import numpy as np

# Action space
action_space = ["up_1", "down_1", "left_1", "right_1",
                "up-left_1", "up-right_1", "down-left_1", "down-right_1"]

# Map the decision to the movement vector
decision_movement = {"left_1": (-1, 0), "right_1": (1, 0), "up_1": (0, -1), "down_1": (0, 1),
                     "up-left_1": (-1, -1), "up-right_1": (1, -1), "down-left_1": (-1, 1), "down-right_1": (1, 1),
                     "left_2": (-2, 0), "right_2": (2, 0), "up_2": (0, -2), "down_2": (0, 2),
                     "up-left_2": (-2, -2), "up-right_2": (2, -2), "down-left_2": (-2, 2), "down-right_2": (2, 2)}

# Descrete values for the state
DESCRETE_PHI = [30, 60]
DESCRETE_DELTAPHI = [-4, -2, 2, 4]
DESCRETE_DELTAD = [-1, 1]


def convertphi(phi) -> int:
    if phi < DESCRETE_PHI[0]:
        return 0  # F
    elif phi < DESCRETE_PHI[1]:
        return 1  # D
    else:
        return 2  # S


def convertdeltad(deltad) -> int:
    if deltad < DESCRETE_DELTAD[0]:
        return -1  # C
    elif deltad < DESCRETE_DELTAD[1]:
        return 0  # U
    else:
        return 1  # A


def convertdeltaphi(deltaphi) -> int:
    if deltaphi < DESCRETE_DELTAPHI[0]:
        return -2  # C
    elif deltaphi < DESCRETE_DELTAPHI[1]:
        return -1  # LC
    elif deltaphi < DESCRETE_DELTAPHI[2]:
        return 0  # U
    elif deltaphi < DESCRETE_DELTAPHI[3]:
        return 1  # LA
    else:
        return 2  # A


def find_octant(x, y, goal) -> int:
    relative_x = goal[0] - x
    relative_y = goal[1] - y

    if relative_x >= 0:
        if relative_y >= 0:
            if relative_x >= relative_y:
                return 0
            else:
                return 1
        else:
            if relative_x >= -relative_y:
                return 7
            else:
                return 6
    else:
        if relative_y >= 0:
            if -relative_x >= relative_y:
                return 3
            else:
                return 2
        else:
            if relative_x <= relative_y:
                return 4
            else:
                return 5


def angle(x1, y1, x2, y2) -> float:
    return np.arccos((x1 * x2 + y1 * y2) / (np.sqrt(x1 * x1 + y1 * y1) * np.sqrt(x2 * x2 + y2 * y2) + 1e-6))


def remap_keys(mapping) -> list:
    return [{'key': k, 'value': v} for k, v in mapping.items()]


class Controller:
    def __init__(self, cell_size, env_size, env_padding, goal):
        self.cell_size = cell_size
        self.env_size = env_size
        self.env_padding = env_padding
        self.goal = goal
        self.policy = {}

    def convertState(self, rb) -> tuple:
        # Convert state to cell
        i = int((rb.pos[0] - self.env_padding) / self.cell_size)
        j = int((rb.pos[1] - self.env_padding) / self.cell_size)

        return i, j

    def makeDecision(self, rb) -> tuple:
        return decision_movement[self.policy[self.convertState(rb)]]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        return self.makeDecision(rb)

    def calculateDistanceToGoal(self, state) -> float:
        # Use diagonal distance bacause of action space
        position = (self.env_padding + self.cell_size / 2 + state[0] * self.cell_size,
                    self.env_padding + self.cell_size / 2 + state[1] * self.cell_size)
        x, y = self.goal[0] - position[0], self.goal[1] - position[1]
        return np.abs(x - y) + np.sqrt(2) * min(np.abs(x), np.abs(y))

    def reset(self) -> None:
        pass

    def setCollision(self, rb) -> None:
        pass

    def setSuccess(self) -> None:
        pass

    def outputPolicy(self, scenario, current_map, run_index) -> None:
        pass


class ControllerTester(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal)

        # Load the policy from the file
        with open(f"policy/{scenario}/{scenario + current_map}/{algorithm}/{run}/policy.json", "r") as f:
            maps = json.load(f)
            for map in maps:
                self.policy[tuple(map['key'])] = map['value']


class ControllerTesterCombined(ControllerTester):
    def __init__(self, cell_size, env_size, env_padding, goal, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal, scenario, current_map, algorithm, run)

    def makeDecision(self, rb) -> tuple:
        state = self.convertState(rb)
        state = (state[0], state[1], -10, -10, -10)
        return decision_movement[self.policy[state]]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        # Get the position of the obstacle before and after moving
        # Then calculate the relative position of the obstacle to the robot
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        distance = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0],
                    obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0],
                         obstacle_after[1] - rb.pos[1])

        # Convert to state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_next - distance))

        state = self.convertState(rb)
        state = (state[0], state[1], c_phi, c_deltaphi, c_deltad)
        return decision_movement[self.policy[state]]


class ControllerTesterDual(ControllerTester):
    def __init__(self, cell_size, env_size, env_padding, goal, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal, scenario, current_map, algorithm, run)

        self.obstaclePolicy = {}
        # Load the policy from the file
        with open(f"policy/{scenario}/{scenario + current_map}/{algorithm}/{run}/obstaclePolicy.json", "r") as f:
            maps = json.load(f)
            for map in maps:
                self.obstaclePolicy[tuple(map['key'])] = map['value']

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        # Get the position of the obstacle before and after moving
        # Then calculate the relative position of the obstacle to the robot
        obstacle_before = obstacle_position[0]
        obstacle_after = obstacle_position[1]

        distance = np.sqrt((rb.pos[0] - obstacle_before[0]) ** 2 + (rb.pos[1] - obstacle_before[1]) ** 2)
        distance_next = np.sqrt((rb.pos[0] - obstacle_after[0]) ** 2 + (rb.pos[1] - obstacle_after[1]) ** 2)

        rb_direction = rb.nextPosition(self.goal)
        phi = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_before[0] - rb.pos[0],
                    obstacle_before[1] - rb.pos[1])
        phi_next = angle(rb_direction[0] - rb.pos[0], rb_direction[1] - rb.pos[1], obstacle_after[0] - rb.pos[0],
                         obstacle_after[1] - rb.pos[1])

        # Convert to state
        c_phi = convertphi(phi / np.pi * 180)
        c_deltaphi = convertdeltaphi((phi_next - phi) / np.pi * 180)
        c_deltad = convertdeltad((distance_next - distance))

        # Find the octant of the goal
        goal_direction = find_octant(rb.pos[0], rb.pos[1], self.goal)

        state = (c_phi, c_deltaphi, c_deltad, goal_direction)
        return decision_movement[self.obstaclePolicy[state]]
