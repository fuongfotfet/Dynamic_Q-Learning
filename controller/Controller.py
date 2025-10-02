import json
import math
import numpy as np

# Action space
action_space = ["up_1", "down_1", "left_1", "right_1",
                "up-left_1", "up-right_1", "down-left_1", "down-right_1", "stay"]

# Map the decision to the movement vector
decision_movement = {"left_1": (-1, 0), "right_1": (1, 0), "up_1": (0, -1), "down_1": (0, 1),
                     "up-left_1": (-1, -1), "up-right_1": (1, -1), "down-left_1": (-1, 1), "down-right_1": (1, 1),
                     "left_2": (-2, 0), "right_2": (2, 0), "up_2": (0, -2), "down_2": (0, 2),
                     "up-left_2": (-2, -2), "up-right_2": (2, -2), "down-left_2": (-2, 2), "down-right_2": (2, 2), 
                     "stay" : (0, 0)}

def get_opposite_action(action):
    mapping = {
        "up_1": "down_1", "down_1": "up_1",
        "left_1": "right_1", "right_1": "left_1",
        "up-left_1": "down-right_1", "down-right_1": "up-left_1",
        "up-right_1": "down-left_1", "down-left_1": "up-right_1",
        "stay": "stay"
    }
    return mapping.get(action, None)

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
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles=None):
        self.cell_size = cell_size
        self.env_size = env_size
        self.env_padding = env_padding
        self.goal = goal
        self.policy = {}
        self.static_obstacles = static_obstacles if static_obstacles is not None else []
        self.dist_to_goal = None 

    def convertState(self, rb) -> tuple:
        # Convert state to cell
        import math

        i = math.ceil((rb.pos[0] - self.env_padding) / self.cell_size)
        j = math.ceil((rb.pos[1] - self.env_padding) / self.cell_size)

        return i, j

    def makeDecision(self, rb) -> tuple:
        return decision_movement[self.policy[self.convertState(rb)]]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        return self.makeDecision(rb)

    def calculateDistanceToGoal(self, state) -> float:
        i, j = state[0], state[1]
        if self.dist_to_goal is not None and 0 <= j < len(self.dist_to_goal) and 0 <= i < len(self.dist_to_goal[0]):
            return self.dist_to_goal[j][i] * self.cell_size
        return float('inf')

    def reset(self) -> None:
        pass

    def setCollision(self, rb) -> None:
        pass

    def setSuccess(self) -> None:
        pass

    def outputPolicy(self, scenario, current_map, run_index) -> None:
        pass


class ControllerTester(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles)

        # Load the policy from the file
        with open(f"policy/{scenario}/{scenario + current_map}/{algorithm}/{run}/policy.json", "r") as f:
            maps = json.load(f)
            for map in maps:
                self.policy[tuple(map['key'])] = map['value']


class ControllerTesterCombined(ControllerTester):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, algorithm, run)

    def makeDecision(self, rb) -> tuple:
        state = self.convertState(rb)
        state = (state[0], state[1], -10, -10, -10)
        return decision_movement[self.policy[state]]

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
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

        grid_pos = self.convertState(rb)
        state = (grid_pos[0], grid_pos[1], c_phi, c_deltaphi, c_deltad)

        action = self.policy[state]

        return decision_movement[action]


class ControllerTesterDual(ControllerTester):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, algorithm, run):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, algorithm, run)
        self.static_obstacles = static_obstacles
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

        min_dist_static = float('inf')
        nearest_obs_octant = 0
        nearest_obs_dist_bin = 2

        for obs in self.static_obstacles:
            x1, x2, y1, y2 = obs.return_coordinate()
            # clamp để tìm điểm trên biên gần robot nhất
            closest_x = min(max(rb.pos[0], x1), x2)
            closest_y = min(max(rb.pos[1], y1), y2)
            d = math.hypot(rb.pos[0] - closest_x, rb.pos[1] - closest_y)
            if d < min_dist_static:
                min_dist_static = d
                nearest_point = (closest_x, closest_y)

        # Xác định octant dựa trên nearest_point
        nearest_obs_octant = find_octant(rb.pos[0], rb.pos[1], nearest_point)

        # Phân bin khoảng cách
        r = 2 * self.cell_size
        if min_dist_static <= r:
            nearest_obs_dist_bin = 0
        elif min_dist_static <= 2 * r:
            nearest_obs_dist_bin = 1
        else:
            nearest_obs_dist_bin = 2

        # (C) Kết hợp thành state đầy đủ
        state = (c_phi, c_deltaphi, c_deltad, goal_direction, nearest_obs_octant, nearest_obs_dist_bin)
        
        return decision_movement[self.obstaclePolicy[state]]


class ControllerTesterDQN(Controller):
    def __init__(self, cell_size, env_size, env_padding, goal, static_obstacles, scenario, current_map, run):
        super().__init__(cell_size, env_size, env_padding, goal, static_obstacles)
        # Lazy import to avoid circular dependency
        from controller.DQNController import DQNLearning
        # Initialize DQN agent and load trained weights
        self.agent = DQNLearning(cell_size, env_size, env_padding, goal, static_obstacles)
        # model_path = f"policy/{scenario}/{scenario + current_map}/DQN/{run}/dqn_model.pth"
        model_path = "policy/uniform/uniform1/DQN/1/dqn_model.pth"
        self.agent.load_model(model_path)
        self.agent.set_training_mode(False)

    def makeDecision(self, rb) -> tuple:
        return self.agent.makeDecision(rb)

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        return self.agent.makeObstacleDecision(rb, obstacle_position)
