import numpy as np
from controller.Controller import Controller


class DynamicWindowApproach(Controller):
    def __init__(self, goal, cell_size, env_size, env_padding, max_speed_ratio=0.5, max_turn_rate=np.pi):
        super().__init__(cell_size, env_size, env_padding, goal)
        self.max_speed_ratio = max_speed_ratio
        self.max_turn_rate = max_turn_rate
        self.goal = goal

    def makeDecision(self, rb) -> tuple:
        # Make decision with no obstacle
        return self.makeObstacleDecision(rb, [])

    def makeObstacleDecision(self, rb, obstacle_position) -> tuple:
        # Generate feasible velocity commands
        valid_speeds = np.linspace(0, self.max_speed_ratio, num=4)
        valid_turn_rates = np.linspace(-self.max_turn_rate, self.max_turn_rate, num=9)

        # Initialize best commands
        best_cmd = (0, 0)
        best_score = float('-inf')
        robot_pose = (rb.pos[0], rb.pos[1])

        # Iterate through feasible velocity commands
        for v in valid_speeds:
            for w in valid_turn_rates:
                # Simulate motion
                simulated_trajectory = self.simulate_motion(robot_pose, v, w)

                # Evaluate trajectory
                score = self.evaluate_trajectory(simulated_trajectory, obstacle_position, rb.vision * 0.75)

                # Update best commands if score is better
                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        direction = (best_cmd[0] * np.cos(best_cmd[1]), best_cmd[0] * np.sin(best_cmd[1]))

        return direction

    def simulate_motion(self, pose, v, w):
        # Simulate robot motion given current pose and velocity commands
        new_x = pose[0] + v * np.cos(w) * self.cell_size
        new_y = pose[1] + v * np.sin(w) * self.cell_size
        return [new_x, new_y]

    def evaluate_trajectory(self, trajectory, obstacles, safety_distance):
        # Simple scoring function for trajectory evaluation
        distance_to_goal = np.linalg.norm(np.array(trajectory) - np.array(self.goal))
        score = 1.0 / (distance_to_goal + 1)  # Penalize distance to goal
        for obstacle in obstacles:
            closest_x = max(obstacle[0] - obstacle[2] / 2, min(trajectory[0], obstacle[0] + obstacle[2] / 2))
            closest_y = max(obstacle[1] - obstacle[3] / 2, min(trajectory[1], obstacle[1] + obstacle[3] / 2))
            dist_to_obstacle = ((closest_x - trajectory[0]) ** 2 + (closest_y - trajectory[1]) ** 2) ** 0.5
            if dist_to_obstacle < safety_distance:  # If too close to obstacle, penalize heavily
                score -= 10.0
        return score
