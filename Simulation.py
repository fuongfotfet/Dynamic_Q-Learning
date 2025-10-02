import pygame
import time
import numpy as np
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '\\controller')
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '\\environment')
from pygame.locals import *
from environment.Robot import Robot
from environment.Colors import *
from environment.MapData import maps



# Choose the version of the algorithm:
# 1 for Classical Q-Learning, 2 for DFQL, 3 for Combined Q-Learning, 4 for Dual Q-Learning, 5 for DWA, 6 for DQN
version = input("Enter version (1-ClassicalQL, 2-DFQL, 3-CombinedQL, 4-DualQL, 5-DWA, 6-DQN): ")
if version == "1":
    from controller.ClassicalQL import QLearning as Controller

    algorithm = "ClassicalQL"
elif version == "2":
    from controller.DFQL import QLearning as Controller

    algorithm = "DFQL"
elif version == "3":
    from controller.CombinedQL import QLearning as Controller

    algorithm = "CombinedQL"
elif version == "4":
    from controller.DualQL import QLearning as Controller

    algorithm = "DualQL"
elif version == "5":
    from controller.DWA import DynamicWindowApproach as Controller

    algorithm = "DWA"
elif version == "6":
    from controller.DQNController import QLearning as Controller

    algorithm = "DQN"
else:
    algorithm = "Unknown"

if version == "4":
    from controller.Controller import ControllerTesterDual as ControllerTester
elif version == "3":
    from controller.Controller import ControllerTesterCombined as ControllerTester
else:
    from controller.Controller import ControllerTester

isTraining = version not in ["5"] and input("Training? (y/n): ") == "y"
# Scenario in one of: [uniform, diverse, complex]
scenario = input("Enter scenario (uniform/diverse/complex): ")
# Input map from 1 to 3
input_map = input("Enter map (1/2/3): ")
# Number of runs
numsOfRuns = 20 if input("Automatically run 20 times? (y/n): ") == "y" else 1

# Initialize the robot
start = maps[scenario + input_map]["Start"]
goal = maps[scenario + input_map]["Goal"]
cell_size = 16
env_size = 512
env_padding = int(env_size * 0.06)
path = []
pathLength = 0
distanceToObstacle = []
turningAngle = []
success_counter = 0

static_obs = [obs for obs in maps[scenario + input_map]["Obstacles"] if obs.static]
controller = Controller(cell_size=cell_size,
                        env_size=env_size,
                        env_padding=env_padding,
                        goal=goal,
                        static_obstacles=static_obs)
robot = Robot(start=start,
              cell_size=cell_size,
              decisionMaker=controller)


def draw_target(window, target) -> None:
    pygame.draw.circle(window, RED, target, 8, 0)


def draw_start(window, start) -> None:
    pygame.draw.circle(window, GREEN, start, 6, 0)


def draw_path(window, path, color) -> None:
    for i in range(1, len(path)):
        pygame.draw.line(window, color, path[i - 1], path[i], 2)


def draw_grid(window, cell_size, env_size, env_padding) -> None:
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)


def get_sum_turning_angle(path):
    total_angle = 0
    for i in range(1, len(path) - 1):
        vector1 = ([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])
        vector2 = ([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])
        angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6))
        total_angle += np.abs(angle)
    return total_angle


def main(test_map):
    global success_counter, pathLength

    env_width = env_height = env_size
    # Pygame setup
    NORTH_PAD, SOUTH_PAD, LEFT_PAD, RIGHT_PAD = env_padding, 3 * env_padding, env_padding, env_padding
    SCREEN_WIDTH = env_width + LEFT_PAD + RIGHT_PAD
    SCREEN_HEIGHT = env_height + NORTH_PAD + SOUTH_PAD

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption("Q-Learning Path Planning")
    my_font = pygame.font.SysFont("arial", SOUTH_PAD // 5)

    # Initialization
    finished = False
    obstacles_list = []
    pause = False
    started = False
    distanceStartGoal = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
    prev_dist = distanceStartGoal         # ← thêm dòng này
    no_improve_steps = 0                  # ← thêm dòng này
    max_no_improve = 10                   # ← tối đa bao nhiêu bước không cải thiện thì terminate
    path.append(start)

    # Reset the robot
    robot.resetPosition(start)

    # Load the obstacles
    if test_map in maps:
        obstacles_list = maps[test_map]["Obstacles"]

    start_time = time.time()

    while not finished:

        screen.fill(WHITE)
        # Button
        # Start
        button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button1_text = my_font.render("Start", True, (0, 0, 0))
        button1_rect = button1_text.get_rect(center=button1.center)
        screen.blit(button1_text, button1_rect)

        # Pause
        button2 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.4), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button2_text = my_font.render("Pause", True, (0, 0, 0))
        button2_rect = button2_text.get_rect(center=button2.center)
        screen.blit(button2_text, button2_rect)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Button 1: Save the obstacles into Obstacles.py and start the simulation
                if button1.collidepoint(mouse_x, mouse_y):
                    started = True
                # Button 2: Pause the simulation
                elif button2.collidepoint(mouse_x, mouse_y):
                    pause = not pause

        if started or numsOfRuns > 1:
            if pause:
                continue
            else:
                # Draw the obstacles
                for obstacle in obstacles_list:
                    obstacle.move()
                    obstacle.draw(screen, with_past=False)

                # Make decision and move
                robot.makeDecision(obstacles_list)
                robot.move()

                # Draw the path
                path.append(robot.pos)
                pathLength += np.sqrt((path[-1][0] - path[-2][0]) ** 2 + (path[-1][1] - path[-2][1]) ** 2)
                draw_path(screen, path, ORANGE)

                # Check if the robot has collided with the obstacles or the robot stuck in local optima
                collision_detected = robot.checkCollision(obstacles_list) or pathLength > 10 * distanceStartGoal or time.time() - start_time > 60
                goal_reached = robot.reach(goal)
                
                # DQN Training Integration
                if version == "6" and hasattr(robot.decisionMaker, 'process_simulation_step'):
                    robot.decisionMaker.process_simulation_step(
                        robot, obstacles_list, goal_reached, collision_detected
                    )
                
                if collision_detected:
                    robot.setDiscount()
                    success_counter = 0
                    finished = True

                # Draw the start, target and robot
                draw_start(screen, start)
                draw_target(screen, goal)
                robot.draw(screen)
                # time.sleep(1)

                # If training mode, speed up the simulation
                if not isTraining:
                    distanceToObstacle.append(robot.getDistanceToClosestObstacle(obstacles_list))
                    time.sleep(0.02)

                # Check if the robot has reached the goal
                if goal_reached:
                    robot.setSuccess()
                    success_counter += 1
                    finished = True

        # Draw the grid
        draw_grid(screen, cell_size, env_size, env_padding)

        # Draw boundary
        pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
        pygame.display.update()

    # Reset the obstacles following the path
    for obstacle in obstacles_list:
        obstacle.reset()


if __name__ == '__main__':
    epochs = 1000

    if isTraining:
        for i in range(numsOfRuns):
            print(f"Run {i + 1}")
            for j in range(epochs):
                path.clear()
                pathLength = 0
                main(scenario + input_map)

            robot.outputPolicy(scenario, scenario + input_map, i + 1)
            # Reinitialize the controller
            robot.resetController()
    else:
        for i in range(numsOfRuns):
            # Initialize the controller tester for QL (ver 1-4) for each run
            if version not in ["5", "6"]:  # Classical QL algorithms
                robot.decisionMaker = ControllerTester(cell_size=cell_size, env_size=env_size, env_padding=env_padding,
                                                       goal=goal, static_obstacles = static_obs, scenario=scenario, current_map=input_map,
                                                       algorithm=algorithm, run=i + 1)
            elif version == "6":  # DQN testing
                from controller.Controller import ControllerTesterDQN
                robot.decisionMaker = ControllerTesterDQN(cell_size=cell_size, env_size=env_size, env_padding=env_padding,
                                                          goal=goal, static_obstacles=static_obs, scenario=scenario, current_map=input_map,
                                                          run=i + 1)
            main(scenario + input_map)

            print(f"Run {i + 1}")
            print(f"Path length: {pathLength}")
            print(f"Average turning angle: {get_sum_turning_angle(path)}")
            print(f"Average distance to obstacle: {np.mean(distanceToObstacle)}")
            print("--------------------")

            with open(f"result/{scenario}/{scenario + input_map}/{algorithm}/metric.txt", "a") as f:
                f.write(scenario + input_map + ": " + "version" + str(version) + " " + str(pathLength) + " "
                        + str(get_sum_turning_angle(path)) + " " + str(np.mean(distanceToObstacle)))
                if success_counter == 0:
                    f.write(" Fail")
                f.write("\n")

            with open(f"result/{scenario}/{scenario + input_map}/{algorithm}/path.txt", "a") as f:
                f.write(str(path))
                f.write("\n")

            time.sleep(3)

            path.clear()
            pathLength = 0
            distanceToObstacle.clear()
            success_counter = 0

    pygame.quit()
