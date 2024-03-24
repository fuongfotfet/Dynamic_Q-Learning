import sys
import pygame
import time
import numpy as np
from pygame.locals import *
from Robot import Robot
from Colors import *
from MapData import maps
from Obstacle import Obstacle

# Choose the version of the algorithm:
# 1 for Conventional Q-Learning, 2 for DFQL, 3 for Combined Q-Learning, 4 for Dual Q-Learning
version = input("Enter version: ")
if version == "1":
    from controller.ConventionalQL import QLearning
    algorithm = "ConventionalQL"
elif version == "2":
    from controller.DFQL import QLearning
    algorithm = "DFQL"
elif version == "3":
    from controller.CombinedQL import QLearning
    algorithm = "CombinedQL"
elif version == "4":
    from controller.DualQL import QLearning
    algorithm = "DualQL"
else:
    algorithm = "Unknown"

if version == "4":
    from controller.Controller import ControllerTesterDual as ControllerTester
elif version == "3":
    from controller.Controller import ControllerTesterCombined as ControllerTester
else:
    from controller.Controller import ControllerTester

# Scenario in one of: [uniform, diverse, complex]
scenario = 'uniform'  # input('Enter scenario: ')
# Input map from 1 to 3
input_map = '3'  # input('Enter map number: ')

# Initialize the robot
start = maps[scenario + input_map]["Start"]
goal = maps[scenario + input_map]["Goal"]
cell_size = 16
env_size = 512
env_padding = int(env_size * 0.06)
isTraining = input("Training? (y/n): ") == "y"
path = []
pathLength = 0
distanceToObstacle = []
success_counter = 0
robot = Robot(start=start, cell_size=cell_size,
              decisionMaker=QLearning(cell_size=cell_size, env_size=env_size, env_padding=env_padding, goal=goal)
              if isTraining else ControllerTester(cell_size=cell_size, env_size=env_size, env_padding=env_padding,
                                                  goal=goal, scenario=scenario, current_map=input_map, version=version))


def draw_target(window, target):
    pygame.draw.circle(window, RED, target, 8, 0)


def draw_start(window, start):
    pygame.draw.circle(window, GREEN, start, 6, 0)


def draw_path(window, path, color):
    for i in range(1, len(path)):
        pygame.draw.line(window, color, path[i - 1], path[i], 2)


def draw_grid(window, cell_size, env_size, env_padding):
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)


def main(scenario, test_map, interactive=True):
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
    mx, my, new_mx, new_my = None, None, None, None
    drawing = False
    done = False
    finished = False
    isStatic = True
    obstacles_list = []
    pause = False
    distanceStartGoal = np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)
    path.append(start)

    # Reset the robot
    robot.resetPosition(start)

    # Auto run
    if not interactive:
        if test_map in maps:
            obstacles_list = maps[test_map]["Obstacles"]
        done = True

    while not finished:

        screen.fill(WHITE)
        if interactive:
            # Button
            # Start
            button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.1), NORTH_PAD * 2 + env_height,
                                                       int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
            button1_text = my_font.render("Start", True, (0, 0, 0))
            button1_rect = button1_text.get_rect(center=button1.center)
            screen.blit(button1_text, button1_rect)

            # Pause
            button2 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                       int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
            button2_text = my_font.render("Pause", True, (0, 0, 0))
            button2_rect = button2_text.get_rect(center=button2.center)
            screen.blit(button2_text, button2_rect)

            # Static / Dynamic
            button3 = pygame.draw.rect(screen,
                                       BLACK,
                                       (LEFT_PAD + int(env_width * 0.4),
                                        NORTH_PAD * 2 + env_height,
                                        int(env_width * 0.2),
                                        int(SOUTH_PAD * 0.4)),
                                       4)
            button3_text = my_font.render("Static", True, (0, 0, 0))
            button3_rect = button3_text.get_rect(center=button3.center)
            screen.blit(button3_text, button3_rect)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos

                    # Button 1: Save the obstacles into Obstacles.py and start the simulation
                    if button1.collidepoint(mouse_x, mouse_y):
                        done = True
                        with open("MapData.py", 'a') as f:
                            f.write(",\n".join([o.__str__() for o in obstacles_list]))
                        if test_map in maps:
                            obstacles_list = maps[test_map]
                    # Button 2: Pause the simulation
                    elif button2.collidepoint(mouse_x, mouse_y):
                        pause = not pause
                    # Button 3: Change the mode of the obstacles when drawing
                    elif button3.collidepoint(mouse_x, mouse_y):
                        isStatic = not isStatic
                        if isStatic:
                            print("Static")
                        else:
                            print("Dynamic")
                    else:
                        mx, my = mouse_x, mouse_y
                        drawing = True
                        done = False

                # If the mouse is released, create a new obstacle
                if event.type == MOUSEBUTTONUP:
                    if drawing:
                        new_mx, new_my = event.pos
                        new_obstacle = Obstacle((mx + new_mx) / 2,
                                                (my + new_my) / 2,
                                                abs(new_mx - mx),
                                                abs(new_my - my),
                                                isStatic,
                                                np.random.randn(2) * 2)
                        obstacles_list.append(new_obstacle)
                        drawing = False

                # Ctrl + Z to undo the last obstacle
                if event.type == KEYDOWN:
                    if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        if obstacles_list and not done:
                            obstacles_list.pop()

        # Continuously draw the obstacles
        if drawing:
            new_mx, new_my = pygame.mouse.get_pos()
            pygame.draw.rect(screen, BLACK, (min(mx, new_mx), min(my, new_my), abs(new_mx - mx), abs(new_my - my)))

        # If the drawing mode is not done, draw the obstacles
        if not done:
            for obstacle in obstacles_list:
                obstacle.draw(screen)
        elif pause:
            continue
        else:

            # Draw the obstacles
            for obstacle in obstacles_list:
                obstacle.move()
                obstacle.draw(screen)

            # Make decision and move
            robot.makeDecision(obstacles_list)
            robot.move()

            # Draw the path
            path.append(robot.pos)
            pathLength += np.sqrt((path[-1][0] - path[-2][0]) ** 2 + (path[-1][1] - path[-2][1]) ** 2)
            draw_path(screen, path, ORANGE)

            # Check if the robot has collided with the obstacles or the robot stuck in local optima
            if robot.checkCollision(obstacles_list) or pathLength > 3 * distanceStartGoal:
                robot.setDiscount()
                success_counter = 0
                finished = True

            # Draw the start, target and robot
            draw_start(screen, start)
            draw_target(screen, goal)
            robot.draw(screen)

            # If training mode, speed up the simulation
            if not isTraining:
                # print(robot.getDistanceToClosestObstacle(obstacles_list))
                distanceToObstacle.append(robot.getDistanceToClosestObstacle(obstacles_list))
                time.sleep(0.05)

            # Check if the robot has reached the goal
            if robot.reach(goal):
                success_counter += 1
                finished = True

        # Draw the grid
        draw_grid(screen, cell_size, env_size, env_padding)

        # Draw boundary
        pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
        pygame.display.update()
        

if __name__ == '__main__':
    numsOfRuns = 20
    epochs = 500

    if isTraining:
        for i in range(numsOfRuns):
            for j in range(epochs):
                print(f"Run {i + 1}, Epoch {j + 1}")
                path.clear()
                pathLength = 0
                main(scenario, scenario + input_map, False)
                robot.updateQ()

                # if success_counter >= 5:
                #     break

            robot.outputPolicy(scenario, scenario + input_map, i + 1)

            # Reinitialize the controller
            robot.resetController()
    else:
        main(scenario, scenario + input_map, True)

        print(f"Path length: {pathLength}")
        print(f"Average distance to obstacle: {np.mean(distanceToObstacle)}")

        with open(f"result/{scenario}/{scenario + input_map}/{algorithm}/metric.txt", "a") as f:
            f.write(scenario + input_map + ": " + "version" + str(version) + " " + str(pathLength) + " " + str(np.mean(distanceToObstacle)))
            if success_counter == 0:
                f.write(" Fail")
            f.write("\n")

        with open(f"result/{scenario}/{scenario + input_map}/{algorithm}/path.txt", "a") as f:
            f.write(str(path))
            f.write("\n")

        time.sleep(5)

    pygame.quit()
