import pygame
import random
from pygame.locals import *
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '\\environment')
from MapData import maps
from Colors import *


def draw_target(window, target) -> None:
    pygame.draw.circle(window, RED, target, 8, 0)


def draw_start(window, start) -> None:
    pygame.draw.circle(window, GREEN, start, 6, 0)


def draw_path(window, path, portion, color) -> None:
    for i in range(1, int(len(path) * portion)):
        # Diagonal line has a thickness sqrt(2) higher than the horizontal or vertical line
        if path[i - 1][0] == path[i][0] or path[i - 1][1] == path[i][1]:
            pygame.draw.line(window, color, path[i - 1], path[i], 4)
        else:
            pygame.draw.line(window, color, path[i - 1], path[i], 6)


def draw_grid(window, cell_size, env_size, env_padding) -> None:
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)


def load_path(path):
    # Choose a random run for each algorithm
    run = [random.randint(1, 20) for i in range(len(index_algorithm))]
    for i in range(len(index_algorithm)):
        print(f"{index_algorithm[i]}: Run {run[i]} chosen")
    print("-----------------------------------")

    # Read the path from the file
    for i in range(len(index_algorithm)):
        with open(os.path.dirname(os.path.realpath(__file__)) + f"\\{scenario}\\{scenario + input_map}\\{index_algorithm[i]}\\path.txt", "r") as f:
            # Read the path in line run[i] from the file and save it in the path[i]
            for j, line in enumerate(f):
                if j == run[i]:
                    # Read list of tuples from the file
                    path[i] = [tuple(map(float, x.split(", "))) for x in line[2:-3].split("), (")]
                    if (path[i][-1][0] - goal[0]) ** 2 + (path[i][-1][1] - goal[1]) ** 2 <= cell_size ** 2:
                        path[i].append(goal)


if __name__ == "__main__":
    index_algorithm = {0: "ClassicalQL", 1: "DFQL", 2: "CombinedQL", 3: "DualQL", 4: "DWA"}

    scenario = input("Enter scenario (uniform/diverse/complex): ")
    input_map = input("Enter map (1/2/3): ")

    # Parameters
    start = maps[scenario + input_map]["Start"]
    goal = maps[scenario + input_map]["Goal"]
    cell_size = 16
    env_size = 512
    env_padding = int(env_size * 0.06)
    path = [[] for _ in range(len(index_algorithm))]
    nums_of_section = 1
    total_section = 3

    # Load the path
    load_path(path)

    # The colors of the path of each algorithm
    colors = [DARK_ORANGE, BLUE, GREEN, RED, PURPLE]

    # Pygame setup
    env_width = env_height = env_size
    NORTH_PAD, SOUTH_PAD, LEFT_PAD, RIGHT_PAD = env_padding, 3 * env_padding, env_padding, env_padding
    SCREEN_WIDTH = env_width + LEFT_PAD + RIGHT_PAD
    SCREEN_HEIGHT = env_height + NORTH_PAD + SOUTH_PAD

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption("Q-Learning Path Planning")
    my_font = pygame.font.SysFont("arial", SOUTH_PAD // 5)

    # Initialization
    obstacles_list = []
    pause = False

    # Load the obstacles
    if scenario + input_map in maps:
        obstacles_list = maps[scenario + input_map]["Obstacles"]

    while True:
        screen.fill(WHITE)

        # Button Next
        button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button1_text = my_font.render("Reload", True, (0, 0, 0))
        button1_rect = button1_text.get_rect(center=button1.center)
        screen.blit(button1_text, button1_rect)

        # Button Pause
        button2 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.4), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button2_text = my_font.render("Pause", True, (0, 0, 0))
        button2_rect = button2_text.get_rect(center=button2.center)
        screen.blit(button2_text, button2_rect)

        # Button Reset
        button3 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.1), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
        button3_text = my_font.render("Reset", True, (0, 0, 0))
        button3_rect = button3_text.get_rect(center=button3.center)
        screen.blit(button3_text, button3_rect)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Button 1: Load new random paths and reset the obstacles
                if button1.collidepoint(mouse_x, mouse_y):
                    load_path(path)
                    for obstacle in obstacles_list:
                        obstacle.reset()

                # Button 2: Pause the simulation
                if button2.collidepoint(mouse_x, mouse_y):
                    pause = not pause

                # Button 3: Reset the obstacles following the path
                if button3.collidepoint(mouse_x, mouse_y):
                    for obstacle in obstacles_list:
                        obstacle.reset()

            # If right arrow key is pressed, move the robot to the right
            if event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    nums_of_section = nums_of_section % total_section + 1
                if event.key == K_LEFT:
                    nums_of_section = nums_of_section - 1 if nums_of_section > 1 else total_section
        if pause:
            continue
        else:
            # Draw the path
            for i in range(len(index_algorithm)):
                draw_path(screen, path[i], nums_of_section / total_section, colors[i])

            # Draw the obstacles
            for obstacle in obstacles_list:
                obstacle.move()
                obstacle.draw(screen)

            # Draw the start, target and robot
            draw_start(screen, start)
            draw_target(screen, goal)

            time.sleep(0.01)

        # Draw the grid
        draw_grid(screen, cell_size, env_size, env_padding)

        # Draw boundary
        pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
        pygame.display.update()
