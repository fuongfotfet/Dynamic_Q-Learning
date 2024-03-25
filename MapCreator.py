import sys
import pygame
import numpy as np
from pygame.locals import *
from Colors import *
from Obstacle import Obstacle


def draw_grid(window, cell_size, env_size, env_padding) -> None:
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(window, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i * cell_size, env_padding + env_size), 1)
        pygame.draw.line(window, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)


cell_size = 16
env_size = 512
env_padding = int(env_size * 0.06)
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

while not finished:
    screen.fill(WHITE)
    # Button
    # Start
    button1 = pygame.draw.rect(screen, BLACK, (LEFT_PAD + int(env_width * 0.7), NORTH_PAD * 2 + env_height,
                                                   int(env_width * 0.2), int(SOUTH_PAD * 0.4)), 4)
    button1_text = my_font.render("Save", True, (0, 0, 0))
    button1_rect = button1_text.get_rect(center=button1.center)
    screen.blit(button1_text, button1_rect)

    # Static / Dynamic
    button3 = pygame.draw.rect(screen,
                               BLACK,
                               (LEFT_PAD + int(env_width * 0.3),
                                NORTH_PAD * 2 + env_height,
                                int(env_width * 0.3),
                                int(SOUTH_PAD * 0.4)),
                               4)
    button3_text = my_font.render("Static/Dynamic", True, (0, 0, 0))
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
                    finished = True
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

    # Draw the grid
    draw_grid(screen, cell_size, env_size, env_padding)

    # Draw boundary
    pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_width, env_height), 3)
    pygame.display.update()

print("Obstacles saved to MapData.py")
