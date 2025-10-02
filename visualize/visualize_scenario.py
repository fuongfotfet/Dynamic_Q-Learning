import pygame
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/environment')

from environment.MapData import maps
from environment.Colors import *

# ========== Cấu hình ========== #
cell_size = 16
env_size = 512
env_padding = int(env_size * 0.06)
LEFT_PAD, RIGHT_PAD, NORTH_PAD, SOUTH_PAD = env_padding, env_padding, env_padding, env_padding * 3
SCREEN_WIDTH = env_size + LEFT_PAD + RIGHT_PAD
SCREEN_HEIGHT = env_size + NORTH_PAD + SOUTH_PAD

# === Load map uniform1 ===
selected_map = maps["diverse2"]
start = selected_map["Start"]
goal = selected_map["Goal"]
obstacles = selected_map["Obstacles"]

# ========== Init Pygame ========== #
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Display Map: uniform1")
clock = pygame.time.Clock()

# ========== Các hàm vẽ ========== #
def draw_target(surface, pos):
    pygame.draw.circle(surface, RED, pos, 8)

def draw_start(surface, pos):
    pygame.draw.circle(surface, GREEN, pos, 6)

def draw_grid(surface):
    for i in range(1, int(env_size / cell_size)):
        pygame.draw.line(surface, BLACK, (env_padding + i * cell_size, env_padding),
                         (env_padding + i   * cell_size, env_padding + env_size), 1)
        pygame.draw.line(surface, BLACK, (env_padding, env_padding + i * cell_size),
                         (env_padding + env_size, env_padding + i * cell_size), 1)

# ========== Vòng lặp chính ========== #
running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Vẽ grid + boundary
    draw_grid(screen)
    pygame.draw.rect(screen, BLACK, (LEFT_PAD, NORTH_PAD, env_size, env_size), 3)

    # Vẽ start + goal
    draw_start(screen, start)
    draw_target(screen, goal)

    # Vẽ obstacles
    for obstacle in obstacles:
        obstacle.move()  
        obstacle.draw(screen)

    pygame.display.update()
    clock.tick(30)

pygame.quit()
sys.exit()