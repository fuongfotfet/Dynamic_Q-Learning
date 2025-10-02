# inspect_state.py
import argparse
import os
import sys
import numpy as np
import pygame

# Ensure imports resolve like Simulation.py
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "controller"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "environment"))

from controller.DQNController import DQNLearning
from environment.MapData import maps

class DummyRB:
    def __init__(self, pos):
        self.pos = pos

    def nextPosition(self, goal):
        # Consistent với CombinedQL: trả về goal để tính góc tới đích
        return goal

def main():
    parser = argparse.ArgumentParser(description="Inspect get_state_representation at a given point")
    parser.add_argument("--scenario", type=str, required=True, choices=["uniform", "diverse", "complex"])
    parser.add_argument("--map", dest="map_id", type=str, required=True, choices=["1", "2", "3"])
    parser.add_argument("--x", type=int, default=None, help="Robot x (pixels), optional when using UI clicks")
    parser.add_argument("--y", type=int, default=None, help="Robot y (pixels), optional when using UI clicks")
    # Optional dynamic obstacle positions (2 snapshots)
    parser.add_argument("--obs1x", type=float, default=None)
    parser.add_argument("--obs1y", type=float, default=None)
    parser.add_argument("--obs2x", type=float, default=None)
    parser.add_argument("--obs2y", type=float, default=None)

    args = parser.parse_args()
    key = args.scenario + args.map_id

    # Environment params (match Simulation.py)
    cell_size = 16
    env_size = 512
    env_padding = int(env_size * 0.06)
    VISION_RADIUS = 40  # same as Robot default vision

    # Build env data
    goal = maps[key]["Goal"]
    static_obs = [obs for obs in maps[key]["Obstacles"] if obs.static]

    # Initialize agent (training/test mode không ảnh hưởng tới get_state_representation)
    agent = DQNLearning(cell_size=cell_size,
                        env_size=env_size,
                        env_padding=env_padding,
                        goal=goal,
                        static_obstacles=static_obs)

    # Dummy robot at provided position or map Start
    start = maps[key]["Start"]
    rb_x = args.x if args.x is not None else start[0]
    rb_y = args.y if args.y is not None else start[1]
    rb = DummyRB((rb_x, rb_y))

    # Optional obstacle tuple: [(x1, y1), (x2, y2)]
    obstacle_position = None
    if args.obs1x is not None and args.obs1y is not None and args.obs2x is not None and args.obs2y is not None:
        obstacle_position = [(args.obs1x, args.obs1y), (args.obs2x, args.obs2y)]

    # --- UI render with pygame ---
    def draw_state_panel(window, state_matrix, grid_distance, triplet, top_left=(560, 20), cell=28, gap=3):
        x0, y0 = top_left
        for j in range(5):
            for i in range(5):
                v = state_matrix[j, i]
                color = (230, 230, 230)
                if v == 1:
                    color = (60, 60, 60)
                elif v == 2:
                    color = (0, 180, 255)
                elif v == 3:
                    color = (255, 180, 0)
                pygame.draw.rect(window, color, (x0 + i * (cell + gap), y0 + j * (cell + gap), cell, cell))
        pygame.draw.rect(window, (0, 0, 0), (x0 - 2, y0 - 2, 5 * (cell + gap) - gap + 4, 5 * (cell + gap) - gap + 4), 2)

        font = pygame.font.SysFont("arial", 18)
        text1 = font.render(f"grid_distance: {grid_distance:.2f}", True, (0, 0, 0))
        text2 = font.render(f"(c_phi, c_deltaphi, c_deltad): {tuple(triplet)}", True, (0, 0, 0))
        center_val = state_matrix[2, 2]
        text3 = font.render(f"center cell: {center_val} (expect 2)", True, (0, 0, 0))
        window.blit(text1, (x0, y0 + 5 * (cell + gap) + 10))
        window.blit(text2, (x0, y0 + 5 * (cell + gap) + 34))
        window.blit(text3, (x0, y0 + 5 * (cell + gap) + 58))

    def draw_environment(window):
        # Boundary
        pygame.draw.rect(window, (0, 0, 0), (env_padding, env_padding, env_size, env_size), 2)
        # Static obstacles
        for obs in static_obs:
            x1, x2, y1, y2 = obs.return_coordinate()
            pygame.draw.rect(window, (100, 100, 100), (x1, y1, x2 - x1, y2 - y1))

    def draw_points(window, rb_pos, obs_pos):
        # Robot point
        pygame.draw.circle(window, (0, 120, 255), rb_pos, 6)
        # Vision circle for visualization
        pygame.draw.circle(window, (255, 0, 0), rb_pos, VISION_RADIUS, 1)
        # Obstacles snapshots (if any)
        if obs_pos is not None:
            (ox1, oy1), (ox2, oy2) = obs_pos
            pygame.draw.circle(window, (255, 0, 0), (int(ox1), int(oy1)), 5)
            pygame.draw.circle(window, (255, 120, 0), (int(ox2), int(oy2)), 5)
            pygame.draw.line(window, (200, 80, 0), (int(ox1), int(oy1)), (int(ox2), int(oy2)), 2)

    pygame.init()
    # Window wide enough to show map and panel
    W, H = env_padding * 2 + env_size + 420, env_padding * 2 + env_size
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Inspect get_state_representation")

    running = True
    clock = pygame.time.Clock()
    pending_obs1 = None  # For right-click sequence
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                obstacle_position = None
                pending_obs1 = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                inside_env = (env_padding <= mx <= env_padding + env_size and env_padding <= my <= env_padding + env_size)
                # Left click: set robot position if inside env
                if event.button == 1 and inside_env:
                    rb.pos = (mx, my)
                # Right click: collect two snapshots for dynamic obstacle
                elif event.button == 3 and inside_env:
                    if pending_obs1 is None:
                        pending_obs1 = (mx, my)
                    else:
                        obstacle_position = [pending_obs1, (mx, my)]
                        pending_obs1 = None
        # Filter obstacle by vision range (check first snapshot)
        filtered_obstacle = None
        if obstacle_position is not None:
            try:
                (ox1, oy1), _ = obstacle_position
                dx = ox1 - rb.pos[0]
                dy = oy1 - rb.pos[1]
                if (dx * dx + dy * dy) <= VISION_RADIUS * VISION_RADIUS:
                    filtered_obstacle = obstacle_position
            except Exception:
                filtered_obstacle = None

        # Recompute state for current UI selections
        state_matrix, grid_distance, triplet = agent.get_state_representation(rb, filtered_obstacle)

        screen.fill((245, 245, 245))
        draw_environment(screen)
        draw_points(screen, rb.pos, filtered_obstacle)
        draw_state_panel(screen, state_matrix, grid_distance, triplet, top_left=(env_padding + env_size + 30, 30))

        # Instructions
        font = pygame.font.SysFont("arial", 16)
        lines = [
            "Left click: set robot position",
            "Right click twice: set dynamic obstacle (2 snapshots)",
            "Within vision range only (40 px)",
            "Press C: clear obstacle snapshots",
            "ESC: quit",
        ]
        for i, t in enumerate(lines):
            txt = font.render(t, True, (0, 0, 0))
            screen.blit(txt, (env_padding + env_size + 30, H - 20 * (len(lines) - i)))
        pygame.display.update()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()