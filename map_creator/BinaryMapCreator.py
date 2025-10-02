import pygame
import numpy as np
import os
import sys

# Thêm đường dẫn để import Obstacle
def get_project_root():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(get_project_root(), 'environment'))
from environment.Obstacle import Obstacle

# Cấu hình hiển thị và kích thước lưới mục tiêu
TARGET_SIZE = 32              # Grid size sau khi scale
CELL_SIZE = 16                # Kích thước mỗi cell (pixel)
ENV_PIXEL = CELL_SIZE * TARGET_SIZE
ENV_PADDING = int(ENV_PIXEL * 0.06)
SCREEN_WIDTH = ENV_PIXEL + 2 * ENV_PADDING
SCREEN_HEIGHT = ENV_PIXEL + 3 * ENV_PADDING

# Khởi tạo pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dynamic Obstacle Editor (32x32)")
font = pygame.font.SysFont("arial", max(12, ENV_PADDING // 3))

# Đọc file MAP_save.txt và tách thành nhiều map
def load_maps(file_path):
    maps = {}
    current_name = None
    current_grid = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s[0].isdigit():  # header
                if current_name and current_grid:
                    maps[current_name] = current_grid
                current_name = s.rstrip('_')
                current_grid = []
            else:
                current_grid.append(list(map(int, s.split())))
        if current_name and current_grid:
            maps[current_name] = current_grid
    return maps

# Scale grid từ kích thước gốc xuống TARGET_SIZE x TARGET_SIZE
def scale_grid(original, target_size):
    orig_rows = len(original)
    orig_cols = len(original[0])
    scaled = [[0] * target_size for _ in range(target_size)]
    for i in range(target_size):
        for j in range(target_size):
            r0 = int(i * orig_rows / target_size)
            r1 = int((i + 1) * orig_rows / target_size)
            c0 = int(j * orig_cols / target_size)
            c1 = int((j + 1) * orig_cols / target_size)
            block = [original[r][c] for r in range(r0, min(r1, orig_rows))
                                for c in range(c0, min(c1, orig_cols))]
            if any(x == 2 for x in block):
                val = 2
            elif any(x == 3 for x in block):
                val = 3
            elif any(x == 1 for x in block):
                val = 1
            else:
                val = 0
            scaled[i][j] = val
    return scaled

# Load và chọn map
def select_map():
    file_path = os.path.join(get_project_root(), 'MAP_save.txt')
    maps_dict = load_maps(file_path)
    print("Available maps:", list(maps_dict.keys()))
    selected = input("Enter map name to edit: ")
    if selected not in maps_dict:
        print("Invalid map name, defaulting to first map")
        selected = next(iter(maps_dict.keys()))
    return selected, maps_dict[selected]

selected, grid_orig = select_map()
# Scale xuống 32x32
grid = scale_grid(grid_orig, TARGET_SIZE)

# Tìm Start, Goal và static obstacles
drawing = False
mx = my = 0
static_obs = []
dynamic_obs = []
start_cell = goal_cell = None
for i, row in enumerate(grid):
    for j, val in enumerate(row):
        x = ENV_PADDING + j * CELL_SIZE + CELL_SIZE / 2
        y = ENV_PADDING + i * CELL_SIZE + CELL_SIZE / 2
        if val == 1:
            static_obs.append(Obstacle(x, y, CELL_SIZE, CELL_SIZE, True, [0, 0]))
        elif val == 2:
            start_cell = (x, y)
        elif val == 3:
            goal_cell = (x, y)
if start_cell is None or goal_cell is None:
    raise ValueError(f"Không tìm thấy Start (2) hoặc Goal (3) trong map '{selected}'")

# Hàm vẽ toàn bộ màn hình
def draw_all():
    screen.fill((255, 255, 255))
    # Vẽ lưới
    for k in range(1, TARGET_SIZE):
        pygame.draw.line(screen, (0, 0, 0),
                         (ENV_PADDING + k * CELL_SIZE, ENV_PADDING),
                         (ENV_PADDING + k * CELL_SIZE, ENV_PADDING + ENV_PIXEL), 1)
        pygame.draw.line(screen, (0, 0, 0),
                         (ENV_PADDING, ENV_PADDING + k * CELL_SIZE),
                         (ENV_PADDING + ENV_PIXEL, ENV_PADDING + k * CELL_SIZE), 1)
    # Vẽ khung ngoài
    pygame.draw.rect(screen, (0, 0, 0), (ENV_PADDING, ENV_PADDING, ENV_PIXEL, ENV_PIXEL), 3)
    # Vẽ static obstacles
    for o in static_obs:
        o.draw(screen)
    # Vẽ dynamic obstacles
    for o in dynamic_obs:
        o.draw(screen)
    # Preview khi vẽ dynamic
    if drawing:
        x2, y2 = pygame.mouse.get_pos()
        pygame.draw.rect(screen, (0, 0, 0),
                         (min(mx, x2), min(my, y2), abs(x2 - mx), abs(y2 - my)), 1)
    # Vẽ nút Save
    btn = pygame.draw.rect(screen, (0, 0, 0),
                           (ENV_PADDING, ENV_PADDING + ENV_PIXEL + 10, 80, ENV_PADDING - 20), 2)
    txt = font.render("Save", True, (0, 0, 0))
    screen.blit(txt, txt.get_rect(center=btn.center))
    return btn

# Vòng lặp chính
finished = False
while not finished:
    save_btn = draw_all()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finished = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            px, py = event.pos
            if save_btn.collidepoint(px, py):
                out = os.path.join(get_project_root(), f"MapData_{selected}.py")
                with open(out, 'w') as f:
                    f.write('from environment.Obstacle import Obstacle\n\n')
                    f.write(f"{selected} = {{\n")
                    f.write(f"    'Start': ({start_cell[0]}, {start_cell[1]}),\n")
                    f.write(f"    'Goal': ({goal_cell[0]}, {goal_cell[1]}),\n")
                    f.write("    'Obstacles': [\n")
                    for o in static_obs + dynamic_obs:
                        f.write(f"        {o.__str__()},\n")
                    f.write("    ]\n}\n")
                print(f"Saved map '{selected}' to {out}")
                finished = True
            elif ENV_PADDING < px < ENV_PADDING + ENV_PIXEL and ENV_PADDING < py < ENV_PADDING + ENV_PIXEL:
                drawing = True
                mx, my = px, py
        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            x2, y2 = event.pos
            w, h = abs(x2 - mx), abs(y2 - my)
            cx, cy = (mx + x2) / 2, (my + y2) / 2
            vel = np.random.randn(2) * 2
            dynamic_obs.append(Obstacle(cx, cy, w, h, False, vel))
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                if dynamic_obs:
                    dynamic_obs.pop()
    pygame.display.update()
pygame.quit()