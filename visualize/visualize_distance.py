import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from environment.MapData import maps
from environment.Obstacle import Obstacle
from controller.CombinedQL import compute_distances
import math

# Cấu hình
cell_size = 16
env_padding = 0

# Chọn bản đồ
map_data = maps["uniform1"]
start = map_data["Start"]
goal = map_data["Goal"]
obstacles = map_data["Obstacles"]

# Khởi tạo lưới occupancy
M = int(600 / cell_size)
grid = [[0]*M for _ in range(M)]

for obs in obstacles:
    x1 = int((obs.x - obs.width/2 - env_padding) // cell_size)
    x2 = int((obs.x + obs.width/2 - env_padding) // cell_size)
    y1 = int((obs.y - obs.height/2 - env_padding) // cell_size)
    y2 = int((obs.y + obs.height/2 - env_padding) // cell_size)
    for i in range(x1, x2+1):
        for j in range(y1, y2+1):
            if 0 <= i < M and 0 <= j < M:
                grid[j][i] = 1

# Tính tọa độ goal (theo cell)
goal_i = int((goal[0] - env_padding) // cell_size)
goal_j = int((goal[1] - env_padding) // cell_size)

# Tính distance map
dist = compute_distances(grid, goal_i, goal_j)

# Vẽ bản đồ
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.set_aspect("equal")
ax.set_title("Distance Map")
ax.invert_yaxis()

# Vẽ lưới và hiển thị khoảng cách trong từng ô
for i in range(M):
    for j in range(M):
        x = env_padding + i * cell_size
        y = env_padding + j * cell_size
        color = "white" if grid[j][i] == 0 else "gray"
        rect = Rectangle((x, y), cell_size, cell_size, edgecolor='lightgray', facecolor=color, linewidth=0.5)
        ax.add_patch(rect)
        if grid[j][i] == 0:
            d = dist[j][i]
            if d != float('inf'):
                ax.text(x + cell_size/2, y + cell_size/2, f"{d:.1f}", ha='center', va='center', fontsize=6, color='black')

# Vẽ Start và Goal
ax.plot(start[0], start[1], 'go', markersize=10, label="Start")
ax.plot(goal[0], goal[1], 'ro', markersize=10, label="Goal")
ax.legend()
plt.tight_layout()
plt.show()