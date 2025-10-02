import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from environment.MapData import maps  # chứa biến maps đã định nghĩa
from environment.Obstacle import Obstacle

# Chọn bản đồ
map_data = maps["complex2"]
start = map_data["Start"]
goal = map_data["Goal"]
obstacles = map_data["Obstacles"]

# Thiết lập figure
fig, ax = plt.subplots(figsize=(8, 8))

# Vẽ lưới trục tọa độ
ax.set_xlim(0, 600)
ax.set_ylim(0, 600)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Map: uniform1")
ax.set_aspect("equal")
ax.grid(True, linestyle='--', alpha=0.5)
ax.invert_yaxis()

# Vẽ obstacle
for obs in obstacles:
    x = obs.x - obs.width / 2
    y = obs.y - obs.height / 2
    color = "gray" if obs.static else "blue"
    rect = Rectangle((x, y), obs.width, obs.height, linewidth=1, edgecolor='black', facecolor=color, alpha=0.8)
    ax.add_patch(rect)

# Vẽ điểm Start và Goal
ax.plot(start[0], start[1], 'go', label="Start", markersize=10)
ax.plot(goal[0], goal[1], 'ro', label="Goal", markersize=10)

# Hiển thị legend và plot
ax.legend()
plt.tight_layout()
plt.show()