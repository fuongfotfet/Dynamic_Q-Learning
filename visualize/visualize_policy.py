import matplotlib.pyplot as plt
import numpy as np
from controller.CombinedQL import QLearning
from environment.MapData import maps
import matplotlib.patches as patches

# Parameters
scenario    = "uniform"
map_idx     = "1"
cell_size   = 16
env_size    = 512
env_padding = int(env_size * 0.06)
M           = env_size // cell_size

# Build static‐occupancy grid
static_obs = [o for o in maps[scenario+map_idx]["Obstacles"] if o.static]
grid = np.zeros((M, M), dtype=int)
for obs in static_obs:
    x1, x2, y1, y2 = obs.return_coordinate()
    i_min = max(0, int((x1 - env_padding)//cell_size))
    i_max = min(M-1, int((x2 - env_padding)//cell_size))
    j_min = max(0, int((y1 - env_padding)//cell_size))
    j_max = min(M-1, int((y2 - env_padding)//cell_size))
    grid[j_min:j_max+1, i_min:i_max+1] = 1

# Instantiate CombinedQL (reset() builds initial policy)
ctrl = QLearning(
    cell_size=cell_size,
    env_size=env_size,
    env_padding=env_padding,
    goal=maps[scenario+map_idx]["Goal"],
    static_obstacles=static_obs
)

# Action→vector map
action_vec = {
  "up_1": (0,-1),    "down_1": (0,1),
  "left_1": (-1,0),  "right_1": (1,0),
  "up-left_1":(-1,-1),"up-right_1":(1,-1),
  "down-left_1":(-1,1),"down-right_1":(1,1)
}

# Collect quiver data only for “no-obstacle” states AND free cells
X,Y,U,V = [],[],[],[]
for (i,j,phi,dphi,dd), act in ctrl.policy.items():
    if (phi,dphi,dd) != (-10,-10,-10): 
        continue
    if grid[j,i] == 1:  # skip walls
        continue
    dx, dy = action_vec[act]
    X.append(i); Y.append(j)
    U.append(dx); V.append(dy)

fig, ax = plt.subplots(figsize=(8,8))

# 1) draw wall‐cells
for obs in static_obs:
    x1, x2, y1, y2 = obs.return_coordinate()
    i_min = max(0, int((x1 - env_padding)//cell_size))
    i_max = min(M-1, int((x2 - env_padding)//cell_size))
    j_min = max(0, int((y1 - env_padding)//cell_size))
    j_max = min(M-1, int((y2 - env_padding)//cell_size))

    rect = patches.Rectangle(
        (i_min-0.5, j_min-0.5),
        i_max-i_min+1, j_max-j_min+1,
        facecolor='black', edgecolor='none'
    )
    ax.add_patch(rect)

# 2) overlay quiver
ax.quiver(
    X, Y, U, V,
    angles='xy', scale_units='xy', scale=1,
    color='blue', pivot='mid'
)

# 3) start & goal
start = maps[scenario+map_idx]["Start"]
goal  = maps[scenario+map_idx]["Goal"]
si = (start[0] - env_padding)/cell_size
sj = (start[1] - env_padding)/cell_size
gi = (goal[0]  - env_padding)/cell_size
gj = (goal[1]  - env_padding)/cell_size

ax.plot(si, sj, 'go', markersize=8, label='Start')
ax.plot(gi, gj, 'ro', markersize=8, label='Goal')

ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xlim(-0.5, M-0.5)
ax.set_ylim(M-0.5, -0.5)
ax.set_title("CombinedQL Initial Policy with Map Overlay")
ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()