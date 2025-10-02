import pandas as pd
import numpy as np
import os
import sys
import heapq
import math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'environment'))
from environment.MapData import maps


def compute_distances(grid, goal_i, goal_j):
    """Compute shortest distances from each cell to goal using A*"""
    M = len(grid)
    INF = float('inf')
    dist = [[INF]*M for _ in range(M)]
    dist[goal_j][goal_i] = 0.0
    heap = [(0.0, goal_i, goal_j)]
    neighbours = [
        ( 1,  0, 1.0), (-1,  0, 1.0),
        ( 0,  1, 1.0), ( 0, -1, 1.0),
        ( 1,  1, math.sqrt(2)), ( 1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    while heap:
        cost, i, j = heapq.heappop(heap)
        if cost > dist[j][i]:
            continue
        for di, dj, w in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < M and 0 <= nj < M and grid[nj][ni] == 0:
                new_cost = cost + w
                if new_cost < dist[nj][ni]:
                    dist[nj][ni] = new_cost
                    heapq.heappush(heap, (new_cost, ni, nj))
    return dist


def calculateDistanceToGoal(start, goal, cell_size, env_padding, static_obstacles) -> float:
    """Calculate distance to goal using A* pathfinding like in Q-Learning algorithms"""
    # Build static occupancy grid
    M = int(512 / cell_size)  # env_size = 512
    grid = [[0]*M for _ in range(M)]
    for obs in static_obstacles:
        x1, x2, y1, y2 = obs.return_coordinate()
        i_min = max(0, int((x1 - env_padding)//cell_size))
        i_max = min(M-1, int((x2 - env_padding)//cell_size))
        j_min = max(0, int((y1 - env_padding)//cell_size))
        j_max = min(M-1, int((y2 - env_padding)//cell_size))
        for ii in range(i_min, i_max+1):
            for jj in range(j_min, j_max+1):
                grid[jj][ii] = 1

    # Precompute distances from goal
    goal_i = int((goal[0] - env_padding)//cell_size)
    goal_j = int((goal[1] - env_padding)//cell_size)
    dist_to_goal = compute_distances(grid, goal_i, goal_j)
    
    # Get distance for start position
    start_i = int((start[0] - env_padding)//cell_size)
    start_j = int((start[1] - env_padding)//cell_size)
    
    if 0 <= start_j < len(dist_to_goal) and 0 <= start_i < len(dist_to_goal[0]):
        return dist_to_goal[start_j][start_i] * cell_size
    return float('inf')


index_algorithm = {0: "ClassicalQL", 1: "DFQL", 2: "CombinedQL", 3: "DualQL", 4:"DWA"}

scenario = input("Enter scenario (uniform/diverse/complex): ")
current_map = scenario + input("Enter map (1/2/3): ")

# Get the start and goal position
start = maps[current_map]["Start"]
goal = maps[current_map]["Goal"]
static_obstacles = maps[current_map]["Obstacles"]

# Parameters for distance calculation
cell_size = 16
env_padding = int(512 * 0.06)

# Get the oracle values using correct distance calculation
oracle_length = calculateDistanceToGoal(start, goal, cell_size, env_padding, static_obstacles)
oracle_angle = np.pi / 4
oracle_safety = 40.0

algorithm = [[] for i in range(len(index_algorithm))]

for i in range(len(index_algorithm)):
    success_length = []
    success_angle = []
    success_safety = []
    fail_counter = 0

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "result", scenario, current_map, index_algorithm[i], "metric.txt"), "r") as f:
        for line in f:
            data = line.split()

            # If the robot fails to reach the goal, all the metrics are 0
            if data[-1] == "Fail":
                fail_counter += 1
                success_length.append(0)
                success_angle.append(0)
                success_safety.append(0)
            else:
                # Use the actual path length from the data instead of oracle
                actual_length = float(data[2])
                success_length.append(oracle_length / max(oracle_length, actual_length))
                success_angle.append(oracle_angle / max(oracle_angle, float(data[3])))
                success_safety.append(min(oracle_safety, float(data[4])) / oracle_safety)

    # Calculate the success rate, success length, success angle, and success safety
    algorithm[i].append(int((1 - (fail_counter / 20)) * 100 + 0.1))
    algorithm[i].append(np.mean(success_length))
    algorithm[i].append(np.mean(success_angle))
    algorithm[i].append(np.mean(success_safety))

# Print the results
print("\t\t\t Success Rate(%) \t Success Length \t Success Angle \t Success Safety")
for i in range(len(index_algorithm)):
    print(f"{index_algorithm[i]} \t\t {algorithm[i][0]} \t {algorithm[i][1]:.4f} \t {algorithm[i][2]:.4f} \t {algorithm[i][3]:.4f}")

# Save the results to xlsx
if input("Save to xlsx? (y/n): ") == "y":
    algorithm_names = [index_algorithm[i] for i in range(len(index_algorithm))]
    column_names = ["Success Rate(%)", "Success Length", "Success Angle", "Success Safety"]
    df = pd.DataFrame(algorithm)
    df.index = algorithm_names
    df.columns = column_names
    
    # Create the target directory if it doesn't exist
    target_dir = os.path.join("result", scenario, current_map)
    os.makedirs(target_dir, exist_ok=True)
    
    # Save to the result directory
    excel_path = f"/Users/fuongfotfet/Desktop/res/{current_map}/" + f"LengthAngleSafety_{current_map}.xlsx"
    df.to_excel(excel_path)
    print(f"Saved to {excel_path}")