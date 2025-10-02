import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from environment.MapData import maps
from controller.DQNController import DQNLearning

def visualize_dqn_distance_map(scenario="uniform", map_num="1"):
    """
    Visualize the distance map computed by DQNController initialization
    """
    # Configuration
    cell_size = 16
    env_size = 512
    env_padding = int(env_size * 0.06)  # ~30 pixels
    
    # Load map data
    map_key = scenario + map_num
    if map_key not in maps:
        print(f"Map {map_key} not found!")
        return
    
    map_data = maps[map_key]
    start = map_data["Start"]
    goal = map_data["Goal"]
    obstacles = map_data["Obstacles"]
    
    print(f"Visualizing DQN Distance Map for {map_key}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Environment: {env_size}x{env_size}, Cell size: {cell_size}, Padding: {env_padding}")
    
    # Get static obstacles only (like DQN does)
    static_obstacles = [obs for obs in obstacles if obs.static]
    print(f"Static obstacles: {len(static_obstacles)}/{len(obstacles)}")
    
    # Initialize DQN controller to get distance matrix
    print("Initializing DQN Controller...")
    dqn_controller = DQNLearning(
        cell_size=cell_size,
        env_size=env_size, 
        env_padding=env_padding,
        goal=goal,
        static_obstacles=static_obstacles
    )
    
    # Get the computed distance matrix
    distance_matrix = dqn_controller.distance_matrix
    M = len(distance_matrix)  # Should be 32x32 for 512/16
    
    print(f"Distance matrix size: {M}x{M}")
    print(f"Goal position in grid: ({int((goal[0] - env_padding)//cell_size)}, {int((goal[1] - env_padding)//cell_size)})")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Distance map with values
    ax1.set_xlim(0, env_size)
    ax1.set_ylim(0, env_size)
    ax1.set_aspect("equal")
    ax1.set_title(f"DQN Distance Map - {map_key}\n(Values shown)")
    ax1.invert_yaxis()
    
    # Right plot: Distance map as heatmap
    ax2.set_xlim(0, env_size)
    ax2.set_ylim(0, env_size)
    ax2.set_aspect("equal")
    ax2.set_title(f"DQN Distance Map - {map_key}\n(Heatmap)")
    ax2.invert_yaxis()
    
    # Prepare heatmap data
    heatmap_data = np.zeros((M, M))
    
    # Draw grid and distances
    for i in range(M):
        for j in range(M):
            # Convert grid coordinates to pixel coordinates
            x = env_padding + i * cell_size
            y = env_padding + j * cell_size
            
            # Get distance value
            dist_val = distance_matrix[j][i]
            
            # Determine if this cell is obstacle
            is_obstacle = (dist_val == float('inf'))
            
            # Left plot: Show values
            if is_obstacle:
                color = "black"
                rect1 = Rectangle((x, y), cell_size, cell_size, 
                                edgecolor='gray', facecolor=color, linewidth=0.5)
                ax1.add_patch(rect1)
                ax1.text(x + cell_size/2, y + cell_size/2, "∞", 
                        ha='center', va='center', fontsize=8, color='white', weight='bold')
                heatmap_data[j][i] = np.nan  # NaN for obstacles in heatmap
            else:
                # Color based on distance (lighter = closer to goal)
                if dist_val == 0:
                    color = "red"  # Goal
                else:
                    # Scale color from white (close) to blue (far)
                    max_dist = np.max([d for row in distance_matrix for d in row if d != float('inf')])
                    intensity = 1 - (dist_val / max_dist)
                    color = plt.cm.Blues(intensity)
                
                rect1 = Rectangle((x, y), cell_size, cell_size, 
                                edgecolor='gray', facecolor=color, linewidth=0.5)
                ax1.add_patch(rect1)
                
                # Show distance value
                if dist_val < 10:
                    ax1.text(x + cell_size/2, y + cell_size/2, f"{dist_val:.1f}", 
                            ha='center', va='center', fontsize=6, color='black')
                else:
                    ax1.text(x + cell_size/2, y + cell_size/2, f"{dist_val:.0f}", 
                            ha='center', va='center', fontsize=6, color='black')
                
                heatmap_data[j][i] = dist_val
    
    # Right plot: Heatmap
    im = ax2.imshow(heatmap_data, extent=[env_padding, env_padding + M*cell_size, 
                                         env_padding + M*cell_size, env_padding], 
                   cmap='Blues_r', origin='upper', interpolation='nearest')
    
    # Add colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Distance to Goal', rotation=270, labelpad=15)
    
    # Draw obstacles on heatmap
    for obs in static_obstacles:
        x1, x2, y1, y2 = obs.return_coordinate()
        width = x2 - x1
        height = y2 - y1
        rect2 = Rectangle((x1, y1), width, height, 
                         edgecolor='red', facecolor='black', linewidth=1, alpha=0.8)
        ax2.add_patch(rect2)
    
    # Mark start and goal on both plots
    for ax in [ax1, ax2]:
        ax.plot(start[0], start[1], 'go', markersize=12, label="Start", markeredgecolor='black', markeredgewidth=2)
        ax.plot(goal[0], goal[1], 'ro', markersize=12, label="Goal", markeredgecolor='black', markeredgewidth=2)
        ax.legend(loc='upper right')
        
        # Add grid lines
        for i in range(M+1):
            ax.axvline(x=env_padding + i*cell_size, color='gray', linewidth=0.3, alpha=0.5)
            ax.axhline(y=env_padding + i*cell_size, color='gray', linewidth=0.3, alpha=0.5)
    
    # Print some statistics
    finite_distances = [d for row in distance_matrix for d in row if d != float('inf')]
    print(f"\nDistance Statistics:")
    print(f"Min distance: {min(finite_distances):.2f}")
    print(f"Max distance: {max(finite_distances):.2f}")
    print(f"Mean distance: {np.mean(finite_distances):.2f}")
    print(f"Reachable cells: {len(finite_distances)}/{M*M} ({100*len(finite_distances)/(M*M):.1f}%)")
    
    plt.tight_layout()
    
    # Save the plot instead of showing
    output_path = f"visualize/dqn_distance_map_{scenario}{map_num}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distance map saved to: {output_path}")
    plt.close()
    
    return dqn_controller, distance_matrix

def compare_scenarios():
    """Compare distance maps across different scenarios"""
    scenarios = [("uniform", "1"), ("diverse", "1"), ("complex", "1")]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for idx, (scenario, map_num) in enumerate(scenarios):
        ax = axes[idx]
        
        # Load map and create controller
        map_key = scenario + map_num
        map_data = maps[map_key]
        goal = map_data["Goal"]
        static_obstacles = [obs for obs in map_data["Obstacles"] if obs.static]
        
        # Initialize DQN controller
        dqn_controller = DQNLearning(
            cell_size=16, env_size=512, env_padding=int(512*0.06),
            goal=goal, static_obstacles=static_obstacles
        )
        
        distance_matrix = dqn_controller.distance_matrix
        M = len(distance_matrix)
        
        # Create heatmap data
        heatmap_data = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                dist_val = distance_matrix[j][i]
                heatmap_data[j][i] = dist_val if dist_val != float('inf') else np.nan
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='Blues_r', origin='upper', interpolation='nearest')
        ax.set_title(f"{scenario.capitalize()} {map_num}")
        ax.set_aspect('equal')
        
        # Mark start and goal
        start = map_data["Start"]
        start_grid = ((start[0] - int(512*0.06))//16, (start[1] - int(512*0.06))//16)
        goal_grid = ((goal[0] - int(512*0.06))//16, (goal[1] - int(512*0.06))//16)
        
        ax.plot(start_grid[0], start_grid[1], 'go', markersize=10, markeredgecolor='black')
        ax.plot(goal_grid[0], goal_grid[1], 'ro', markersize=10, markeredgecolor='black')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Visualize specific map
    print("=== DQN Distance Map Visualization ===")
    
    # Single map visualization
    controller, dist_matrix = visualize_dqn_distance_map("uniform", "1")
    
    # Compare across scenarios
    print("\n=== Comparing Different Scenarios ===")
    compare_scenarios()
