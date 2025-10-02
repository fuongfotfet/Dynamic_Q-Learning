import matplotlib.pyplot as plt
import numpy as np
import random
import os

def process_reward_data(raw_data):
    """Process raw reward data into cumulative log rewards"""
    processed_data = [[] for i in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        # Limit to first 1000 episodes
        max_episodes = min(1000, len(raw_data[i]))
        for j in range(1, max_episodes + 1):
            # Calculate the cumulative reward
            reward = np.sum(raw_data[i][0:j]) / j
            
            # Take the natural logarithm of the cumulative reward
            if reward > 0:
                processed_data[i].append(np.log(reward + 1))
            else:
                processed_data[i].append(-np.log(-reward + 1))
    
    return processed_data

scenario = input("Enter scenario (uniform/diverse/complex): ")
current_map = scenario + input("Enter map (1/2/3): ")

index_algorithm = {0: "ClassicalQL", 1: "DFQL", 2: "CombinedQL", 3: "DualQL"}

# Choose a random run for each algorithm
run = [random.randint(1, 20) for i in range(len(index_algorithm))]
for i in range(len(index_algorithm)):
    print(f"{index_algorithm[i]}: Run {run[i]} chosen")

# Read the sum of rewards from the file
raw_data = [[] for i in range(len(index_algorithm))]
for i in range(len(index_algorithm)):
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "policy", scenario, current_map, index_algorithm[i], str(run[i]), "sumOfRewards.txt"), "r") as f:
        for line in f:
            # Read sum of rewards after removing '[' and ']' at the beginning and end of the line
            # Handle both regular floats and numpy float format
            content = line[1:-1].split(", ")
            raw_data[i] = []
            for x in content:
                # Remove 'np.float64(' and ')' if present
                if 'np.float64(' in x:
                    x = x.replace('np.float64(', '').replace(')', '')
                elif 'np.float32(' in x:
                    x = x.replace('np.float32(', '').replace(')', '')
                raw_data[i].append(float(x))

# Process the data
processed_data = process_reward_data(raw_data)

# plot the data as a line graph
plt.figure(figsize=(8.7, 7.2))  # Exact ratio to match 870x720 output (870/100 = 8.7, 720/100 = 7.2)

plt.xlabel('Episode', fontsize=24)
plt.ylabel('Natural logarithm of cumulative reward', fontsize=24)

# We change the fontsize of minor ticks label
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=18)

# Plot with better line styles and colors
plt.plot(processed_data[0], alpha=1.0, color='darkorange', lw=2, label='ClassicalQL')
plt.plot(processed_data[1], alpha=1.0, color='blue', lw=2, label='DFQL')
plt.plot(processed_data[2], alpha=1.0, color='green', lw=2, label='CombinedQL')
plt.plot(processed_data[3], alpha=1.0, color='red', lw=2, label='DualQL')

# Add legend with positioning to match the reference image (lower right corner)
plt.legend(loc='lower right', fontsize=20, frameon=True, fancybox=True, shadow=True)

# Adjust layout to prevent label cutoff
plt.subplots_adjust(left=0.22, bottom=0.12, right=0.95, top=1)

# Remove grid lines for cleaner appearance
# plt.grid(True, alpha=0.3)  # Commented out to remove grid

# Create the target directory if it doesn't exist
target_dir = f"/Users/fuongfotfet/Desktop/res/"
os.makedirs(target_dir, exist_ok=True)

# Save the plot with exact 870x720 pixel dimensions
plot_path = os.path.join(target_dir, f"Convergence_{current_map}.png")
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

plt.show()
