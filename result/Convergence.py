import matplotlib.pyplot as plt
import numpy as np
import random

scenario = input("Enter scenario (uniform/diverse/complex): ")
current_map = scenario + input("Enter map (1/2/3): ")

index_algorithm = {0: "ConventionalQL", 1: "DFQL", 2: "CombinedQL", 3: "DualQL"}

# Choose a random run for each algorithm
run = [random.randint(1, 20) for i in range(4)]
for i in range(4):
    print(f"{index_algorithm[i]}: Run {run[i]} chosen")

# Read the sum of rewards from the file
raw_data = [[] for i in range(4)]
for i in range(4):
    with open(f"../policy/{scenario}/{current_map}/{index_algorithm[i]}/{run[i]}/sumOfRewards.txt", "r") as f:
        for line in f:
            # Read sum of rewards after removing '[' and ']' at the beginning and end of the line
            raw_data[i] = [float(x) for x in line[1:-1].split(", ")]

# Process the data
processed_data = [[] for i in range(4)]
for i in range(4):
    for j in range(1, len(raw_data[i]) + 1):
        # Calculate the cumulative reward
        reward = np.sum(raw_data[i][0:j]) / j

        # Take the natural logarithm of the cumulative reward
        if reward > 0:
            processed_data[i].append(np.log(reward))
        else:
            processed_data[i].append(-np.log(-reward))

# plot the data as a line graph
plt.xlabel('Epoch', fontsize='20')
plt.ylabel('Cumulative Reward', fontsize='20')
plt.title('Cumulative Reward of 4 QL approaches', fontsize='20')

# We change the fontsize of minor ticks label
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.plot(processed_data[0], alpha=1.0, color='darkorange')
plt.plot(processed_data[1], alpha=1.0, color='blue')
plt.plot(processed_data[2], alpha=1.0, color='green')
plt.plot(processed_data[3], alpha=1.0, color='darkred')
# plt.plot(raw_data[0], alpha=0.2, color='darkorange')
# plt.plot(raw_data[1], alpha=0.2, color='blue')
# plt.plot(raw_data[2], alpha=0.2, color='green')
# plt.plot(raw_data[3], alpha=0.2, color='darkred')

plt.legend([index_algorithm[i] for i in range(4)], loc='lower right', fontsize='20')
plt.show()