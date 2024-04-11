import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '\\environment')
from MapData import maps


def calculateDistanceToGoal(start, goal) -> float:
    # Use diagonal distance bacause of action space
    x, y = goal[0] - start[0], goal[1] - start[1]
    return np.abs(x - y) + np.sqrt(2) * min(np.abs(x), np.abs(y))


index_algorithm = {0: "ClassicalQL", 1: "DFQL", 2: "CombinedQL", 3: "DualQL", 4: "DWA"}

scenario = input("Enter scenario (uniform/diverse/complex): ")
current_map = scenario + input("Enter map (1/2/3): ")

# Get the start and goal position
start = maps[current_map]["Start"]
goal = maps[current_map]["Goal"]

# Get the oracle values
oracle_length = calculateDistanceToGoal(start, goal)
oracle_angle = np.pi / 4
oracle_safety = 40.0

algorithm = [[] for i in range(len(index_algorithm))]

for i in range(len(index_algorithm)):
    success_length = []
    success_angle = []
    success_safety = []
    fail_counter = 0

    with open(os.path.dirname(os.path.realpath(__file__)) + f"\\{scenario}\\{current_map}\\{index_algorithm[i]}\\metric.txt", "r") as f:
        for line in f:
            data = line.split()

            # If the robot fails to reach the goal, all the metrics are 0
            if data[-1] == "Fail":
                fail_counter += 1
                success_length.append(0)
                success_angle.append(0)
                success_safety.append(0)
            else:
                success_length.append(oracle_length/ max(oracle_length, float(data[2])))
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
    print(f"{index_algorithm[i]} \t\t {algorithm[i][0]} \t {np.mean(algorithm[i][1])} \t {np.mean(algorithm[i][2])} \t {np.mean(algorithm[i][3])}")

# Save the results to xlsx
if input("Save to xlsx? (y/n): ") == "y":
    df = pd.DataFrame(algorithm, index=[index_algorithm[i] for i in range(len(index_algorithm))],
                      columns=["Success Rate(%)", "Success Length", "Success Angle", "Success Safety"])
    df.to_excel(f"{scenario}/{current_map}/LengthAngleSafety.xlsx")
    print("Saved to xlsx")