import matplotlib.pyplot as plt
import numpy as np

path = '../result/real/real15'
raw_data = np.loadtxt(f'{path}/QLver2/sumOfRewards.txt', delimiter=",")
raw_data_ver2 = np.loadtxt(f'{path}/QLver3/sumOfRewards.txt', delimiter=",")
raw_data_ver3 = np.loadtxt(f'{path}/QLver4/sumOfRewards.txt', delimiter=",")

# new data equat sum of 12 elements in raw_data
new_data = []
for i in range(len(raw_data)):
    new_data.append(np.sum(raw_data[0:i]) / (i + 1))


new_data_ver2 = []
for i in range(1, len(raw_data_ver2)):
    new_data_ver2.append(np.sum(raw_data_ver2[0:i]) / (i + 1))


new_data_ver3 = []
for i in range(1, len(raw_data_ver3)):
    new_data_ver3.append(np.sum(raw_data_ver3[0:i]) / (i + 1))


# plot the data as a line graph
plt.xlabel('Epoch', fontsize='20')
plt.ylabel('Sum of Reward', fontsize='20')
plt.title('Sum of Reward of Q-Learning and Improved Q-Learning', fontsize='20')

# We change the fontsize of minor ticks label
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=12)

plt.plot(new_data, alpha=1.0, color='darkorange')
plt.plot(new_data_ver2, alpha=1.0, color='blue')
plt.plot(new_data_ver3, alpha=1.0, color='green')
# plt.plot(raw_data, alpha=0.2, color='darkorange')
# plt.plot(raw_data_ver2, alpha=0.2, color='blue')
# plt.plot(raw_data_ver3, alpha=0.2, color='green')

plt.legend(['Q-Learning ver2', 'Q-Learning ver3', 'Q-Learning ver4'], loc='upper left', fontsize='20')
plt.show()