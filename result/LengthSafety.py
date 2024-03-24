import matplotlib.pyplot as plt
import numpy as np

length = [[] for i in range(2)]
safety = [[] for j in range(2)]

with open('../controller/metric.txt', 'r') as file:
    datas = file.readlines()

for data in datas:
    if data[0] == '-':
        continue

    split_data = data.split()

    if split_data[1] == 'version2':
        if split_data[-1] == 'Fail':
            length[0].append(0)
            safety[0].append(0)
        else:
            length[0].append(float(split_data[2]))
            safety[0].append(float(split_data[3]))

    elif split_data[1] == 'version3':
        if split_data[-1] == 'Fail':
            length[1].append(0)
            safety[1].append(0)
        else:
            length[1].append(float(split_data[2]))
            safety[1].append(float(split_data[3]))

# plot the length as a bar graph, 2 bars for each map
barWidth = 0.25

br1 = np.arange(len(length[0]))
br2 = [x + barWidth for x in br1]

plt.bar(br1, length[0], color='r', width=barWidth, edgecolor='grey', label='QLver2')
plt.bar(br2, length[1], color='b', width=barWidth, edgecolor='grey', label='QLver3')

plt.xlabel('Map', fontweight='bold', fontsize='15')
plt.ylabel('Length', fontweight='bold', fontsize='15')
plt.xticks([r + barWidth for r in range(len(length[0]))], ['Map' + str(i) for i in range(1, len(length[0]) + 1)])

plt.legend(fontsize='20')
plt.show()

# plot the safety as a bar graph, 2 bars for each map
br1 = np.arange(len(safety[0]))
br2 = [x + barWidth for x in br1]

plt.bar(br1, safety[0], color='r', width=barWidth, edgecolor='grey', label='version2')
plt.bar(br2, safety[1], color='b', width=barWidth, edgecolor='grey', label='version3')

plt.xlabel('Map', fontweight='bold', fontsize='15')
plt.ylabel('Safety', fontweight='bold', fontsize='15')
plt.xticks([r + barWidth for r in range(len(safety[0]))], ['Map' + str(i) for i in range(1, len(safety[0]) + 1)])

plt.legend(fontsize='20')
plt.show()