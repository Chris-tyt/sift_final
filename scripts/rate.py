import matplotlib.pyplot as plt
import numpy as np

# 读取文件数据
def read_data(filename):
    nums = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            n, t = map(float, line.strip().split())
            nums.append(n)
            times.append(t)
    return nums, times

# 读取两个文件的数据
time_nums, time_times = read_data('res/time.txt')
quick_nums, quick_times = read_data('res/quick.txt')

# 计算加速比
speedup = [t1/t2 for t1, t2 in zip(time_times, quick_times)]

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(time_nums, speedup, 'b-o', linewidth=2)
plt.grid(True)
plt.xlabel('Number of Images')
plt.ylabel('Speedup Ratio')
plt.title('Performance Speedup Analysis')

# 保存图表
plt.savefig('speedup.png')
plt.show()
