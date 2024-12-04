import subprocess
import matplotlib.pyplot as plt

# List of image counts to test
image_counts = [10, 20, 30, 40, 50]  # Example values, adjust as needed

# File to store the results
result_file = "../res/time.txt"

# Open the result file for writing
# with open(result_file, "w") as f:  # 删除写入结果文件的部分
scales = []
times = []
for count in image_counts:
    # Run the match_features executable with "plot" and the current image count
    print(f"Running match_features with {count} images...")
    subprocess.run(["./bin/match_features", "plot", str(count)])  # 更新路径以指向bin文件夹
    
# 读取当前结果文件
with open(result_file, "r") as f_read:
    for line in f_read:
        count = int(line.split()[0])  # 获取当前的count
        time = float(line.split()[1])  # 直接读取时间
        scales.append(count)  # 添加当前的count
        times.append(time)    # 添加当前的时间

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(scales, times, marker='o')
plt.title("Processing Time vs. Number of Images")
plt.xlabel("Number of Images")
plt.ylabel("Processing Time (s)")
plt.grid(True)
# Save the image
plt.savefig("./res/processing_time_vs_images.png")