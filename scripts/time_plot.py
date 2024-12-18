import subprocess
import matplotlib.pyplot as plt

# List of image counts to test
image_counts = [1,3,6,10,15,30,60,100]  # Example values, adjust as needed

scales = []
times = []
for count in image_counts:
    # Run the match_features executable with "plot" and the current image count
    print(f"Running match_features with {count} images...")
    subprocess.run(["./bin/match_features", "plot", str(count)])
    
# File to store the results
result_file = "./res/time.txt"

# Read the current result file
with open(result_file, "r") as f_read:
    for line in f_read:
        count = int(line.split()[0])  # Get the current count
        time = float(line.split()[1])  # Directly read the time
        scales.append(count)  # Add the current count
        times.append(time)    # Add the current time

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(scales, times, marker='o')
plt.title("Processing Time vs. Number of Images")
plt.xlabel("Number of Images")
plt.ylabel("Processing Time (s)")
plt.grid(True)
# Save the image
plt.savefig("./res/processing_time_vs_images.png")