import filecmp
import sys

def compare_files(file1, file2):
    # Use filecmp.cmp to compare the two files
    return filecmp.cmp(file1, file2, shallow=False)

# Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python cmp.py <file1_path> <file2_path>")
    sys.exit(1)

# Get file paths from command-line arguments
file1_path = sys.argv[1]
file2_path = sys.argv[2]

if compare_files(file1_path, file2_path):
    print("The two files are the same.")
else:
    print("The two files are different.")
