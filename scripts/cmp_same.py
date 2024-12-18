import filecmp
import sys

def compare_files(file1, file2):
    # Read files and store lines in sets to ignore order
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = set(f1.readlines())
        lines2 = set(f2.readlines())
    
    # Find differences
    diff1 = lines1 - lines2  # Lines in file1 but not in file2
    diff2 = lines2 - lines1  # Lines in file2 but not in file1
    
    if diff1 or diff2:
        print("Different lines:")
        if diff1:
            print(f"\nIn {file1} but not in {file2}:")
            for line in diff1:
                print(line.rstrip())
        if diff2:
            print(f"\nIn {file2} but not in {file1}:")
            for line in diff2:
                print(line.rstrip())
        print(f"\nTotal different lines: {len(diff1) + len(diff2)}")
        return False
    return True

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
