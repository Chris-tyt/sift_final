import filecmp
import sys

def compare_files(file1, file2, tolerance=0.01):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    
    # Convert lines to lists of tuples (not sets) to maintain order
    processed1 = []
    processed2 = []
    
    # Process only lines with exactly 4 numbers
    for line in lines1:
        try:
            parts = line.replace('Matched Image:', '').split(')')
            numbers = []
            for part in parts:
                if '(' in part:
                    nums = part.split('(')[1].strip().split(',')
                    numbers.extend([float(n.strip()) for n in nums if n.strip()])
            if len(numbers) == 4:  # 确保我们得到了4个数字
                processed1.append(tuple(numbers))
        except ValueError:
            continue
    
    for line in lines2:
        try:
            parts = line.replace('Matched Image:', '').split(')')
            numbers = []
            for part in parts:
                if '(' in part:
                    nums = part.split('(')[1].strip().split(',')
                    numbers.extend([float(n.strip()) for n in nums if n.strip()])
            if len(numbers) == 4:
                processed2.append(tuple(numbers))
        except ValueError:
            continue

    # Compare numbers with tolerance
    unmatched1 = []
    
    for nums1 in processed1:
        found_match = False
        for nums2 in processed2:
            if all(abs(n1 - n2) < tolerance for n1, n2 in zip(nums1, nums2)):
                found_match = True
                break
        if not found_match:
            unmatched1.append(nums1)
    
    if unmatched1:
        print(f"\nLines in {file1} that have no match in {file2}:")
        print(f"Tolerance: {tolerance}")
        for line in unmatched1:
            print(f"({line[0]}, {line[1]}) ({line[2]}, {line[3]})")
        print(f"\nTotal unmatched lines: {len(unmatched1)}")
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
