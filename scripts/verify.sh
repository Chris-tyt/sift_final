#!/bin/bash

# Define the paths for the list files and match data folders
list_file1="res/matching_list.txt"
list_file2="res/matching_list_base.txt"
match_data_folder1="res/matching_coordinates.txt"
match_data_folder2="res/matching_coordinates_base.txt"

# Output message for verifying list files
echo "Verifying list files..."
python3 scripts/cmp_same.py "$list_file1" "$list_file2"

# Output message for verifying match data
echo "Verifying match data..."
python3 scripts/cmp_deviation.py "$match_data_folder2" "$match_data_folder1"
python3 scripts/cmp_deviation.py "$match_data_folder1" "$match_data_folder2"