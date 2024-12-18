#!/bin/bash
cd build
cmake ..
make
cd ..
# mode can be either test or plot; other values are invalid
mode=test
# image_num is the number of images to process; 0 means all images
image_num=10

# Run the script bin/match_festures, passing mode and image_num as parameters
if [ "$image_num" -ne 0 ]; then
    bin/sift_test "$mode" "$image_num"
else
    bin/sift_test "$mode"
fi