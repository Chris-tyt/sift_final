#!/bin/bash
cd build
cmake ..
make
cd ..
# mode can be either test or plot; other values are invalid
mode=plot
# image_num is the number of images to process; 0 means all images
image_num=2

# Run the script bin/match_festures, passing mode and image_num as parameters
if [ "$image_num" -ne 0 ]; then
    bin/match_features "$mode" "$image_num"
else
    bin/match_features "$mode"
fi