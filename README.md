# sift-cpp

## Introduction
This is a C++ implementation of [SIFT], a feature detection algorithm.
Based on https://github.com/dbarac/sift-cpp

## Libraries used
[stb_image](https://github.com/nothings/stb) and stb_image_write for loading and saving images. (included in this repo)

## Usage example
store data under 'data/images', use scripts 'generate_list.py' to generate base line list.


## Build and run the examples
### Build
```bash
$ mkdir build/ && cd build && cmake .. && make
```
The executables will be in sift-cpp/bin/.

### Run
Find image keypoints, draw them and save the result:
```bash
$ cd bin/ && ./find_keypoints ../imgs/book_rotated.jpg
```
Input images can be .jpg or .png. Result image is saved as result.jpg

Find keypoints in two images and match them, draw matches and save the result:
```bash
$ cd bin/ && ./match_features ../imgs/book_rotated.jpg ../imgs/book_in_scene.jpg
```
Result image is saved as result.jpg

## Useful links

* [SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)