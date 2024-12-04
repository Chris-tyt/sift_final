# sift-cpp

## Introduction
This is a C++ implementation of [SIFT], a feature detection algorithm.
Based on https://github.com/dbarac/sift-cpp

## Libraries used
[stb_image](https://github.com/nothings/stb) and stb_image_write for loading and saving images. (included in this repo)

## Data example
store data under 'data/images', use scripts 'generate_list.py' to generate base line list.

## Usage example
mode can be either test or plot; other values are invalid
image_num is the number of images to process; 0 means all images
```bash
$ ./scripts/run.sh
```

## Verify example
```bash
$ bash scripts/verify.sh
```

## Useful links

* [SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)