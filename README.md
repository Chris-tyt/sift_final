# sift-cpp

## Introduction
This is a C++ implementation of [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), a feature detection algorithm that identifies and describes local features in images. SIFT is widely used in computer vision tasks such as object recognition, image stitching, and 3D reconstruction. This implementation is based on the work of [dbarac](https://github.com/dbarac/sift-cpp) and aims to provide a straightforward interface for researchers and developers.

## Features
- **Robustness**: SIFT is invariant to scaling, rotation, and partially invariant to changes in illumination and 3D viewpoint.
- **Efficiency**: The algorithm is optimized for performance, making it suitable for real-time applications.
- **Flexibility**: Users can easily modify the code to suit their specific needs.

## Libraries used
This project utilizes the following libraries:
- [stb_image](https://github.com/nothings/stb) and stb_image_write for loading and saving images. (included in this repo)

## Data example
Store data under `data/images`. Use the script `generate_list.py` to generate a baseline list of images for processing.

## Usage example
To run the SIFT algorithm, you can use the following command:
- `mode` can be either `test` or `plot`; other values are invalid.
- `image_num` is the number of images to process; `0` means all images.

```bash
$ ./scripts/run.sh
```

## Verify example
To verify the implementation, you can run the following command:

```bash
$ bash scripts/verify.sh
```

## Useful links
- [SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [OpenCV SIFT documentation](https://docs.opencv.org/4.x/da/af0/tutorial_intro_to_sift.html)