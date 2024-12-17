#ifndef IMAGE_CU_H
#define IMAGE_CU_H

// ... existing code ...

// Convert RGB image to grayscale using CUDA
Image rgb_to_grayscale_cuda(const Image& img);

// CUDA高斯模糊函数声明
Image gaussian_blur_cuda(const Image& img, float sigma);

#endif // IMAGE_HPP