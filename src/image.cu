#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <vector>

#include "sift.hpp"
#include "image.hpp"



__global__ void gaussian_blur_horizontal(float* input, float* output, float* kernel,
                                       int width, int height, int kernel_size, int center) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int dx = -center + k;
            int src_x = min(max(x + dx, 0), width - 1);
            sum += input[y * width + src_x] * kernel[k];
        }
        output[y * width + x] = sum;
    }
}

__global__ void gaussian_blur_vertical(float* input, float* output, float* kernel,
                                     int width, int height, int kernel_size, int center) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int dy = -center + k;
            int src_y = min(max(y + dy, 0), height - 1);
            sum += input[src_y * width + x] * kernel[k];
        }
        output[y * width + x] = sum;
    }
}

Image gaussian_blur_cuda(const Image& img, float sigma) 
{
    assert(img.channels == 1);
    
    // 计算高斯核
    int size = std::ceil(6 * sigma);
    if (size % 2 == 0) size++;
    int center = size / 2;
    
    std::vector<float> h_kernel(size);
    float sum = 0.0f;
    for (int k = -size/2; k <= size/2; k++) {
        float val = std::exp(-(k*k) / (2*sigma*sigma));
        h_kernel[center + k] = val;
        sum += val;
    }
    for (int k = 0; k < size; k++) {
        h_kernel[k] /= sum;
    }
    
    // 分配设备内存
    float *d_input, *d_temp, *d_output, *d_kernel;
    cudaMalloc(&d_input, img.width * img.height * sizeof(float));
    cudaMalloc(&d_temp, img.width * img.height * sizeof(float));
    cudaMalloc(&d_output, img.width * img.height * sizeof(float));
    cudaMalloc(&d_kernel, size * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_input, img.data, img.width * img.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置grid和block大小
    dim3 block(16, 16);
    dim3 grid((img.width + block.x - 1) / block.x, 
              (img.height + block.y - 1) / block.y);
    
    // 执行水平和垂直方向的卷积
    gaussian_blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel,
                                            img.width, img.height, size, center);
    gaussian_blur_vertical<<<grid, block>>>(d_temp, d_output, d_kernel,
                                          img.width, img.height, size, center);
    
    // 创建输出图像
    Image result(img.width, img.height, 1);
    
    // 复制结果回主机
    cudaMemcpy(result.data, d_output, img.width * img.height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    return result;
} 


__global__ void compute_gradient_kernel(const float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = 0.5f * (input[y*width + (x+1)] - input[y*width + (x-1)]);
        float gy = 0.5f * (input[(y+1)*width + x] - input[(y-1)*width + x]);

        // output前一半存gx，后一半存gy
        output[y*width + x] = gx;  
        output[width*height + y*width + x] = gy;
    } else if (x < width && y < height) {
        // 边界处可设为0或保持默认值
        output[y*width + x] = 0.0f;
        output[width*height + y*width + x] = 0.0f;
    }
}

namespace sift {

ScaleSpacePyramid generate_gradient_pyramid_cuda(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };

    // 为每张图像分配GPU内存进行处理
    for (int i = 0; i < pyramid.num_octaves; i++) {
        grad_pyramid.octaves[i].reserve(pyramid.imgs_per_octave);
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            const Image& in_img = pyramid.octaves[i][j];
            int width = in_img.width;
            int height = in_img.height;
            assert(in_img.channels == 1);

            Image grad(width, height, 2);

            float *d_input, *d_output;
            size_t img_size = width * height * sizeof(float);
            cudaMalloc(&d_input, img_size);
            cudaMalloc(&d_output, img_size * 2); // 两个通道

            // 复制输入数据到GPU
            cudaMemcpy(d_input, in_img.data, img_size, cudaMemcpyHostToDevice);

            dim3 block(16,16);
            dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

            compute_gradient_kernel<<<grid, block>>>(d_input, d_output, width, height);
            cudaDeviceSynchronize();

            // 拷贝结果回CPU
            cudaMemcpy(grad.data, d_output, img_size*2, cudaMemcpyDeviceToHost);

            // 释放GPU内存
            cudaFree(d_input);
            cudaFree(d_output);

            grad_pyramid.octaves[i].push_back(grad);
        }
    }

    return grad_pyramid;
}
} // namespace sift