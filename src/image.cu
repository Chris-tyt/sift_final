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


__global__ void dog_kernel(const float* img1, const float* img2, float* out, int pixels) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < pixels) {
        out[idx] = img2[idx]-img1[idx];
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

            dim3 block(32,8);
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



ScaleSpacePyramid generate_dog_pyramid_cuda(const ScaleSpacePyramid& gauss_pyr) {
    ScaleSpacePyramid dog_pyr = {
        gauss_pyr.num_octaves,
        gauss_pyr.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(gauss_pyr.num_octaves)
    };

    for(int i=0; i<gauss_pyr.num_octaves; i++){
        dog_pyr.octaves[i].reserve(dog_pyr.imgs_per_octave);
        for (int j=1; j<gauss_pyr.imgs_per_octave; j++){
            const Image& img1 = gauss_pyr.octaves[i][j-1];
            const Image& img2 = gauss_pyr.octaves[i][j];
            assert(img1.channels == 1 && img2.channels == 1);
            assert(img1.width == img2.width && img1.height == img2.height);

            int pixels = img1.width * img1.height;
            size_t sz = pixels*sizeof(float);
            float *d_img1, *d_img2, *d_out;
            cudaMalloc(&d_img1, sz);
            cudaMalloc(&d_img2, sz);
            cudaMalloc(&d_out, sz);

            cudaMemcpy(d_img1, img1.data, sz, cudaMemcpyHostToDevice);
            cudaMemcpy(d_img2, img2.data, sz, cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((pixels+255)/256);
            dog_kernel<<<grid,block>>>(d_img1,d_img2,d_out,pixels);
            cudaDeviceSynchronize();

            Image diff(img1.width, img1.height, 1);
            cudaMemcpy(diff.data, d_out, sz, cudaMemcpyDeviceToHost);

            dog_pyr.octaves[i].push_back(diff);

            cudaFree(d_img1);
            cudaFree(d_img2);
            cudaFree(d_out);
        }
    }
    return dog_pyr;
}
} // namespace sift