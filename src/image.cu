#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <vector>

#include "sift.hpp"
#include "image.hpp"


__global__ void rgb_to_grayscale_kernel(const float* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float red = rgb[3 * idx];
        float green = rgb[3 * idx + 1];
        float blue = rgb[3 * idx + 2];
        gray[idx] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
}

Image rgb_to_grayscale_cuda(const Image& img) {
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);

    // Allocate device memory
    float *d_rgb, *d_gray;
    size_t rgb_size = img.width * img.height * 3 * sizeof(float);
    size_t gray_size = img.width * img.height * sizeof(float);
    
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    // Copy RGB data to device
    cudaMemcpy(d_rgb, img.data, rgb_size, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((img.width + block.x - 1) / block.x,
              (img.height + block.y - 1) / block.y);

    // Launch kernel
    rgb_to_grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, img.width, img.height);

    // Copy result back to host
    cudaMemcpy(gray.data, d_gray, gray_size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_rgb);
    cudaFree(d_gray);

    return gray;
}


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

__global__ void check_contrast_and_extremum_kernel(const float* dog_image, const float* prev_image, 
    const float* next_image, int width, int height, int* potential_keypoints, 
    int* counter, float contrast_thresh) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (x >= width-1 || y >= height-1) return;
    
    int idx = y * width + x;
    float val = dog_image[idx];
    
    // Early contrast threshold check
    if (abs(val) < 0.8f * contrast_thresh) return;
    
    bool is_min = true, is_max = true;
    
    // Check against current, previous and next DoG images
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int curr_idx = (y + dy) * width + (x + dx);
            
            float prev_val = prev_image[curr_idx];
            float next_val = next_image[curr_idx];
            float curr_val = dog_image[curr_idx];
            
            if (prev_val > val || next_val > val || 
                (curr_val > val && !(dx == 0 && dy == 0))) {
                is_max = false;
            }
            if (prev_val < val || next_val < val || 
                (curr_val < val && !(dx == 0 && dy == 0))) {
                is_min = false;
            }
            
            if (!is_min && !is_max) return;
        }
    }
    
    if (is_min || is_max) {
        int insert_idx = atomicAdd(counter, 1);
        potential_keypoints[insert_idx * 3] = x;
        potential_keypoints[insert_idx * 3 + 1] = y;
        potential_keypoints[insert_idx * 3 + 2] = (is_max ? 1 : -1); // Store extremum type
    }
}

std::vector<Keypoint> find_keypoints_cuda(const std::vector<std::vector<Image>>& dog_octaves,
                                         float contrast_thresh, float edge_thresh) {
    std::vector<Keypoint> tmp_kps;
    
    // Allocate device memory for counter
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    
    for (int i = 0; i < dog_octaves.size(); i++) {
        const std::vector<Image>& octave = dog_octaves[i];
        for (int j = 1; j < octave.size()-1; j++) {
            const Image& curr_img = octave[j];
            const Image& prev_img = octave[j-1];
            const Image& next_img = octave[j+1];
            
            int width = curr_img.width;
            int height = curr_img.height;
            
            // Allocate device memory
            float *d_curr, *d_prev, *d_next;
            cudaMalloc(&d_curr, width * height * sizeof(float));
            cudaMalloc(&d_prev, width * height * sizeof(float));
            cudaMalloc(&d_next, width * height * sizeof(float));
            
            // Copy data to device
            cudaMemcpy(d_curr, curr_img.data, width * height * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_prev, prev_img.data, width * height * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_next, next_img.data, width * height * sizeof(float), cudaMemcpyHostToDevice);
            
            // Reset counter
            cudaMemset(d_counter, 0, sizeof(int));
            
            // Allocate memory for potential keypoints (x, y, extremum_type)
            int max_keypoints = width * height; // Maximum possible keypoints
            int* d_potential_keypoints;
            cudaMalloc(&d_potential_keypoints, max_keypoints * 3 * sizeof(int));
            
            // Launch kernel
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, 
                     (height + block.y - 1) / block.y);
            
            check_contrast_and_extremum_kernel<<<grid, block>>>(
                d_curr, d_prev, d_next, width, height,
                d_potential_keypoints, d_counter, contrast_thresh);
            
            // Get number of detected keypoints
            int num_keypoints;
            cudaMemcpy(&num_keypoints, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
            
            if (num_keypoints > 0) {
                // Allocate host memory for keypoint coordinates
                std::vector<int> keypoint_data(num_keypoints * 3);
                cudaMemcpy(keypoint_data.data(), d_potential_keypoints, 
                          num_keypoints * 3 * sizeof(int), cudaMemcpyDeviceToHost);
                
                // Process keypoints
                for (int k = 0; k < num_keypoints; k++) {
                    int x = keypoint_data[k * 3];
                    int y = keypoint_data[k * 3 + 1];
                    
                    Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                    bool kp_is_valid = refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh);
                    if (kp_is_valid) {
                        tmp_kps.push_back(kp);
                    }
                }
            }
            
            // Cleanup
            cudaFree(d_curr);
            cudaFree(d_prev);
            cudaFree(d_next);
            cudaFree(d_potential_keypoints);
        }
    }
    
    cudaFree(d_counter);
    return tmp_kps;
}

} // namespace sift

__global__ void smooth_histogram_kernel(float* hist, float* tmp_hist, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_bins) {
        int prev_idx = (idx - 1 + n_bins) % n_bins;
        int next_idx = (idx + 1) % n_bins;
        tmp_hist[idx] = (hist[prev_idx] + hist[idx] + hist[next_idx]) / 3.0f;
    }
}

void smooth_histogram_cuda(float* hist, int n_bins) {
    float *d_hist, *d_tmp_hist;
    
    // Allocate device memory
    cudaMalloc(&d_hist, n_bins * sizeof(float));
    cudaMalloc(&d_tmp_hist, n_bins * sizeof(float));
    
    // Copy histogram to device
    cudaMemcpy(d_hist, hist, n_bins * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int grid_size = (n_bins + block_size - 1) / block_size;
    
    // Perform 6 iterations of smoothing
    for (int i = 0; i < 6; i++) {
        smooth_histogram_kernel<<<grid_size, block_size>>>(d_hist, d_tmp_hist, n_bins);
        cudaMemcpy(d_hist, d_tmp_hist, n_bins * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Copy result back to host
    cudaMemcpy(hist, d_hist, n_bins * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_hist);
    cudaFree(d_tmp_hist);
}
