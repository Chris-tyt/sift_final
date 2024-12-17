// sift.cu
#include <cuda_runtime.h>
#include "sift_cu.h"

namespace sift {

// 辅助函数：检查CUDA错误
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
template<typename T>
void check_cuda_error(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// CUDA kernels
__global__ void generate_dog_kernel(float* dog_data, const float* img_data,
                                  int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        dog_data[idx] = img_data[idx + width * height] - img_data[idx];
    }
}

__global__ void detect_extrema_kernel(int* extrema_map, const float* dog_data,
                                    int width, int height, float contrast_thresh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
        float val = dog_data[y*width + x];
        if (fabs(val) < 0.8f*contrast_thresh)
            return;
            
        bool is_max = true;
        bool is_min = true;
        
        // Check 26 neighbors
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int ds = -1; ds <= 1; ds++) {
                    if (dx == 0 && dy == 0 && ds == 0)
                        continue;
                        
                    float neighbor = dog_data[(y+dy)*width + (x+dx) + ds*width*height];
                    if (neighbor >= val)
                        is_max = false;
                    if (neighbor <= val)
                        is_min = false;
                }
            }
        }
        
        if (is_max || is_min)
            extrema_map[y*width + x] = 1;
    }
}

__global__ void generate_gradient_kernel(float* grad_data, const float* img_data,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
        int idx = y * width + x;
        // Calculate gradients
        float gx = (img_data[idx + 1] - img_data[idx - 1]) * 0.5f;
        float gy = (img_data[idx + width] - img_data[idx - width]) * 0.5f;
        
        // Store gradients (interleaved format)
        grad_data[2 * idx] = gx;
        grad_data[2 * idx + 1] = gy;
    }
}

__global__ void compute_orientation_histogram_kernel(float* hist, const float* grad_data,
                                                  int width, int height,
                                                  float x, float y, float sigma,
                                                  float lambda_ori) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float patch_sigma = lambda_ori * sigma;
        float dx = idx - x;
        float dy = idy - y;
        float dist_sq = dx*dx + dy*dy;
        
        if (dist_sq <= 9.0f * patch_sigma * patch_sigma) {
            int grad_idx = 2 * (idy * width + idx);
            float gx = grad_data[grad_idx];
            float gy = grad_data[grad_idx + 1];
            
            float magnitude = sqrtf(gx*gx + gy*gy);
            float theta = atan2f(gy, gx);
            if (theta < 0) theta += 2*M_PI;
            
            float weight = expf(-dist_sq/(2*patch_sigma*patch_sigma));
            
            int bin = (int)(N_BINS * theta / (2*M_PI));
            bin = min(bin, N_BINS-1);
            
            atomicAdd(&hist[bin], magnitude * weight);
        }
    }
}

__global__ void compute_descriptor_kernel(float* descriptor_hist, const float* grad_data,
                                        int width, int height, float x, float y,
                                        float sigma, float theta, float lambda_desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);
        
        // 计算相对于关键点的旋转后坐标
        float dx = ((idx - x)*cos_t + (idy - y)*sin_t) / sigma;
        float dy = (-(idx - x)*sin_t + (idy - y)*cos_t) / sigma;
        
        if (fmaxf(fabsf(dx), fabsf(dy)) <= lambda_desc*(N_HIST+1.)/N_HIST) {
            int grad_idx = 2 * (idy * width + idx);
            float gx = grad_data[grad_idx];
            float gy = grad_data[grad_idx + 1];
            
            // 计算梯度方向和大小
            float magnitude = sqrtf(gx*gx + gy*gy);
            float angle = atan2f(gy, gx) - theta;
            if (angle < 0) angle += 2*M_PI;
            
            // 计算贡献权重
            float gaussian_weight = expf(-(dx*dx + dy*dy)/(2*(lambda_desc*lambda_desc)));
            
            // 更新直方图
            for (int i = 0; i < N_HIST; i++) {
                for (int j = 0; j < N_HIST; j++) {
                    float cx = ((float)i - (N_HIST-1)/2.0f) * 2.0f*lambda_desc/N_HIST;
                    float cy = ((float)j - (N_HIST-1)/2.0f) * 2.0f*lambda_desc/N_HIST;
                    
                    float hist_weight = (1.0f - N_HIST*fabsf(dx-cx)/(2.0f*lambda_desc)) *
                                      (1.0f - N_HIST*fabsf(dy-cy)/(2.0f*lambda_desc));
                    
                    if (hist_weight > 0) {
                        for (int k = 0; k < N_ORI; k++) {
                            float theta_k = 2.0f*M_PI*k/N_ORI;
                            float theta_diff = fmodf(angle - theta_k + 2.0f*M_PI, 2.0f*M_PI);
                            if (theta_diff < 2.0f*M_PI/N_ORI) {
                                float bin_weight = 1.0f - N_ORI*theta_diff/(2.0f*M_PI);
                                int hist_idx = (i*N_HIST + j)*N_ORI + k;
                                atomicAdd(&descriptor_hist[hist_idx],
                                        hist_weight * bin_weight * magnitude * gaussian_weight);
                            }
                        }
                    }
                }
            }
        }
    }
}

ScaleSpacePyramid generate_dog_pyramid_cuda(const ScaleSpacePyramid& img_pyramid) {
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        int width = img_pyramid.octaves[i][0].width;
        int height = img_pyramid.octaves[i][0].height;
        int size = width * height;
        
        float *d_img_data, *d_dog_data;
        CHECK_CUDA_ERROR(cudaMalloc(&d_img_data, size * img_pyramid.imgs_per_octave * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_dog_data, size * (img_pyramid.imgs_per_octave-1) * sizeof(float)));
        
        // Copy image data to device
        for (int j = 0; j < img_pyramid.imgs_per_octave; j++) {
            CHECK_CUDA_ERROR(cudaMemcpy(d_img_data + j*size, img_pyramid.octaves[i][j].data,
                           size * sizeof(float), cudaMemcpyHostToDevice));
        }
        
        // Launch kernel
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        for (int j = 0; j < img_pyramid.imgs_per_octave-1; j++) {
            generate_dog_kernel<<<grid, block>>>(d_dog_data + j*size,
                                               d_img_data + j*size,
                                               width, height);
        }
        
        // Copy results back and create Images
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 0; j < dog_pyramid.imgs_per_octave; j++) {
            Image dog_img(width, height, 1);
            CHECK_CUDA_ERROR(cudaMemcpy(dog_img.data, d_dog_data + j*size,
                           size * sizeof(float), cudaMemcpyDeviceToHost));
            dog_pyramid.octaves[i].push_back(std::move(dog_img));
        }
        
        // Cleanup
        CHECK_CUDA_ERROR(cudaFree(d_img_data));
        CHECK_CUDA_ERROR(cudaFree(d_dog_data));
    }
    
    return dog_pyramid;
}

ScaleSpacePyramid generate_gradient_pyramid_cuda(const ScaleSpacePyramid& pyramid) {
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    
    for (int i = 0; i < pyramid.num_octaves; i++) {
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            float *d_img_data, *d_grad_data;
            CHECK_CUDA_ERROR(cudaMalloc(&d_img_data, width * height * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_data, width * height * 2 * sizeof(float)));
            
            // Copy image to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_img_data, pyramid.octaves[i][j].data,
                           width * height * sizeof(float), cudaMemcpyHostToDevice));
            
            // Launch kernel
            dim3 block(16, 16);
            dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
            generate_gradient_kernel<<<grid, block>>>(d_grad_data, d_img_data, width, height);
            
            // Copy results back
            Image grad_img(width, height, 2);
            CHECK_CUDA_ERROR(cudaMemcpy(grad_img.data, d_grad_data,
                           width * height * 2 * sizeof(float), cudaMemcpyDeviceToHost));
            grad_pyramid.octaves[i].push_back(std::move(grad_img));
            
            // Cleanup
            CHECK_CUDA_ERROR(cudaFree(d_img_data));
            CHECK_CUDA_ERROR(cudaFree(d_grad_data));
        }
    }
    
    return grad_pyramid;
}

std::vector<float> find_keypoint_orientations_cuda(Keypoint& kp,
                                                 const ScaleSpacePyramid& grad_pyramid,
                                                 float lambda_ori, float lambda_desc) {
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    
    // Allocate histogram on device
    float *d_hist, *h_hist;
    h_hist = new float[N_BINS]();
    CHECK_CUDA_ERROR(cudaMalloc(&d_hist, N_BINS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_hist, 0, N_BINS * sizeof(float)));
    
    // Copy gradient data to device
    float *d_grad_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_data, img_grad.width * img_grad.height * 2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grad_data, img_grad.data,
                   img_grad.width * img_grad.height * 2 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((img_grad.width + block.x - 1) / block.x,
              (img_grad.height + block.y - 1) / block.y);
    
    compute_orientation_histogram_kernel<<<grid, block>>>(d_hist, d_grad_data,
                                                        img_grad.width, img_grad.height,
                                                        kp.x/pix_dist, kp.y/pix_dist,
                                                        kp.sigma, lambda_ori);
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist, d_hist, N_BINS * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Process histogram to find orientations (on CPU as it's a small operation)
    std::vector<float> orientations;
    float max_val = 0;
    for (int i = 0; i < N_BINS; i++) {
        if (h_hist[i] > max_val) max_val = h_hist[i];
    }
    
    float thresh = 0.8f * max_val;
    for (int i = 0; i < N_BINS; i++) {
        if (h_hist[i] >= thresh) {
            float prev = h_hist[(i-1+N_BINS)%N_BINS];
            float next = h_hist[(i+1)%N_BINS];
            if (prev > h_hist[i] || next > h_hist[i]) continue;
            
            float theta = 2*M_PI*(i+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*h_hist[i]+next);
            orientations.push_back(theta);
        }
    }
    
    // Cleanup
    delete[] h_hist;
    CHECK_CUDA_ERROR(cudaFree(d_hist));
    CHECK_CUDA_ERROR(cudaFree(d_grad_data));
    
    return orientations;
}

void compute_keypoint_descriptor_cuda(Keypoint& kp, float theta,
                                    const ScaleSpacePyramid& grad_pyramid,
                                    float lambda_desc) {
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    
    // 分配设备内存
    float *d_grad_data, *d_descriptor_hist, *h_descriptor_hist;
    h_descriptor_hist = new float[N_HIST*N_HIST*N_ORI]();
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_data, img_grad.width * img_grad.height * 2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_descriptor_hist, N_HIST*N_HIST*N_ORI * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_descriptor_hist, 0, N_HIST*N_HIST*N_ORI * sizeof(float)));
    
    // 复制梯度数据到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_grad_data, img_grad.data,
                               img_grad.width * img_grad.height * 2 * sizeof(float),
                               cudaMemcpyHostToDevice));
    
    // 启动kernel
    dim3 block(16, 16);
    dim3 grid((img_grad.width + block.x - 1) / block.x,
              (img_grad.height + block.y - 1) / block.y);
    
    compute_descriptor_kernel<<<grid, block>>>(d_descriptor_hist, d_grad_data,
                                             img_grad.width, img_grad.height,
                                             kp.x/pix_dist, kp.y/pix_dist,
                                             kp.sigma, theta, lambda_desc);
    
    // 复制结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_descriptor_hist, d_descriptor_hist,
                               N_HIST*N_HIST*N_ORI * sizeof(float),
                               cudaMemcpyDeviceToHost));
    
    // 归一化描述���
    float norm = 0.0f;
    for (int i = 0; i < N_HIST*N_HIST*N_ORI; i++) {
        norm += h_descriptor_hist[i] * h_descriptor_hist[i];
    }
    norm = sqrtf(norm);
    
    // 截断大值并重新归一化
    for (int i = 0; i < N_HIST*N_HIST*N_ORI; i++) {
        h_descriptor_hist[i] = fminf(h_descriptor_hist[i], 0.2f*norm);
    }
    
    float norm2 = 0.0f;
    for (int i = 0; i < N_HIST*N_HIST*N_ORI; i++) {
        norm2 += h_descriptor_hist[i] * h_descriptor_hist[i];
    }
    norm2 = sqrtf(norm2);
    
    // 转换为整数描述子
    for (int i = 0; i < 128; i++) {
        float val = floorf(512.0f * h_descriptor_hist[i] / norm2);
        kp.descriptor[i] = static_cast<uint8_t>(fminf(val, 255.0f));
    }
    
    // 清理内存
    delete[] h_descriptor_hist;
    CHECK_CUDA_ERROR(cudaFree(d_grad_data));
    CHECK_CUDA_ERROR(cudaFree(d_descriptor_hist));
}

std::vector<Keypoint> find_keypoints_cuda(const ScaleSpacePyramid& dog_pyramid,
                                         float contrast_thresh, float edge_thresh) {
    std::vector<Keypoint> keypoints;
    
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        int width = dog_pyramid.octaves[i][0].width;
        int height = dog_pyramid.octaves[i][0].height;
        
        // 分配设备内存
        float *d_dog_data;
        int *d_extrema_map, *h_extrema_map;
        
        h_extrema_map = new int[width * height]();
        CHECK_CUDA_ERROR(cudaMalloc(&d_dog_data,
                                  width * height * dog_pyramid.imgs_per_octave * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_extrema_map, width * height * sizeof(int)));
        
        // 复制DoG数据到设备
        for (int j = 0; j < dog_pyramid.imgs_per_octave; j++) {
            CHECK_CUDA_ERROR(cudaMemcpy(d_dog_data + j*width*height,
                                      dog_pyramid.octaves[i][j].data,
                                      width * height * sizeof(float),
                                      cudaMemcpyHostToDevice));
        }
        
        // 检测极值点
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                 (height + block.y - 1) / block.y);
        
        for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
            CHECK_CUDA_ERROR(cudaMemset(d_extrema_map, 0, width * height * sizeof(int)));
            
            detect_extrema_kernel<<<grid, block>>>(d_extrema_map,
                                                 d_dog_data + j*width*height,
                                                 width, height, contrast_thresh);
            
            // 复制结果回主机
            CHECK_CUDA_ERROR(cudaMemcpy(h_extrema_map, d_extrema_map,
                                      width * height * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            
            // 处理检测到的极值点
            for (int x = 1; x < width-1; x++) {
                for (int y = 1; y < height-1; y++) {
                    if (h_extrema_map[y*width + x]) {
                        Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                        bool kp_is_valid = refine_or_discard_keypoint(kp,
                                                                     dog_pyramid.octaves[i],
                                                                     contrast_thresh,
                                                                     edge_thresh);
                        if (kp_is_valid) {
                            keypoints.push_back(kp);
                        }
                    }
                }
            }
        }
        
        // 清理内存
        delete[] h_extrema_map;
        CHECK_CUDA_ERROR(cudaFree(d_dog_data));
        CHECK_CUDA_ERROR(cudaFree(d_extrema_map));
    }
    
    return keypoints;
}

ScaleSpacePyramid generate_gaussian_pyramid_cuda(const Image& img, float sigma_min,
                                               int num_octaves, int scales_per_octave) {
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur_cuda(base_img, sigma_diff);
    
    int imgs_per_octave = scales_per_octave + 3;
    
    // 计算sigma值
    float k = std::pow(2.0f, 1.0f/scales_per_octave);
    std::vector<float> sigma_vals{base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }
    
    // 创建高斯金字塔
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].reserve(imgs_per_octave);
        pyramid.octaves[i].push_back(std::move(base_img));
        
        for (int j = 1; j < imgs_per_octave; j++) {
            const Image& prev_img = pyramid.octaves[i].back();
            pyramid.octaves[i].push_back(gaussian_blur_cuda(prev_img, sigma_vals[j]));
        }
        
        // 准备下一个octave的基础图像
        if (i < num_octaves - 1) {
            const Image& next_base_img = pyramid.octaves[i][imgs_per_octave-3];
            base_img = next_base_img.resize(next_base_img.width/2,
                                          next_base_img.height/2,
                                          Interpolation::NEAREST);
        }
    }
    
    return pyramid;
}

// Other CUDA implementations...

std::vector<Keypoint> find_keypoints_and_descriptors_cuda(const Image& img,
                                                        float sigma_min,
                                                        int num_octaves,
                                                        int scales_per_octave,
                                                        float contrast_thresh,
                                                        float edge_thresh,
                                                        float lambda_ori,
                                                        float lambda_desc) {
    // Convert image to grayscale if needed
    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    
    // Generate pyramids using CUDA
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid_cuda(input, sigma_min,
                                                                      num_octaves,
                                                                      scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid_cuda(gaussian_pyramid);
    
    // Find keypoints using CUDA
    std::vector<Keypoint> keypoints = find_keypoints_cuda(dog_pyramid,
                                                         contrast_thresh,
                                                         edge_thresh);
    
    // Generate gradient pyramid using CUDA
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid_cuda(gaussian_pyramid);
    
    // Process keypoints
    for (Keypoint& kp : keypoints) {
        std::vector<float> orientations = find_keypoint_orientations_cuda(kp,
                                                                        grad_pyramid,
                                                                        lambda_ori,
                                                                        lambda_desc);
        for (float theta : orientations) {
            Keypoint oriented_kp = kp;
            compute_keypoint_descriptor_cuda(oriented_kp, theta,
                                          grad_pyramid, lambda_desc);
            // Add to final keypoints
        }
    }
    
    return keypoints;
}

// 辅助函数：计算高斯核
__global__ void compute_gaussian_kernel(float* kernel, int size, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int x = idx - size/2;
        kernel[idx] = expf(-(x*x)/(2*sigma*sigma));
    }
}

// 辅助函数：归一化kernel
__global__ void normalize_kernel(float* kernel, int size, float sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        kernel[idx] /= sum;
    }
}

// 辅助函数：水平方向的高斯卷积
__global__ void horizontal_convolution(const float* input, float* output,
                                     const float* kernel, int width, int height,
                                     int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = kernel_size / 2;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int k = -kernel_radius; k <= kernel_radius; k++) {
            int src_x = min(max(x + k, 0), width - 1);
            sum += input[y * width + src_x] * kernel[k + kernel_radius];
        }
        output[y * width + x] = sum;
    }
}

// 辅助函数：垂直方向的高斯卷积
__global__ void vertical_convolution(const float* input, float* output,
                                   const float* kernel, int width, int height,
                                   int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = kernel_size / 2;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int k = -kernel_radius; k <= kernel_radius; k++) {
            int src_y = min(max(y + k, 0), height - 1);
            sum += input[src_y * width + x] * kernel[k + kernel_radius];
        }
        output[y * width + x] = sum;
    }
}

// 辅助函数：图像缩放kernel
__global__ void resize_image_kernel(const float* input, float* output,
                                  int in_width, int in_height,
                                  int out_width, int out_height,
                                  bool use_linear) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        float src_x = x * (float)in_width / out_width;
        float src_y = y * (float)in_height / out_height;
        
        if (use_linear) {  // 双线性插值
            int x0 = floorf(src_x);
            int y0 = floorf(src_y);
            int x1 = min(x0 + 1, in_width - 1);
            int y1 = min(y0 + 1, in_height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            float v00 = input[y0 * in_width + x0];
            float v01 = input[y0 * in_width + x1];
            float v10 = input[y1 * in_width + x0];
            float v11 = input[y1 * in_width + x1];
            
            float value = v00 * (1-dx) * (1-dy) +
                         v01 * dx * (1-dy) +
                         v10 * (1-dx) * dy +
                         v11 * dx * dy;
                         
            output[y * out_width + x] = value;
        } else {  // 最近邻插值
            int src_x_round = roundf(src_x);
            int src_y_round = roundf(src_y);
            src_x_round = min(max(src_x_round, 0), in_width - 1);
            src_y_round = min(max(src_y_round, 0), in_height - 1);
            
            output[y * out_width + x] = input[src_y_round * in_width + src_x_round];
        }
    }
}

// 辅助函数：计算图像梯度
__global__ void compute_gradient_magnitude_orientation(const float* input,
                                                     float* magnitude,
                                                     float* orientation,
                                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width-1 && y > 0 && y < height-1) {
        float dx = (input[y * width + (x+1)] - input[y * width + (x-1)]) * 0.5f;
        float dy = (input[(y+1) * width + x] - input[(y-1) * width + x]) * 0.5f;
        
        magnitude[y * width + x] = sqrtf(dx*dx + dy*dy);
        orientation[y * width + x] = atan2f(dy, dx);
        if (orientation[y * width + x] < 0) {
            orientation[y * width + x] += 2*M_PI;
        }
    }
}

// 辅助函数：计算直方图
__global__ void compute_histogram(const float* magnitude, const float* orientation,
                                float* histogram, int width, int height,
                                float x, float y, float sigma, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        float dx = idx - x;
        float dy = idy - y;
        float dist_sq = dx*dx + dy*dy;
        float weight = expf(-dist_sq/(2*sigma*sigma));
        
        if (weight > 0.001f) {
            float angle = orientation[idy * width + idx];
            float mag = magnitude[idy * width + idx];
            
            int bin = (int)(n_bins * angle / (2*M_PI));
            bin = min(bin, n_bins-1);
            
            atomicAdd(&histogram[bin], mag * weight);
        }
    }
}

// 辅助函数：平滑直方图
__global__ void smooth_histogram(float* histogram, int n_bins) {
    int idx = threadIdx.x;
    if (idx < n_bins) {
        float prev = histogram[(idx-1+n_bins)%n_bins];
        float curr = histogram[idx];
        float next = histogram[(idx+1)%n_bins];
        
        __shared__ float smoothed[N_BINS];
        smoothed[idx] = (prev + curr + next) / 3.0f;
        __syncthreads();
        
        histogram[idx] = smoothed[idx];
    }
}

// 辅助函数：检查CUDA错误
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// 辅助函数：分配和初始化设备内存
template<typename T>
T* cuda_alloc_and_copy(const T* host_data, size_t size) {
    T* device_data;
    cudaError_t err = cudaMalloc(&device_data, size * sizeof(T));
    check_cuda_error(err, "cudaMalloc failed");
    
    if (host_data != nullptr) {
        err = cudaMemcpy(device_data, host_data, size * sizeof(T),
                        cudaMemcpyHostToDevice);
        check_cuda_error(err, "cudaMemcpy H2D failed");
    }
    
    return device_data;
}

// 辅助函数：计算网格维度
dim3 calculate_grid_size(int width, int height, int block_size) {
    dim3 block(block_size, block_size);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    return grid;
}

// 辅助函数：计算Hessian矩阵
__device__ void compute_hessian_3d(const float* dog_data, int width, int height,
                                 int x, int y, int s, float* H) {
    // 计算二阶偏导数
    float dxx = dog_data[s*width*height + y*width + (x+1)] +
                dog_data[s*width*height + y*width + (x-1)] -
                2.0f * dog_data[s*width*height + y*width + x];
    
    float dyy = dog_data[s*width*height + (y+1)*width + x] +
                dog_data[s*width*height + (y-1)*width + x] -
                2.0f * dog_data[s*width*height + y*width + x];
    
    float dss = dog_data[(s+1)*width*height + y*width + x] +
                dog_data[(s-1)*width*height + y*width + x] -
                2.0f * dog_data[s*width*height + y*width + x];
    
    float dxy = (dog_data[s*width*height + (y+1)*width + (x+1)] -
                 dog_data[s*width*height + (y+1)*width + (x-1)] -
                 dog_data[s*width*height + (y-1)*width + (x+1)] +
                 dog_data[s*width*height + (y-1)*width + (x-1)]) / 4.0f;
    
    float dxs = (dog_data[(s+1)*width*height + y*width + (x+1)] -
                 dog_data[(s+1)*width*height + y*width + (x-1)] -
                 dog_data[(s-1)*width*height + y*width + (x+1)] +
                 dog_data[(s-1)*width*height + y*width + (x-1)]) / 4.0f;
    
    float dys = (dog_data[(s+1)*width*height + (y+1)*width + x] -
                 dog_data[(s+1)*width*height + (y-1)*width + x] -
                 dog_data[(s-1)*width*height + (y+1)*width + x] +
                 dog_data[(s-1)*width*height + (y-1)*width + x]) / 4.0f;
    
    // 填充Hessian矩阵
    H[0] = dxx; H[1] = dxy; H[2] = dxs;
    H[3] = dxy; H[4] = dyy; H[5] = dys;
    H[6] = dxs; H[7] = dys; H[8] = dss;
}

// 辅助函数：计算梯度
__device__ void compute_gradient_3d(const float* dog_data, int width, int height,
                                  int x, int y, int s, float* grad) {
    grad[0] = (dog_data[s*width*height + y*width + (x+1)] -
               dog_data[s*width*height + y*width + (x-1)]) / 2.0f;
    
    grad[1] = (dog_data[s*width*height + (y+1)*width + x] -
               dog_data[s*width*height + (y-1)*width + x]) / 2.0f;
    
    grad[2] = (dog_data[(s+1)*width*height + y*width + x] -
               dog_data[(s-1)*width*height + y*width + x]) / 2.0f;
}

// 辅助函数：3x3矩阵求逆
__device__ bool invert_3x3_matrix(const float* H, float* H_inv) {
    float det = H[0]*(H[4]*H[8] - H[5]*H[7]) -
                H[1]*(H[3]*H[8] - H[5]*H[6]) +
                H[2]*(H[3]*H[7] - H[4]*H[6]);
    
    if (fabsf(det) < 1e-10f)
        return false;
    
    float inv_det = 1.0f / det;
    
    H_inv[0] = (H[4]*H[8] - H[5]*H[7]) * inv_det;
    H_inv[1] = (H[2]*H[7] - H[1]*H[8]) * inv_det;
    H_inv[2] = (H[1]*H[5] - H[2]*H[4]) * inv_det;
    H_inv[3] = (H[5]*H[6] - H[3]*H[8]) * inv_det;
    H_inv[4] = (H[0]*H[8] - H[2]*H[6]) * inv_det;
    H_inv[5] = (H[2]*H[3] - H[0]*H[5]) * inv_det;
    H_inv[6] = (H[3]*H[7] - H[4]*H[6]) * inv_det;
    H_inv[7] = (H[1]*H[6] - H[0]*H[7]) * inv_det;
    H_inv[8] = (H[0]*H[4] - H[1]*H[3]) * inv_det;
    
    return true;
}

__global__ void refine_keypoint_kernel(const float* dog_data,
                                     int width, int height, int n_scales,
                                     float contrast_thresh, float edge_thresh,
                                     KeypointData* kp_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= kp_data->n_keypoints)
        return;
    
    int x = kp_data->x[idx];
    int y = kp_data->y[idx];
    int s = kp_data->scale[idx];
    
    // 检查边界
    if (x < 1 || x >= width-1 || y < 1 || y >= height-1 || s < 1 || s >= n_scales-1) {
        kp_data->valid[idx] = false;
        return;
    }
    
    float H[9], H_inv[9], grad[3], offset[3];
    
    // 迭代优化关键点位置
    for (int i = 0; i < MAX_REFINEMENT_ITERS; i++) {
        compute_hessian_3d(dog_data, width, height, x, y, s, H);
        compute_gradient_3d(dog_data, width, height, x, y, s, grad);
        
        if (!invert_3x3_matrix(H, H_inv)) {
            kp_data->valid[idx] = false;
            return;
        }
        
        // 计算偏移量
        offset[0] = -(H_inv[0]*grad[0] + H_inv[1]*grad[1] + H_inv[2]*grad[2]);
        offset[1] = -(H_inv[3]*grad[0] + H_inv[4]*grad[1] + H_inv[5]*grad[2]);
        offset[2] = -(H_inv[6]*grad[0] + H_inv[7]*grad[1] + H_inv[8]*grad[2]);
        
        // 如果偏移量足够小，停止迭代
        if (fabsf(offset[0]) < 0.5f && fabsf(offset[1]) < 0.5f && fabsf(offset[2]) < 0.5f)
            break;
        
        x += roundf(offset[0]);
        y += roundf(offset[1]);
        s += roundf(offset[2]);
        
        // 检查边界
        if (x < 1 || x >= width-1 || y < 1 || y >= height-1 || s < 1 || s >= n_scales-1) {
            kp_data->valid[idx] = false;
            return;
        }
    }
    
    // 计算对比度
    float extremum_value = dog_data[s*width*height + y*width + x] +
                          0.5f * (grad[0]*offset[0] + grad[1]*offset[1] + grad[2]*offset[2]);
    
    if (fabsf(extremum_value) < contrast_thresh) {
        kp_data->valid[idx] = false;
        return;
    }
    
    // 计算主曲率比
    float trace = H[0] + H[4];
    float det = H[0]*H[4] - H[1]*H[3];
    float edge_ratio = trace*trace / det;
    
    if (det <= 0 || edge_ratio >= (edge_thresh + 1)*(edge_thresh + 1)/edge_thresh) {
        kp_data->valid[idx] = false;
        return;
    }
    
    // 保存精细化后的关键点信息
    kp_data->x[idx] = x + offset[0];
    kp_data->y[idx] = y + offset[1];
    kp_data->scale[idx] = s + offset[2];
    kp_data->extremum_value[idx] = extremum_value;
    kp_data->valid[idx] = true;
}

__global__ void find_input_img_coords_kernel(KeypointData* kp_data,
                                           int n_keypoints,
                                           float sigma_min,
                                           int octave) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_keypoints || !kp_data->valid[idx])
        return;
    
    float scale = sigma_min * powf(2.0f, kp_data->scale[idx] / kp_data->n_scales);
    float pix_dist = powf(2.0f, octave);
    
    kp_data->x_img[idx] = kp_data->x[idx] * pix_dist;
    kp_data->y_img[idx] = kp_data->y[idx] * pix_dist;
    kp_data->sigma[idx] = scale * pix_dist;
}

// 结构体用于在GPU和CPU之间传递关键点数据
struct KeypointData {
    float *x, *y, *scale;
    float *x_img, *y_img, *sigma;
    float *extremum_value;
    bool *valid;
    int n_keypoints;
    int n_scales;
};

// 在主机端调用的函数
bool refine_or_discard_keypoint_cuda(Keypoint& kp,
                                   const std::vector<Image>& dog_imgs,
                                   float contrast_thresh,
                                   float edge_thresh) {
    int width = dog_imgs[0].width;
    int height = dog_imgs[0].height;
    int n_scales = dog_imgs.size();
    
    // 分配设备内存
    float *d_dog_data;
    KeypointData *d_kp_data, h_kp_data;
    
    // 分配和初始化关键点数据
    CHECK_CUDA_ERROR(cudaMalloc(&d_dog_data, width * height * n_scales * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_kp_data, sizeof(KeypointData)));
    
    // 复制DoG数据到设备
    for (int i = 0; i < n_scales; i++) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_dog_data + i*width*height,
                                  dog_imgs[i].data,
                                  width * height * sizeof(float),
                                  cudaMemcpyHostToDevice));
    }
    
    // 分配和初始化关键点数据
    h_kp_data.n_keypoints = 1;
    h_kp_data.n_scales = n_scales;
    
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.x, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.y, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.scale, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.x_img, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.y_img, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.sigma, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.extremum_value, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_kp_data.valid, sizeof(bool)));
    
    // 复制关键点数据到设备
    float x = kp.x, y = kp.y;
    int scale = kp.scale;
    CHECK_CUDA_ERROR(cudaMemcpy(h_kp_data.x, &x, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_kp_data.y, &y, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_kp_data.scale, &scale, sizeof(float), cudaMemcpyHostToDevice));
    
    // 复制结构体到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_kp_data, &h_kp_data, sizeof(KeypointData), cudaMemcpyHostToDevice));
    
    // 启动kernel
    int block_size = 256;
    int grid_size = (1 + block_size - 1) / block_size;
    
    refine_keypoint_kernel<<<grid_size, block_size>>>(d_dog_data,
                                                     width, height, n_scales,
                                                     contrast_thresh, edge_thresh,
                                                     d_kp_data);
    
    // 检查结果
    bool valid;
    float refined_x, refined_y, refined_scale, extremum_val;
    CHECK_CUDA_ERROR(cudaMemcpy(&valid, h_kp_data.valid, sizeof(bool), cudaMemcpyDeviceToHost));
    
    if (valid) {
        CHECK_CUDA_ERROR(cudaMemcpy(&refined_x, h_kp_data.x, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&refined_y, h_kp_data.y, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&refined_scale, h_kp_data.scale, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&extremum_val, h_kp_data.extremum_value, sizeof(float), cudaMemcpyDeviceToHost));
        
        kp.x = refined_x;
        kp.y = refined_y;
        kp.scale = refined_scale;
        kp.extremum_val = extremum_val;
    }
    
    // 清理内存
    CHECK_CUDA_ERROR(cudaFree(d_dog_data));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.x));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.y));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.scale));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.x_img));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.y_img));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.sigma));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.extremum_value));
    CHECK_CUDA_ERROR(cudaFree(h_kp_data.valid));
    CHECK_CUDA_ERROR(cudaFree(d_kp_data));
    
    return valid;
}

} // namespace sift