// sift_cu.h
#ifndef SIFT_CU_H
#define SIFT_CU_H

#include "sift.hpp"

namespace sift {

// CUDA versions of the functions
ScaleSpacePyramid generate_gaussian_pyramid_cuda(const Image& img, float sigma_min,
                                               int num_octaves, int scales_per_octave);

ScaleSpacePyramid generate_dog_pyramid_cuda(const ScaleSpacePyramid& img_pyramid);

std::vector<Keypoint> find_keypoints_cuda(const ScaleSpacePyramid& dog_pyramid,
                                        float contrast_thresh, float edge_thresh);

ScaleSpacePyramid generate_gradient_pyramid_cuda(const ScaleSpacePyramid& pyramid);

std::vector<float> find_keypoint_orientations_cuda(Keypoint& kp,
                                                 const ScaleSpacePyramid& grad_pyramid,
                                                 float lambda_ori, float lambda_desc);

void compute_keypoint_descriptor_cuda(Keypoint& kp, float theta,
                                   const ScaleSpacePyramid& grad_pyramid,
                                   float lambda_desc);

std::vector<Keypoint> find_keypoints_and_descriptors_cuda(const Image& img,
                                                        float sigma_min,
                                                        int num_octaves,
                                                        int scales_per_octave,
                                                        float contrast_thresh,
                                                        float edge_thresh,
                                                        float lambda_ori,
                                                        float lambda_desc);

bool refine_or_discard_keypoint_cuda(Keypoint& kp,
                                   const std::vector<Image>& dog_imgs,
                                   float contrast_thresh,
                                   float edge_thresh);

void check_cuda_error(cudaError_t err, const char* msg);

template<typename T>
T* cuda_alloc_and_copy(const T* host_data, size_t size);

dim3 calculate_grid_size(int width, int height, int block_size);

Image gaussian_blur_cuda(const Image& img, float sigma);

} // namespace sift

#endif