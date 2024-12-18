#ifndef SIFT_H
#define SIFT_H

#include <vector>
#include <array>
#include <cstdint>

#include "image.hpp"

namespace sift {

struct ScaleSpacePyramid {
    int num_octaves;
    int imgs_per_octave;
    std::vector<std::vector<Image>> octaves; 
};

struct Keypoint {
    // discrete coordinates
    int i;
    int j;
    int octave;
    int scale; //index of gaussian image inside the octave

    // continuous coordinates (interpolated)
    float x;
    float y;
    float sigma;
    float extremum_val; //value of interpolated DoG extremum
    
    std::array<uint8_t, 128> descriptor;
};

//*******************************************
// SIFT algorithm parameters, used by default
//*******************************************

// digital scale space configuration and keypoint detection
const int MAX_REFINEMENT_ITERS = 5;
const float SIGMA_MIN = 0.8;
const float MIN_PIX_DIST = 0.5;
const float SIGMA_IN = 0.5;
const int N_OCT = 8;
const int N_SPO = 3;
const float C_DOG = 0.015;
const float C_EDGE = 10;

// computation of the SIFT descriptor
const int N_BINS = 36;
const float LAMBDA_ORI = 1.5;
const int N_HIST = 4;
const int N_ORI = 8;
const float LAMBDA_DESC = 6;

// feature matching
const float THRESH_ABSOLUTE = 350;
const float THRESH_RELATIVE = 0.7;

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                               float contrast_thresh, float edge_thresh);

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min=SIGMA_MIN,
                                                     int num_octaves=N_OCT, 
                                                     int scales_per_octave=N_SPO, 
                                                     float contrast_thresh=C_DOG,
                                                     float edge_thresh=C_EDGE,
                                                     float lambda_ori=LAMBDA_ORI,
                                                     float lambda_desc=LAMBDA_DESC);

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
                                                       std::vector<Keypoint>& b,
                                                       float thresh_relative=THRESH_RELATIVE,
                                                       float thresh_absolute=THRESH_ABSOLUTE);

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps);

Image draw_matches(const Image& a, const Image& b, std::vector<Keypoint>& kps_a,
                   std::vector<Keypoint>& kps_b, std::vector<std::pair<int, int>> matches);

ScaleSpacePyramid generate_gradient_pyramid_cuda(const ScaleSpacePyramid& pyramid);

ScaleSpacePyramid generate_dog_pyramid_cuda(const ScaleSpacePyramid& gauss_pyr);

std::vector<Keypoint> find_keypoints_cuda(const std::vector<std::vector<Image>>& dog_octaves,
                                         float contrast_thresh, float edge_thresh);

} // namespace sift
#endif
