#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <chrono>

#include "sift.hpp"
#include "image.hpp"
#include "image_cu.h"


namespace sift {

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], 0.2f*norm);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    for (int i = 0; i < size; i++) {
        float val = std::floor(512*hist[i]/norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    std::cout << "================= detail time of findkeypoints =================" << std::endl;
    assert(img.channels == 1 || img.channels == 3);
    auto total_start = std::chrono::high_resolution_clock::now();

    // RGB to Grayscale conversion (if needed) ================================
    auto rgb_start = std::chrono::high_resolution_clock::now();
    Image input(img.width, img.height, 1);
    if (img.channels == 1) {
        input = img;
    } else {
        std::cout << "=================IN=================" << std::endl;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int x = 0; x < img.width; x++) {
            for (int y = 0; y < img.height; y++) {
                float red = img.get_pixel(x, y, 0);
                float green = img.get_pixel(x, y, 1);
                float blue = img.get_pixel(x, y, 2);
                input.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
            }
        }
    }
    auto rgb_end = std::chrono::high_resolution_clock::now();
    auto rgb_duration = std::chrono::duration_cast<std::chrono::duration<double>>(rgb_end - rgb_start).count();
    std::cout << "RGB to Grayscale: " << rgb_duration << "s\n";

    // Inlined generate_gaussian_pyramid =====================================
    auto gaussian_start = std::chrono::high_resolution_clock::now();
    
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = input.resize(input.width*2, input.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur_cuda(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals;
    sigma_vals.push_back(base_sigma);
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    std::vector<std::vector<Image>> gaussian_octaves(num_octaves);
    int gaussian_imgs_per_octave = scales_per_octave + 3;
    
    for (int i = 0; i < num_octaves; i++) {
        gaussian_octaves[i].reserve(gaussian_imgs_per_octave);
        gaussian_octaves[i].push_back(std::move(base_img));
        for (int j = 1; j < sigma_vals.size(); j++) {
            const Image& prev_img = gaussian_octaves[i].back();
            gaussian_octaves[i].push_back(gaussian_blur_cuda(prev_img, sigma_vals[j]));
        }
        const Image& next_base_img = gaussian_octaves[i][gaussian_imgs_per_octave-3];
        base_img = next_base_img.resize(next_base_img.width/2, next_base_img.height/2,
                                      Interpolation::NEAREST);
    }

    auto gaussian_end = std::chrono::high_resolution_clock::now();
    auto gaussian_duration = std::chrono::duration_cast<std::chrono::duration<double>>(gaussian_end - gaussian_start).count();
    std::cout << "Generate Gaussian pyramid: " << gaussian_duration << "s\n";

    // Generate DoG pyramid ===================================================
    auto dog_start = std::chrono::high_resolution_clock::now();
    
    // Inline the generate_dog_pyramid_cuda function
    std::vector<std::vector<Image>> dog_octaves(num_octaves);
    int dog_imgs_per_octave = gaussian_imgs_per_octave - 1;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < dog_octaves.size(); i++) {
        dog_octaves[i].reserve(dog_imgs_per_octave);
        for (int j = 1; j < gaussian_octaves[i].size(); j++) {
            Image diff = gaussian_octaves[i][j];
            #pragma omp simd
            for (int pix_idx = 0; pix_idx < diff.size; pix_idx++) {
                diff.data[pix_idx] -= gaussian_octaves[i][j-1].data[pix_idx];
            }
            dog_octaves[i].push_back(diff);
        }
    }

    auto dog_end = std::chrono::high_resolution_clock::now();
    auto dog_duration = std::chrono::duration_cast<std::chrono::duration<double>>(dog_end - dog_start).count();
    std::cout << "Generate DoG pyramid: " << dog_duration << "s\n";

    // Find keypoints =========================================================
    auto keypoints_start = std::chrono::high_resolution_clock::now();
    std::vector<Keypoint> tmp_kps = find_keypoints_cuda(dog_octaves, contrast_thresh, edge_thresh);
    auto keypoints_end = std::chrono::high_resolution_clock::now();
    auto keypoints_duration = std::chrono::duration_cast<std::chrono::duration<double>>(keypoints_end - keypoints_start).count();
    std::cout << "Find keypoints: " << keypoints_duration << "s\n";

    // Generate gradient pyramid ===============================================
    auto grad_start = std::chrono::high_resolution_clock::now();
    
    // Replace the CPU gradient computation with CUDA implementation
    ScaleSpacePyramid gaussian_pyramid = {
        num_octaves,
        gaussian_imgs_per_octave,
        gaussian_octaves
    };
    
    std::vector<std::vector<Image>> grad_octaves = generate_gradient_pyramid_cuda(gaussian_pyramid).octaves;
    
    auto grad_end = std::chrono::high_resolution_clock::now();
    auto grad_duration = std::chrono::duration_cast<std::chrono::duration<double>>(grad_end - grad_start).count();
    std::cout << "Generate gradient pyramid: " << grad_duration << "s\n";
    
    std::vector<Keypoint> kps;
    
    // Process keypoints loop ==================================================
    auto loop_start = std::chrono::high_resolution_clock::now();
    int orientation_count = 0;
    int total_orientations = 0;
    
    #pragma omp parallel
    {
        std::vector<Keypoint> local_kps;
        
        #pragma omp for reduction(+:orientation_count,total_orientations) schedule(dynamic)
        for (size_t kp_idx = 0; kp_idx < tmp_kps.size(); kp_idx++) {
            Keypoint& kp_tmp = tmp_kps[kp_idx];
            float pix_dist = MIN_PIX_DIST * std::pow(2, kp_tmp.octave);
            const Image& img_grad = grad_octaves[kp_tmp.octave][kp_tmp.scale];

            // Border check
            float min_dist_from_border = std::min({kp_tmp.x, kp_tmp.y, 
                                                 pix_dist*img_grad.width-kp_tmp.x,
                                                 pix_dist*img_grad.height-kp_tmp.y});
            if (min_dist_from_border <= std::sqrt(2)*lambda_desc*kp_tmp.sigma) {
                continue;
            }

            // Orientation histogram computation
            float hist[N_BINS] = {};
            float patch_sigma = lambda_ori * kp_tmp.sigma;
            float patch_radius = 3 * patch_sigma;
            int x_start = std::round((kp_tmp.x - patch_radius)/pix_dist);
            int x_end = std::round((kp_tmp.x + patch_radius)/pix_dist);
            int y_start = std::round((kp_tmp.y - patch_radius)/pix_dist);
            int y_end = std::round((kp_tmp.y + patch_radius)/pix_dist);

            // Pre-compute exp table for the patch
            std::vector<std::vector<float>> weights((x_end-x_start+1), 
                                                  std::vector<float>(y_end-y_start+1));
            #pragma omp simd collapse(2)
            for (int x = x_start; x <= x_end; x++) {
                for (int y = y_start; y <= y_end; y++) {
                    weights[x-x_start][y-y_start] = std::exp(
                        -(std::pow(x*pix_dist-kp_tmp.x, 2)+std::pow(y*pix_dist-kp_tmp.y, 2))
                        /(2*patch_sigma*patch_sigma));
                }
            }

            // Accumulate gradients
            for (int x = x_start; x <= x_end; x++) {
                for (int y = y_start; y <= y_end; y++) {
                    float gx = img_grad.get_pixel(x, y, 0);
                    float gy = img_grad.get_pixel(x, y, 1);
                    float grad_norm = std::sqrt(gx*gx + gy*gy);
                    float theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
                    int bin = (int)std::round(N_BINS/(2*M_PI)*theta) % N_BINS;
                    hist[bin] += weights[x-x_start][y-y_start] * grad_norm;
                }
            }

            smooth_histogram(hist);

            // Find orientations
            float ori_thresh = 0.8;
            float ori_max = 0;
            std::vector<float> orientations;
            
            #pragma omp simd reduction(max:ori_max)
            for (int j = 0; j < N_BINS; j++) {
                ori_max = std::max(ori_max, hist[j]);
            }
            
            for (int j = 0; j < N_BINS; j++) {
                if (hist[j] >= ori_thresh * ori_max) {
                    float prev = hist[(j-1+N_BINS)%N_BINS];
                    float next = hist[(j+1)%N_BINS];
                    if (prev > hist[j] || next > hist[j])
                        continue;
                    float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
                    orientations.push_back(theta);
                }
            }

            orientation_count++;
            total_orientations += orientations.size();
            
            // Process each orientation
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                float histograms[N_HIST][N_HIST][N_ORI] = {0};

                float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
                int x_start = std::round((kp.x-half_size) / pix_dist);
                int x_end = std::round((kp.x+half_size) / pix_dist);
                int y_start = std::round((kp.y-half_size) / pix_dist);
                int y_end = std::round((kp.y+half_size) / pix_dist);

                float cos_t = std::cos(theta), sin_t = std::sin(theta);
                float patch_sigma = lambda_desc * kp.sigma;

                // Pre-compute transformation matrices
                std::vector<float> x_transformed((x_end-x_start+1) * (y_end-y_start+1));
                std::vector<float> y_transformed((x_end-x_start+1) * (y_end-y_start+1));
                std::vector<float> weights((x_end-x_start+1) * (y_end-y_start+1));
                
                #pragma omp simd collapse(2)
                for (int m = x_start; m <= x_end; m++) {
                    for (int n = y_start; n <= y_end; n++) {
                        int idx = (m-x_start)*(y_end-y_start+1) + (n-y_start);
                        x_transformed[idx] = ((m*pix_dist - kp.x)*cos_t
                                            +(n*pix_dist - kp.y)*sin_t) / kp.sigma;
                        y_transformed[idx] = (-(m*pix_dist - kp.x)*sin_t
                                            +(n*pix_dist - kp.y)*cos_t) / kp.sigma;
                        weights[idx] = std::exp(-(std::pow(m*pix_dist-kp.x, 2)+std::pow(n*pix_dist-kp.y, 2))
                                                /(2*patch_sigma*patch_sigma));
                    }
                }

                // Compute descriptor
                for (int m = x_start; m <= x_end; m++) {
                    for (int n = y_start; n <= y_end; n++) {
                        int idx = (m-x_start)*(y_end-y_start+1) + (n-y_start);
                        float x = x_transformed[idx];
                        float y = y_transformed[idx];

                        if (std::max(std::abs(x), std::abs(y)) > lambda_desc*(N_HIST+1.)/N_HIST)
                            continue;

                        float gx = img_grad.get_pixel(m, n, 0);
                        float gy = img_grad.get_pixel(m, n, 1);
                        float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
                        float grad_norm = std::sqrt(gx*gx + gy*gy);
                        float contribution = weights[idx] * grad_norm;

                        // Update histograms
                        update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
                    }
                }

                // Convert histograms to descriptor vector
                hists_to_vec(histograms, kp.descriptor);
                local_kps.push_back(kp);
            }
        }

        // Merge local results
        #pragma omp critical
        {
            kps.insert(kps.end(), local_kps.begin(), local_kps.end());
        }
    }
    auto loop_end = std::chrono::high_resolution_clock::now();
    auto loop_duration = std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - loop_start).count();
    
    std::cout << "Keypoint processing loop:\n";
    std::cout << "  Total keypoints processed: " << orientation_count << "\n";
    std::cout << "  Total orientations found: " << total_orientations << "\n";
    std::cout << "  Average time per keypoint: " << loop_duration / orientation_count << "s\n";
    std::cout << "  Total loop time: " << loop_duration << "s\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count();
    std::cout << "Total execution time: " << total_duration << "s\n";
    std::cout << "================================================================" << std::endl;

    return kps;
}


float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    float dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt(dist);
}

std::vector<std::pair<int, int>> find_keypoint_matches(std::vector<Keypoint>& a,
                                                       std::vector<Keypoint>& b,
                                                       float thresh_relative,
                                                       float thresh_absolute)
{
    assert(a.size() >= 2 && b.size() >= 2);

    std::vector<std::pair<int, int>> matches;

    #pragma omp parallel for
    for (int i = 0; i < a.size(); i++) {
        // find two nearest neighbours in b for current keypoint from a
        int nn1_idx = -1;
        float nn1_dist = 100000000, nn2_dist = 100000000;
        for (int j = 0; j < b.size(); j++) {
            float dist = euclidean_dist(a[i].descriptor, b[j].descriptor);
            if (dist < nn1_dist) {
                nn2_dist = nn1_dist;
                nn1_dist = dist;
                nn1_idx = j;
            } else if (nn1_dist <= dist && dist < nn2_dist) {
                nn2_dist = dist;
            }
        }
        if (nn1_dist < thresh_relative*nn2_dist && nn1_dist < thresh_absolute) {
            #pragma omp critical
            matches.push_back({i, nn1_idx});
        }
    }
    return matches;
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}

Image draw_matches(const Image& a, const Image& b, std::vector<Keypoint>& kps_a,
                   std::vector<Keypoint>& kps_b, std::vector<std::pair<int, int>> matches)
{
    Image res(a.width+b.width, std::max(a.height, b.height), 3);

    for (int i = 0; i < a.width; i++) {
        for (int j = 0; j < a.height; j++) {
            res.set_pixel(i, j, 0, a.get_pixel(i, j, 0));
            res.set_pixel(i, j, 1, a.get_pixel(i, j, a.channels == 3 ? 1 : 0));
            res.set_pixel(i, j, 2, a.get_pixel(i, j, a.channels == 3 ? 2 : 0));
        }
    }
    for (int i = 0; i < b.width; i++) {
        for (int j = 0; j < b.height; j++) {
            res.set_pixel(a.width+i, j, 0, b.get_pixel(i, j, 0));
            res.set_pixel(a.width+i, j, 1, b.get_pixel(i, j, b.channels == 3 ? 1 : 0));
            res.set_pixel(a.width+i, j, 2, b.get_pixel(i, j, b.channels == 3 ? 2 : 0));
        }
    }

    for (auto& m : matches) {
        Keypoint& kp_a = kps_a[m.first];
        Keypoint& kp_b = kps_b[m.second];
        draw_line(res, kp_a.x, kp_a.y, a.width+kp_b.x, kp_b.y);
    }
    return res;
}
} // namespace sift
