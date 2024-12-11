#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include "image.hpp"
#include "sift.hpp"

int main(int argc, char *argv[])
{
    // Validate mode argument
    if (argc < 2) {
        std::cerr << "Error: Mode argument is required ('test' or 'plot')" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    if (mode != "test" && mode != "plot") {
        std::cerr << "Invalid mode: " << mode << "\nValid modes are: 'test', 'plot'" << std::endl;
        return 1;
    }

    // Handle optional image_num argument
    int requested_num_images = -1;
    if (argc > 2) {
        // Validate if argument is a number
        std::string num_str = argv[2];
        if (!std::all_of(num_str.begin(), num_str.end(), ::isdigit)) {
            std::cerr << "Error: Second argument must be a number" << std::endl;
            return 1;
        }
        requested_num_images = std::stoi(num_str);
    }

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    const char* input_path = "data/list.txt";
    std::cout << "Opening list file: " << input_path << std::endl;
    std::ifstream file(input_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << input_path << "\n";
        return 1;
    }
    std::cout << "Successfully opened list file" << std::endl;

    // Get the directory path of the list file
    std::string list_dir = "data/";

    std::cout << "Reading base image path..." << std::endl;
    std::string base_image_path;
    std::getline(file, base_image_path);
    base_image_path = list_dir + base_image_path;
    std::cout << "Loading base image: " << base_image_path << std::endl;
    
    // 开始总计时
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 开始计算基准图像处理时间
    auto base_start = std::chrono::high_resolution_clock::now();
    
    Image base_image(base_image_path);
    std::cout << "Base image loaded, dimensions: " << base_image.width << "x" << base_image.height << std::endl;
    
    base_image = base_image.channels == 1 ? base_image : rgb_to_grayscale(base_image);
    std::cout << "Converting base image to grayscale if needed" << std::endl;
    
    std::cout << "Finding SIFT features in base image..." << std::endl;
    std::vector<sift::Keypoint> base_kps = sift::find_keypoints_and_descriptors_omp(base_image);
    std::cout << "Found " << base_kps.size() << " keypoints in base image" << std::endl;

    // 计算并输出基准图像处理时间
    auto base_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> base_duration = base_end - base_start;
    std::cout << "Base image processing time: " << base_duration.count() << "s" << std::endl;

    std::cout << "Reading number of images to process..." << std::endl;
    int num_images;
    file >> num_images;
    file.ignore();
    
    // Use requested number if valid
    if (requested_num_images > 0) {
        num_images = std::min(requested_num_images, num_images);
    }
    std::cout << "Will process " << num_images << " images" << std::endl;

    std::vector<std::pair<int, std::string>> matches_count;
    for (int i = 0; i < num_images; ++i) {
        auto img_start = std::chrono::high_resolution_clock::now();
        
        std::string image_path;
        std::getline(file, image_path);
        image_path = list_dir + image_path;
        
        std::cout << "\nProcessing image " << (i + 1) << " of " << num_images << ": " << image_path << std::endl;

        std::cout << "Loading image..." << std::endl;
        Image img(image_path);
        std::cout << "Image loaded, dimensions: " << img.width << "x" << img.height << std::endl;
        
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
        std::cout << "Converting to grayscale if needed" << std::endl;
        
        // Timing SIFT feature extraction
        auto sift_start = std::chrono::high_resolution_clock::now();
        std::cout << "Finding SIFT features..." << std::endl;
        std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors_omp(img);
        auto sift_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> sift_duration = sift_end - sift_start;
        std::cout << "Found " << kps.size() << " keypoints in " << sift_duration.count() << "s" << std::endl;

        // Timing feature matching
        auto match_start = std::chrono::high_resolution_clock::now();
        std::cout << "Matching features with base image..." << std::endl;
        std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(base_kps, kps);
        auto match_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> match_duration = match_end - match_start;
        std::cout << "Found " << matches.size() << " matches in " << match_duration.count() << "s" << std::endl;
        
        matches_count.push_back({matches.size(), image_path});

        // 计算并输出当前图像处理时间
        auto img_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> img_duration = img_end - img_start;
        std::cout << "Total image processing time: " << img_duration.count() << "s" << std::endl;
    }

    std::cout << "\nSorting results by number of matches..." << std::endl;
    std::sort(matches_count.begin(), matches_count.end(), std::greater<>());

    // Write results to file
    std::string output_filename = "res/matching_list.txt";
    std::ofstream output_file(output_filename);
    if (!output_file.is_open()) {
        std::cerr << "Failed to create output file: " << output_filename << std::endl;
        return 1;
    }

    if (mode == "test") {
        std::cout << "\nWriting results to " << output_filename << "..." << std::endl;
        for (const auto& [count, path] : matches_count) {
            output_file << path << " " << count << "\n";
        }
        output_file.close();
        std::cout << "Writing matching_list finished." << std::endl;

        int output_limit = std::min(3, static_cast<int>(matches_count.size()));
        for (int i = 0; i < output_limit; ++i) {
            const auto& [count, path] = matches_count[i];
            std::cout << "Save Image: " << path << " - Matches: " << count << std::endl;

            // Reload the image to draw matches
            Image img(path);
            img = img.channels == 1 ? img : rgb_to_grayscale(img);

            // Find keypoints and descriptors again
            std::vector<sift::Keypoint> kps = sift::find_keypoints_and_descriptors(img);

            // Draw matches
            std::vector<std::pair<int, int>> matches = sift::find_keypoint_matches(base_kps, kps);
            Image result = sift::draw_matches(base_image, img, base_kps, kps, matches);

            // Save result image
            std::string result_filename = "./res/result_" + std::to_string(i + 1) + ".jpg";
            result.save(result_filename);
            std::cout << "Result image saved as " << result_filename << std::endl;

            // Save coordinates of matching keypoints for the most matching image
            if (i == 0) {
                std::ofstream coord_file("./res/matching_coordinates.txt");
                if (coord_file.is_open()) {
                    for (const auto& match : matches) {
                        const auto& base_kp = base_kps[match.first];
                        const auto& img_kp = kps[match.second];
                        coord_file << "Base Image: (" << base_kp.x << ", " << base_kp.y << ") "
                                   << "Matched Image: (" << img_kp.x << ", " << img_kp.y << ")\n";
                    }
                    coord_file.close();
                    std::cout << "Matching coordinates saved to ./res/matching_coordinates.txt" << std::endl;
                } else {
                    std::cerr << "Failed to open file for writing matching coordinates." << std::endl;
                }
            }
        }
    } 

    // 计算并输出总处理时间
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;
    std::cout << "\nTotal processing time: " << total_duration.count() << "s" << std::endl;

        // Add handling for "plot" mode
    if (mode == "plot") {
        // Write the number of processed images and total processing time to time.txt
        std::string time_filename = "res/time.txt";
        std::ofstream time_file(time_filename, std::ios::app); // Open in append mode
        if (time_file.is_open()) {
            time_file << num_images << " " << total_duration.count() << "\n"; // Write the data
            time_file.close();
            std::cout << "Processing time and image count saved to " << time_filename << std::endl;
        } else {
            std::cerr << "Failed to open file for writing processing time." << std::endl;
        }
    }

    return 0;
}