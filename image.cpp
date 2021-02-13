#include <cmath>
#include <iostream>
#include <cassert>
#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    this->data = new float[size]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = y*width*channels + x*channels + c;
                int dst_idx = c*height*width + y*width + x;
                this->data[dst_idx] = img_data[src_idx] / 255.;
            }
        }
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
{
    width = w;
    height = h;
    channels = c;
    size = width * height * channels;
    data = new float[w*h*c]();
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    delete[] data;
    //std::cout << "copy assignment\n";
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;
    data = new float[other.size];
    for (int i = 0; i < other.size; i++)
        data[i] = other.data[i];
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int c = 0; c < channels; c++) {
                int dst_idx = y*width*channels + x*channels + c;
                int src_idx = c*height*width + y*width + x;
                out_data[dst_idx] = std::roundf(data[src_idx] * 255.);
            }
        }
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        //return 0.0f;
        x = 0;
    if (x >= width)
        //return 0.f;
        x = width - 1;
    if (y < 0)
        //return 0.f;
        y = 0;
    if (y >= height)
        //return 0.0f;
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        float val = data[i];
        val = (val > 1.0) ? 1.0 : val;
        val = (val < 0.0) ? 0.0 : val;
        data[i] = val;
    }
}

Image Image::resize(int new_w, int new_h, Interpolation interp) const
{
    Image resized(new_w, new_h, this->channels);
    float value;
    for (int x = 0; x < new_w; x++) {
        for (int y = 0; y < new_h; y++) {
            for (int c = 0; c < resized.channels; c++) {
                float old_x = map_coordinate(this->width, new_w, x);
                float old_y = map_coordinate(this->height, new_h, y);
                if (interp == Interpolation::BILINEAR)
                    value = bilinear_interpolate(*this, old_x, old_y, c);
                else if (interp == Interpolation::NEAREST)
                    value = nn_interpolate(*this, old_x, old_y, c);
                resized.set_pixel(x, y, c, value);
            }
        }
    }
    return resized;
}

inline float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

inline float bilinear_interpolate(const Image& img, float x, float y, int c) //const image
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), x_ceil = std::ceil(x);
    float y_floor = std::floor(y), y_ceil = std::ceil(y);
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float red, green, blue;
            red = img.get_pixel(x, y, 0);
            green = img.get_pixel(x, y, 1);
            blue = img.get_pixel(x, y, 2);
            gray.set_pixel(x, y, 0, 0.299*red + 0.587*green + 0.114*blue);
        }
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            float gray_val = img.get_pixel(x, y, 0);
            rgb.set_pixel(x, y, 0, gray_val);
            rgb.set_pixel(x, y, 1, gray_val);
            rgb.set_pixel(x, y, 2, gray_val);
        }
    }
    return rgb;
}

void draw_point(Image& img, int x, int y)
{
    for (int i = x-3; i <= x+3; i++) {
        for (int j = y-3; j <= y+3; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > 1) continue;
            //std::cout << i << " " << j << "\n";
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        int tmp = x1;
        x1 = x2;
        x2 = tmp;
        tmp = y1;
        y1 = y2;
        y2 = tmp;
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        img.set_pixel(x, y, 0, 0.f);
        img.set_pixel(x, y, 1, 1.f);
        img.set_pixel(x, y, 2, 0.f);
    }
}

Image convolve(const Image& img, const Image& filter, bool preserve)
{
    assert(filter.channels == img.channels || filter.channels == 1);
    int new_c = 1;
    if (preserve) //preserve number of channels after filtering
        new_c = img.channels;
    Image filtered(img.width, img.height, new_c);
    for (int x = 0; x < img.width; x++) {
        for (int y = 0; y < img.height; y++) {
            for (int c = 0; c < img.channels; c++) {
                float val = 0;
                for (int i = 0; i < filter.width; i++) {
                    for (int j = 0; j < filter.height; j++) {
                        int dx = -filter.width/2 + i;
                        int dy = -filter.height/2 + j;
                        int filter_c = (filter.channels == 1) ? 0 : c;
                        val += img.get_pixel(x+dx, y+dy, c) * filter.get_pixel(i, j, filter_c);
                    }
                }
                if (preserve) {
                    filtered.set_pixel(x, y, c, val);
                } else {
                    float prev_channel_sum = filtered.get_pixel(x, y, 0);
                    filtered.set_pixel(x, y, 0, prev_channel_sum+val);
                }
            }
        }
    }
    return filtered;
}
