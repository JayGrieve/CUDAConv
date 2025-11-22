#include "image_loader.hpp"

#include <omp.h> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<cv::Mat> image_loader(const std::string& directory) {
    std::cout << "Loading from directory: " << directory << "\n";

    // Find number of images in directory
    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(directory)) {
        imagePaths.push_back(entry.path().string());
    }
    std::cout << "Found " << imagePaths.size() << " images" << std::endl;

    // loading images to vector
    std::vector<cv::Mat> loadedImages(imagePaths.size());
    #pragma omp parallel for
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        loadedImages[i] = cv::imread(imagePaths[i], cv::IMREAD_COLOR);
        if (!loadedImages[i].empty()) {
            std::cout << "Loaded (OpenMP): " << imagePaths[i] << std::endl;
        } else {
            std::cerr << "Failed to load (OpenMP): " << imagePaths[i] << std::endl;
        }
    }

    std::cout << "All images loaded with OpenMP." << std::endl;
    return loadedImages;
}