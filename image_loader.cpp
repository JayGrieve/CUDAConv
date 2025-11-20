    #include <iostream>
    #include <vector>
    #include <opencv2/opencv.hpp>
    #include <omp.h> 

    int main() {
        std::vector<std::string> imagePaths = {"images/test_image_1.png", "images/test_image_2.png", "images/test_image_3.png"};
        std::vector<cv::Mat> loadedImages(imagePaths.size()); // Pre-allocate for direct assignment

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
        return 0;
    }