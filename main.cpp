#include "image_loader.hpp"
#include "conv.hpp"

#include <argparse.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("image_loader");
    program.add_argument("--input_dir").help("Directory containing images").required().default_value("../images");
    program.add_argument("--output_dir").help("Output directory for filtered images").required().default_value("../output");

    // parse args
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        std::cerr << program;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // load images from directory
    std::string inputDir = program.get<std::string>("--input_dir");
    std::vector<cv::Mat> images = image_loader(inputDir);

    // create filter 3x3
    cv::Mat filter = (cv::Mat_<float>(3,3) <<
         -1, 0,  1,
        -2,  0, 2,
         -1, 0,  1
    );

    // create output directory if not there
    std::string outputDir = program.get<std::string>("--output_dir");
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
        std::cout << "Created " << outputDir << std::endl;
    }

    // conv it with each image and output to folder
    for (size_t idx = 0; idx < images.size(); ++idx) {
        cv::Mat& img = images[idx];
        cv::Mat result = conv(img, filter);
    
        std::string outName = outputDir + "/conv_" + std::to_string(idx) + ".png";
        cv::imwrite(outName, result);
    
        std::cout << "Saved: " << outName << std::endl;
    }
    std::cout << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Program ran for: " << duration.count() << " seconds." << std::endl;
    return 0;
}