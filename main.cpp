#include "image_loader.hpp"

#include <argparse.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("image_loader");
    program.add_argument("--dir").help("Directory containing images").required();

    // parse args
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        std::cerr << program;
        return 1;
    }

    // load images from directory
    std::string directory = program.get<std::string>("--dir");
    std::vector<cv::Mat> images = image_loader(directory);

    //
    
    return 0;
}