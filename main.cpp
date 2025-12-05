#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <tuple>

namespace fs = std::filesystem;

// Remember, this in in build/ so probably want ../
const std::string INPUT_DIR = "../frames";
const std::string OUTPUT_DIR = "../output_frames";

//Assuming 3 channel image
__device__ float3 getPixelRGB(unsigned char *image,int pixelIDx,int pixelIDy,int imageWidth, int imageHeight){
    
    float3 rgb;
    //boundary conditions
    if(pixelIDx < 0 || pixelIDx > imageWidth || pixelIDy < 0 || pixelIDy > imageHeight){
        rgb.x = 0.0f;
        rgb.y = 0.0f;
        rgb.z = 0.0f;
        return rgb;

    }
    //B
    rgb.x = (image[(pixelIDy*imageWidth + pixelIDx) * 3]);
    //G
    rgb.y = (image[(pixelIDy*imageWidth + pixelIDx) * 3 + 1]);
    //R
    rgb.z = (image[(pixelIDy*imageWidth + pixelIDx) * 3 + 2]);

    return rgb;
}


__global__ void convolve(int imageWidth, int imageHeight, float *filter, unsigned char *image, unsigned char *output){

    int pixel_index = (blockDim.x*blockDim.y*gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x) + (blockDim.x*threadIdx.y) + threadIdx.x;
    
    int pixelIDx = pixel_index % imageWidth;
    int pixelIDy = pixel_index / imageWidth;

    //Center
    float cumsum_b = 0;
    float cumsum_g = 0;
    float cumsum_r = 0;
    float3 output_temp;
    int i,j,kernelIdx;
    float value;
    // Assuming conv filter is [topleft, topcenter, topright, midleft, midcenter, midright,bottomleft,bottomcenter,bottomright]
    for(j = -1; j<=1; ++j){
        for(i = -1; i<=1; ++i){
            kernelIdx = 3*(j+1) + (i+1);
            value = filter[kernelIdx];
            output_temp = getPixelRGB(image,pixelIDx + i,pixelIDy+j,imageWidth,imageHeight);
            cumsum_b += output_temp.x * value;
            cumsum_g += output_temp.y * value;
            cumsum_r += output_temp.z * value;
        }
    }

    //B
    output[(pixelIDy*imageWidth + pixelIDx) * 3] = static_cast<unsigned char>(fminf(fmaxf(cumsum_b, 0.0f), 255.0f));
    //G
    output[(pixelIDy*imageWidth + pixelIDx) * 3 + 1] = static_cast<unsigned char>(fminf(fmaxf(cumsum_g, 0.0f), 255.0f));
    //R
    output[(pixelIDy*imageWidth + pixelIDx) * 3 + 2] = static_cast<unsigned char>(fminf(fmaxf(cumsum_r, 0.0f), 255.0f));
}

void distributeImageIndexes(unsigned int buf[], std::string& dir, int size) {
    // Find number of images in directory
    unsigned int imagesInPath = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        imagesInPath += 1;
    }

    int divisor = imagesInPath / size;
    int remainder = imagesInPath % size;
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        int n_y_rank = divisor + (i < remainder ? 1 : 0);
        int starting_index = i * divisor + (i < remainder ? i : remainder);

        // create (numImages, startingImage) pairs per rank
        buf[2*i] = n_y_rank;
        buf[2*i+1] = starting_index;
    }
}

std::tuple<double, double> loadAndResizeImages(std::vector<cv::Mat>& images, std::string& dir, unsigned int numImages, unsigned int startingImage) {
    // Load Images
    auto loadStart = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < numImages; ++i) {
        std::string filepath = dir + "/" + std::to_string(startingImage + i) + ".png";
        images[i] = cv::imread(filepath, cv::IMREAD_COLOR);
        if (!images[i].empty()) {
            // std::cout << "Loaded (OpenMP): " << filepath << std::endl;
        } else {
            std::cerr << "Failed to load (OpenMP): " << filepath << std::endl;
        }
    }

    auto loadEnd = std::chrono::high_resolution_clock::now();
    double loadTime = std::chrono::duration<double>(loadEnd - loadStart).count();

    // Resize all images to 512x512
    // std::cout << "\nResizing images to 512x512..." << std::endl;
    auto resizeStart = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); ++i) {
        if (!images[i].empty()) {
            cv::resize(images[i], images[i], cv::Size(512, 512));
        }
    }

    auto resizeEnd = std::chrono::high_resolution_clock::now();
    double resizeTime = std::chrono::duration<double>(resizeEnd - resizeStart).count();

    // return timings
    return std::tuple(loadTime, resizeTime);
}

int main(int argc, char *argv[]) {
    // Initalize OpenMPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    std::string inputDir = INPUT_DIR;
    std::string outputDir = OUTPUT_DIR;

    if (rank == 0) {
        std::cout << "Input directory: " << inputDir << std::endl;
        std::cout << "Output directory: " << outputDir << std::endl;
    }

    // Create output directory if it doesn't exist
    if (rank == 0 && !fs::exists(outputDir)) {
        fs::create_directory(outputDir);
        std::cout << "Created " << outputDir << std::endl;
    }
    
    // Wait for rank 0 to create directory
    MPI_Barrier(MPI_COMM_WORLD);

    // distribute workload across cpus
    unsigned int buf[size*2]; // num cpus and (numImages, startingImage) pairs
    if (rank == 0) {
        printf("Running with %d processes.\n", size);
        distributeImageIndexes(buf, inputDir, size);
    }

    // Broadcast buf to all ranks
    MPI_Bcast(buf,size*2,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    unsigned int numImages = buf[2*rank];
    unsigned int startingImage = buf[2*rank+1];

    // Timing variables
    double loadTime = 0.0, resizeTime = 0.0;
    double totalGpuTime = 0.0, totalMemcpyH2D = 0.0, totalKernelTime = 0.0, totalMemcpyD2H = 0.0;

    // Load Images
    std::vector<cv::Mat> images(numImages);
    auto [lt, rt] = loadAndResizeImages(images, inputDir, numImages, startingImage);
    loadTime = lt;
    resizeTime = rt;

    // Create filiter
    float filter[9] = {-1.0f, 0.0f,  1.0f, 
                       -2.0f,  0.0f, 2.0f, 
                       -1.0f, 0.0f, 1.0f};

    // float filter[9] = {0.111111f, 0.111111f,  0.111111f, 
    //                 0.111111f,  0.111111f,0.111111f, 
    //                 0.111111f, 0.111111f, 0.111111f};
    /*
    float filter[9] = {-1.0, 0.0, -1.0,
                        0.0, 5.0, 0.0,
                        -1.0, 0.0, -1.0};
    */
    
    // Convolve Every Image
    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Size sz = images[i].size();
        int imageWidth = sz.width;
        int imageHeight = sz.height;

        // CUDA timing events
        cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
        cudaEventCreate(&start_h2d);
        cudaEventCreate(&stop_h2d);
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_d2h);

        unsigned char* output_host = new unsigned char[imageWidth * imageHeight*3];

        unsigned char *image_device_ptr,*output_device_ptr;
        float *filter_device_ptr;        
        cudaMalloc(&image_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3); //w * h * CV_8UC3 * 3
        cudaMalloc(&output_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3); //w * h * CV_8UC3 * 3
        cudaMalloc(&filter_device_ptr,3*3*sizeof(float)); //3*3 array

        // Time Host to Device copy
        cudaEventRecord(start_h2d);
        cudaMemcpy(image_device_ptr,images[i].data,imageWidth*imageHeight*sizeof(unsigned char)*3,cudaMemcpyHostToDevice);
        cudaMemcpy(filter_device_ptr,filter,9*sizeof(float),cudaMemcpyHostToDevice);
        cudaEventRecord(stop_h2d);

        dim3 numBlocks(32,32);
        dim3 threadsPerBlock(16,16);

        // Time kernel execution
        cudaEventRecord(start_kernel);
        convolve<<<numBlocks,threadsPerBlock>>>(imageWidth,imageHeight,filter_device_ptr,image_device_ptr,output_device_ptr);
        cudaEventRecord(stop_kernel);

        cudaDeviceSynchronize();
        
        // Time Device to Host copy
        cudaEventRecord(start_d2h);
        cudaMemcpy(output_host,output_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3,cudaMemcpyDeviceToHost);
        cudaEventRecord(stop_d2h);
        cudaDeviceSynchronize();

        // Calculate elapsed times
        float milliseconds_h2d = 0, milliseconds_kernel = 0, milliseconds_d2h = 0;
        cudaEventElapsedTime(&milliseconds_h2d, start_h2d, stop_h2d);
        cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel);
        cudaEventElapsedTime(&milliseconds_d2h, start_d2h, stop_d2h);

        #pragma omp critical
        {
            totalMemcpyH2D += milliseconds_h2d / 1000.0;
            totalKernelTime += milliseconds_kernel / 1000.0;
            totalMemcpyD2H += milliseconds_d2h / 1000.0;
            totalGpuTime += (milliseconds_h2d + milliseconds_kernel + milliseconds_d2h) / 1000.0;
        }

        // Convert output_host back to cv::Mat
        cv::Mat outputImage(imageHeight, imageWidth, CV_8UC3, output_host);

        // Save to output_images directory
        std::string outName = outputDir + "/conv_cuda_" + std::to_string(startingImage + i) + ".png";
        cv::imwrite(outName, outputImage);
        // std::cout << "Saved: " << outName << std::endl;

        // Clean up memory
        delete[] output_host;
        cudaFree(image_device_ptr);
        cudaFree(output_device_ptr);
        cudaFree(filter_device_ptr);
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(stop_h2d);
        cudaEventDestroy(start_kernel);
        cudaEventDestroy(stop_kernel);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);
    }

    // std::cout << "\nAll images processed and saved to " << outputDir << std::endl;

    // Print timing statistics
    std::cout << "\n========== TIMING STATISTICS (Rank " << rank << ") ==========" << std::endl;
    std::cout << "Number of images processed: " << numImages << std::endl;
    std::cout << "\nCPU Times:" << std::endl;
    std::cout << "  Image Loading:   " << loadTime << " s (" << (loadTime / numImages) << " s/image)" << std::endl;
    std::cout << "  Image Resizing:  " << resizeTime << " s (" << (resizeTime / numImages) << " s/image)" << std::endl;
    std::cout << "\nGPU Times:" << std::endl;
    std::cout << "  Host to Device:  " << totalMemcpyH2D << " s (" << (totalMemcpyH2D / numImages) << " s/image)" << std::endl;
    std::cout << "  Kernel Execution:" << totalKernelTime << " s (" << (totalKernelTime / numImages) << " s/image)" << std::endl;
    std::cout << "  Device to Host:  " << totalMemcpyD2H << " s (" << (totalMemcpyD2H / numImages) << " s/image)" << std::endl;
    std::cout << "  Total GPU Time:  " << totalGpuTime << " s (" << (totalGpuTime / numImages) << " s/image)" << std::endl;
    std::cout << "\nTotal Time:        " << (loadTime + resizeTime + totalGpuTime) << " s" << std::endl;
    std::cout << "========================================\n" << std::endl;

    MPI_Finalize();
    return 0;
}