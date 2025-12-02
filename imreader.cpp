#include "image_loader.hpp"
#include <cuda_runtime.h>
#include <omp.h> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <tuple>
#include <algorithm>

namespace fs = std::filesystem;

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
    //
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


int main() {
    std::string inputDir = "images";
    std::string outputDir = "output_images";

    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
        std::cout << "Created " << outputDir << std::endl;
    }

    // Load images using image_loader
    std::vector<cv::Mat> images = image_loader(inputDir);

    // Resize all images to 512x512
    std::cout << "\nResizing images to 512x512..." << std::endl;
    for (size_t i = 0; i < images.size(); ++i) {
        if (!images[i].empty()) {
            cv::resize(images[i], images[i], cv::Size(512, 512));
            std::cout << "Resized image " << i << " to 512x512" << std::endl;
        }
    }

    //Sobel
    /*
    float filter[9] = {-1.0f, 0.0f,  1.0f, 
                       -2.0f,  0.0f, 2.0f, 
                       -1.0f, 0.0f, 1.0f};
    */
    float filter[9] = {0.111111f, 0.111111f,  0.111111f, 
                    0.111111f,  0.111111f,0.111111f, 
                    0.111111f, 0.111111f, 0.111111f};
    /*
    float filter[9] = {-1.0, 0.0, -1.0,
                        0.0, 5.0, 0.0,
                        -1.0, 0.0, -1.0};
    */
    for (size_t i = 0; i < images.size(); ++i) {


        cv::Size sz = images[i].size();
        int imageWidth = sz.width;
        int imageHeight = sz.height;

        unsigned char* output_host = new unsigned char[imageWidth * imageHeight*3];

        unsigned char *image_device_ptr,*output_device_ptr;
        float *filter_device_ptr;        
        cudaMalloc(&image_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3); //w * h * CV_8UC3 * 3
        cudaMalloc(&output_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3); //w * h * CV_8UC3 * 3
        cudaMalloc(&filter_device_ptr,3*3*sizeof(float)); //3*3 array

        cudaMemcpy(image_device_ptr,images[i].data,imageWidth*imageHeight*sizeof(unsigned char)*3,cudaMemcpyHostToDevice);
        cudaMemcpy(filter_device_ptr,filter,9*sizeof(float),cudaMemcpyHostToDevice);

        dim3 numBlocks(32,32);
        dim3 threadsPerBlock(16,16);

            
        convolve<<<numBlocks,threadsPerBlock>>>(imageWidth,imageHeight,filter_device_ptr,image_device_ptr,output_device_ptr);
        cudaDeviceSynchronize();
        //Copys back result from device
        cudaMemcpy(output_host,output_device_ptr,imageWidth*imageHeight*sizeof(unsigned char)*3,cudaMemcpyDeviceToHost);

        // Convert output_host back to cv::Mat
        cv::Mat outputImage(imageHeight, imageWidth, CV_8UC3, output_host);

        // Save to output_images directory
        std::string outName = outputDir + "/conv_cuda_" + std::to_string(i) + ".png";
        cv::imwrite(outName, outputImage);
        std::cout << "Saved: " << outName << std::endl;

        // Clean up memory
        delete[] output_host;
        cudaFree(image_device_ptr);
        cudaFree(output_device_ptr);
        cudaFree(filter_device_ptr);
    }

    std::cout << "\nAll images processed and saved to " << outputDir << std::endl;

    return 0;
}