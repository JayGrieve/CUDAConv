#include "conv.hpp"

#include <iostream>

cv::Mat pad(const cv::Mat& image, const cv::Mat& filter, const std::string padType) {
    if (padType != "zero") {
        std::cout << "[WARNING]: padType '" << padType << "' supplied but only 'zero' is implemented." << std::endl;
        return image; // dont pad
    }

    int pad_h = filter.rows / 2;
    int pad_w = filter.cols / 2;

    cv::Mat padded;
    cv::copyMakeBorder(
        image,
        padded,
        pad_h, pad_h, pad_w, pad_w,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    return padded;
}

// basic convolution. always zero pads. stride 1
cv::Mat conv(const cv::Mat& image, const cv::Mat& filter) {
    CV_Assert(image.type() == CV_8UC3); // 8 bits (1 byte char) with 3 channels
    CV_Assert(filter.type() == CV_32F); // 32 bit float (4 byte float)

    int fh = filter.rows;
    int fw = filter.cols;
    int rh = fh / 2;  // radius of filter height
    int rw = fw / 2;  // radius of filter width

    cv::Mat padded = pad(image, filter);
    cv::Mat output(image.rows, image.cols, CV_8UC3); // output will be 8 bits (1 byte char) with 3 channels

    for (int r = 0; r < image.rows; ++r) {
        for (int c = 0; c < image.cols; ++c) {
            float accB = 0.0f;
            float accG = 0.0f;
            float accR = 0.0f;

            for (int r_offset = -rh; r_offset <= rh; ++r_offset) {
                for (int c_offset = -rw; c_offset <= rw; ++c_offset) {
                    int pr = r + r_offset + rh;   // padded row index
                    int pc = c + c_offset + rw;   // padded col index

                    const cv::Vec3b& pix = padded.at<cv::Vec3b>(pr, pc);
                    float scalar = filter.at<float>(r_offset + rh, c_offset + rw);

                    accB += pix[0] * scalar;
                    accG += pix[1] * scalar;
                    accR += pix[2] * scalar;
                }
            }

            // BGR because opencv imread does BGR
            output.at<cv::Vec3b>(r, c)[0] = cv::saturate_cast<uchar>(accB); // cast float into [0, 255] 
            output.at<cv::Vec3b>(r, c)[1] = cv::saturate_cast<uchar>(accG);
            output.at<cv::Vec3b>(r, c)[2] = cv::saturate_cast<uchar>(accR);
        }
    }

    return output;
}