#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat pad(const cv::Mat& image, const cv::Mat& filter, const std::string padType = "zero");
cv::Mat conv(const cv::Mat& image, const cv::Mat& filter);