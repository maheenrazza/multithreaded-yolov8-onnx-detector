#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect2f box;
    float conf;
    int cls;
};

std::vector<Detection> postprocess(
    const cv::Mat& predictions,
    cv::Size original_image_size,
    float conf_threshold = 0.25f,
    float iou_threshold = 0.45f
);