#include "../headers/preprocess.h"
#include <iostream>

Preprocessor::Preprocessor(int input_width, int input_height)
    : input_width_(input_width), input_height_(input_height) {}

cv::Mat Preprocessor::process(const cv::Mat& frame) {
    std::cout << "Received frame size: [" << frame.cols << " x " << frame.rows << "]" << std::endl;

    float scale = std::min(
        static_cast<float>(input_width_) / frame.cols,
        static_cast<float>(input_height_) / frame.rows
    );
    
    int new_width = static_cast<int>(frame.cols * scale);
    int new_height = static_cast<int>(frame.rows * scale);
    
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));
    
    //letterboxed image (640x640) with gray padding
    cv::Mat letterboxed(input_height_, input_width_, frame.type(), cv::Scalar(114, 114, 114));
    int x_offset = (input_width_ - new_width) / 2;
    int y_offset = (input_height_ - new_height) / 2;
    resized.copyTo(letterboxed(cv::Rect(x_offset, y_offset, new_width, new_height)));
    
    std::cout << "Letterboxed image: " << letterboxed.cols << "x" << letterboxed.rows 
              << " (content: " << new_width << "x" << new_height << " at offset " 
              << x_offset << "," << y_offset << ")" << std::endl;
    

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);  
    std::cout << "RGB image size: [" << rgb.cols << " x " << rgb.rows << "]" << std::endl;

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    cv::Mat blob_1d(1, 3 * input_height_ * input_width_, CV_32F);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            blob_1d.ptr<float>(0) + c * input_height_ * input_width_,
            channels[c].ptr<float>(0),
            input_height_ * input_width_ * sizeof(float)
        );
    }

    std::vector<int> blob_shape = {1, 3, input_height_, input_width_};
    cv::Mat blob = blob_1d.reshape(1, blob_shape);

    
    std::cout << "NCHW blob size: [1 x " << 3 << " x " << input_height_ << " x " << input_width_ << "]" << std::endl;
    
    return blob;
}