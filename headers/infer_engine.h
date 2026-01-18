#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

class InferEngine {
public:
    InferEngine(); // default constructor
    explicit InferEngine(const std::string& model_path); // convenience constructor
    ~InferEngine(); // destructor

    bool loadModel(const std::string& model_path);
    cv::Mat infer(const cv::Mat& input_blob);

    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    std::string model_path_;
    int input_width_ = 640;
    int input_height_ = 640;
};
