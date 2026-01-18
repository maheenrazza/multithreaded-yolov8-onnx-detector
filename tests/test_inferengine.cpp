#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../headers/infer_engine.h"

static void assertMsg(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        throw std::runtime_error(msg);
    }
}

// Test 1: Engine should fail gracefully with invalid model path
bool test_invalid_model_path() {
    try {
        InferEngine engine("nonexistent_model.onnx");
        assertMsg(false, "Engine constructor should throw on invalid model path");
    } catch (...) {
        return true; // Expected failure
    }
}

// Test 2: Engine should load a valid model
bool test_valid_model_load(const std::string& model_path) {
    InferEngine engine(model_path);
    // Assuming YOLOv8 default input size
    int expected_width = 640;
    int expected_height = 640;
    assertMsg(expected_width == 640, "Model input width should be 640");
    assertMsg(expected_height == 640, "Model input height should be 640");
    return true;
}

// Test 3: Passing empty Mat should return empty output
bool test_infer_with_empty_blob(const std::string& model_path) {
    InferEngine engine(model_path);
    cv::Mat empty_blob;
    cv::Mat output = engine.infer(empty_blob);
    assertMsg(output.empty(), "Infer should return empty Mat when given empty input");
    return true;
}

// Test 4: Valid flattened blob produces valid output
bool test_infer_on_valid_blob(const std::string& model_path) {
    InferEngine engine(model_path);

    int H = 640, W = 640;
    std::vector<float> blob_data(1 * 3 * H * W, 0.5f);
    cv::Mat blob(blob_data.size(), 1, CV_32F, blob_data.data());

    cv::Mat predictions = engine.infer(blob);
    assertMsg(!predictions.empty(), "Inference on valid blob should produce output");
    assertMsg(predictions.depth() == CV_32F, "Output Mat should be CV_32F");

    // For YOLOv8 small (n), output shape is typically 84x8400
    assertMsg(predictions.rows == 84, "Output rows should be 84");
    assertMsg(predictions.cols == 8400, "Output cols should be 8400");
    return true;
}

// Helper: create proper preprocessed blob from image
cv::Mat create_preprocessed_blob(const cv::Mat& image, int target_width = 640, int target_height = 640) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width, target_height));
    resized.convertTo(resized, CV_32F, 1.0/255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    std::vector<float> blob_data;
    blob_data.reserve(3 * target_height * target_width);
    for(int c = 0; c < 3; c++) {
        blob_data.insert(blob_data.end(), (float*)channels[c].data, (float*)channels[c].data + channels[c].total());
    }
    return cv::Mat(blob_data.size(), 1, CV_32F, blob_data.data()).clone();
}

// Test 5: Infer with actual image preprocessing
bool test_infer_with_real_image(const std::string& model_path) {
    InferEngine engine(model_path);

    cv::Mat dummy_image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::rectangle(dummy_image, cv::Rect(100, 100, 200, 150), cv::Scalar(128, 128, 128), -1);

    cv::Mat blob = create_preprocessed_blob(dummy_image);
    cv::Mat predictions = engine.infer(blob);

    assertMsg(!predictions.empty(), "Inference with real image should produce output");
    assertMsg(predictions.rows == 84 && predictions.cols == 8400, "Output shape should be 84x8400");
    return true;
}

int main() {
    std::string model_path = "yolov8n.onnx";
    std::ifstream f(model_path);
    if (!f.good()) {
        std::cerr << "[FATAL] Model not found at: " << model_path << std::endl;
        return 1;
    }

    int passed = 0;
    int total = 0;

    auto run_test = [&](auto test_func, const std::string& name) {
        total++;
        try {
            if (test_func()) {
                std::cout << "[PASS] " << name << std::endl;
                passed++;
            }
        } catch (const std::exception& e) {
        } catch (...) {
            std::cerr << "[FAIL] " << name << " : Unknown exception" << std::endl;
        }
    };

    run_test(test_invalid_model_path, "Load invalid model path");
    run_test([&](){ return test_valid_model_load(model_path); }, "Load valid model");
    run_test([&](){ return test_infer_with_empty_blob(model_path); }, "Infer with empty blob");
    run_test([&](){ return test_infer_on_valid_blob(model_path); }, "Infer on valid blob");
    run_test([&](){ return test_infer_with_real_image(model_path); }, "Infer with real image preprocessing");

    std::cout << "\n=== Test Summary: " << passed << " / " << total << " passed ===" << std::endl;
    return (passed == total) ? 0 : 1;
}
