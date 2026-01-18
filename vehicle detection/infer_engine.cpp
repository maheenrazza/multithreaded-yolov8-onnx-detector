#include "infer_engine.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <cpu_provider_factory.h>

InferEngine::InferEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "InferEngine") {}

InferEngine::InferEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "InferEngine") {
    if (!loadModel(model_path)) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
}

InferEngine::~InferEngine() = default;

bool InferEngine::loadModel(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model file not found: " << model_path << std::endl;
        return false;
    }

    Ort::SessionOptions session_options;

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
        std::cout << "Model loaded successfully: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }

    return true;
}

cv::Mat InferEngine::infer(const cv::Mat& input_blob) {
    if (input_blob.empty()) {
        std::cerr << "Error: Empty blob received for inference." << std::endl;
        return cv::Mat();
    }

    //ensuring input is a 4D blob [1, 3, 640, 640] 
    cv::Mat blob_4d;
    if (input_blob.dims == 2 && input_blob.cols == 1 &&
        input_blob.total() == 1 * 3 * 640 * 640) {
        int shape[] = {1, 3, 640, 640};
        blob_4d = input_blob.reshape(1, 4, shape);
    } else if (input_blob.dims == 4) {
        blob_4d = input_blob;
    } else {
        std::cerr << "Error: Expected 4D blob [1, 3, 640, 640], got "
                  << input_blob.dims << "D tensor" << std::endl;
        return cv::Mat();
    }

    if (!blob_4d.isContinuous()) {
        blob_4d = blob_4d.clone(); 
    }

    //creating ONNX input tensor from the blob
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        blob_4d.ptr<float>(),
        blob_4d.total(),
        input_shape.data(),
        input_shape.size()
    );

    //get ONNX input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name_ptr.get() };
    const char* output_names[] = { output_name_ptr.get() };


    //run inference 
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );


    Ort::Value& output_tensor = output_tensors.front();

    auto output_shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = output_shape_info.GetShape();

    // YOLOv8 output shape is [batch_size, num_features, num_predictions]
    if (output_shape.size() != 3 || output_shape[0] != 1) {
        std::cerr << "Error: Unexpected output tensor shape. Expected [1, C, N]." << std::endl;
        return cv::Mat();
    }

    const size_t num_features = output_shape[1]; 
    const size_t num_predictions = output_shape[2];

    float* output_data = output_tensor.GetTensorMutableData<float>();
    cv::Mat output_as_mat(1, num_features * num_predictions, CV_32F, output_data);
 
    cv::Mat predictions = output_as_mat.reshape(1, num_features).clone();
    
    return predictions;
}