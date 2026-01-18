#include "nms.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

// --- Helper function: Compute IoU ---
float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float intersection_area = (a & b).area();
    if (intersection_area <= 0.0f) return 0.0f;
    float union_area = a.area() + b.area() - intersection_area;
    return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
}

//Helper function: Apply Non-Maximum Suppression 
void applyNMS(std::vector<Detection>& detections, float iou_threshold) {
    if (detections.empty()) return;

    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.conf > b.conf;
              });

    std::vector<bool> keep(detections.size(), true);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j]) continue;
            // Only suppress boxes of the same class
            if (detections[i].cls == detections[j].cls) {
                if (computeIoU(detections[i].box, detections[j].box) > iou_threshold) {
                    keep[j] = false;
                }
            }
        }
    }

    std::vector<Detection> final_detections;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (keep[i]) {
            final_detections.push_back(detections[i]);
        }
    }
    detections = final_detections;
}

std::vector<Detection> postprocess(
    const cv::Mat& predictions_in,
    cv::Size original_image_size,
    float conf_threshold,
    float iou_threshold
)
{
    std::vector<Detection> detections;


    if (predictions_in.empty() || predictions_in.type() != CV_32F) {
        return detections;
    }

    cv::Mat predictions;
    
    if (predictions_in.cols == 84) {
        predictions = predictions_in;  
    } else if (predictions_in.rows == 84) {
        predictions = predictions_in.t();  
    } else {
        std::cerr << "Error: Unexpected predictions shape [" << predictions_in.rows 
                  << ", " << predictions_in.cols << "]. Expected 84 columns or 84 rows." << std::endl;
        return detections;
    }
    
    // std::cout << "[DEBUG] Processed Shape: [" << predictions.rows 
    //           << ", " << predictions.cols << "]" << std::endl;

    const float model_width = 640.0f;
    const float model_height = 640.0f;

    float scale = std::min(model_width / original_image_size.width, 
                          model_height / original_image_size.height);
    float x_offset = (model_width - original_image_size.width * scale) / 2.0f;
    float y_offset = (model_height - original_image_size.height * scale) / 2.0f;
    
    // std::cout << "[DEBUG] Conf Threshold: " << conf_threshold << std::endl;

    for (int i = 0; i < predictions.rows; ++i) {
        const float* row = predictions.ptr<float>(i);
        
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        int best_class = -1;
        float max_prob = 0.0f;
        
        for (int class_idx = 0; class_idx < 80; ++class_idx) {
            float prob = row[4 + class_idx];
            if (prob > max_prob) {
                max_prob = prob;
                best_class = class_idx;
            }
        }

        if (max_prob < conf_threshold) {
            continue;
        }

        //convert from model coordinates to original image coordinates
        float x1 = (cx - w / 2.0f - x_offset) / scale;
        float y1 = (cy - h / 2.0f - y_offset) / scale;
        float x2 = (cx + w / 2.0f - x_offset) / scale;
        float y2 = (cy + h / 2.0f - y_offset) / scale;

        //clamp to image boundaries
        x1 = std::max(0.0f, std::min(x1, (float)original_image_size.width));
        y1 = std::max(0.0f, std::min(y1, (float)original_image_size.height));
        x2 = std::max(0.0f, std::min(x2, (float)original_image_size.width));
        y2 = std::max(0.0f, std::min(y2, (float)original_image_size.height));

        if (x2 <= x1 || y2 <= y1) {
            if (y2 <= y1) {
                y2 = y1 + 1.0f; //1 pixel of height
            }
            
            if (x2 <= x1) {
                x2 = x1 + 1.0f; //1 pixel of width
            }
            
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
        }

        Detection det;
        det.box = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        det.conf = max_prob;
        det.cls = best_class;

        detections.push_back(det);
    }

    applyNMS(detections, iou_threshold);

    return detections;
}