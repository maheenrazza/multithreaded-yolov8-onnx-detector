#include <iostream>
#include <vector>
#include "../headers/nms.h"

int main() {
    using namespace std;

    try {
        // --- Test 1: Single dummy detection ---
        {
            cv::Mat preds(1, 84, CV_32F, cv::Scalar(0));
            preds.at<float>(0,0) = 5.0f; preds.at<float>(0,1) = 5.0f;
            preds.at<float>(0,2) = 10.0f; preds.at<float>(0,3) = 10.0f;
            preds.at<float>(0,4) = 0.9f;
            auto res = postprocess(preds, {640,480}, 0.5f, 0.5f);
            cout << "[TEST1] " << (res.size() == 1 ? "PASS" : "FAIL") << "\n";
        }

        // --- Test 2: Empty input ---
        {
            cv::Mat empty_preds;
            auto res = postprocess(empty_preds, {640,480}, 0.5f, 0.5f);
            cout << "[TEST2] " << (res.empty() ? "PASS" : "FAIL") << "\n";
        }

        // --- Test 3: Two overlapping detections, same class ---
        {
            cv::Mat preds(2, 84, CV_32F, cv::Scalar(0));
            preds.at<float>(0,0) = 10; preds.at<float>(0,1) = 10; preds.at<float>(0,2) = 20; preds.at<float>(0,3) = 20; preds.at<float>(0,4) = 0.9f;
            preds.at<float>(1,0) = 12; preds.at<float>(1,1) = 12; preds.at<float>(1,2) = 20; preds.at<float>(1,3) = 20; preds.at<float>(1,4) = 0.8f;
            auto res = postprocess(preds, {640,480}, 0.5f, 0.5f);
            cout << "[TEST3] " << (res.size() == 1 ? "PASS" : "FAIL") << "\n";
        }

        // --- Test 4: Two overlapping detections, different classes ---
        {
            cv::Mat preds(2, 84, CV_32F, cv::Scalar(0));
            preds.at<float>(0,0) = 10; preds.at<float>(0,1) = 10; preds.at<float>(0,2) = 20; preds.at<float>(0,3) = 20; preds.at<float>(0,4) = 0.9f;
            preds.at<float>(1,0) = 12; preds.at<float>(1,1) = 12; preds.at<float>(1,2) = 20; preds.at<float>(1,3) = 20; preds.at<float>(1,5) = 0.85f;
            auto res = postprocess(preds, {640,480}, 0.5f, 0.5f);
            cout << "[TEST4] " << (res.size() == 2 ? "PASS" : "FAIL") << "\n";
        }

    } catch (...) {
        cerr << "Error: test failed\n";
        return 1;
    }

    return 0;
}
