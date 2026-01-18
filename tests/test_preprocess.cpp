#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../headers/preprocess.h" 

static void assertMsg(bool cond, const std::string &msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        throw std::runtime_error(msg);
    }
}

struct CaseResult { std::string name; bool ok; std::string msg; };

CaseResult run_case(const std::string &name, int orig_w, int orig_h, int input_w = 640, int input_h = 640) {
    try {
        cv::Mat img(orig_h, orig_w, CV_8UC3);
        cv::randu(img, 0, 255);
        Preprocessor prep(input_w, input_h);
        cv::Mat blob = prep.process(img);
        assertMsg(!blob.empty(), name + ": The output blob should not be empty.");
        assertMsg(blob.dims == 4, name + ": Expected blob to have 4 dimensions (NCHW), but got " + std::to_string(blob.dims));
        assertMsg(blob.size[0] == 1, name + ": Expected batch size of 1, but got " + std::to_string(blob.size[0]));
        assertMsg(blob.size[1] == 3, name + ": Expected 3 color channels, but got " + std::to_string(blob.size[1]));
        assertMsg(blob.size[2] == input_h && blob.size[3] == input_w,
                  name + ": Expected blob dimensions to be HxW (" + std::to_string(input_h) + "x" + std::to_string(input_w) +
                  "), but got " + std::to_string(blob.size[2]) + "x" + std::to_string(blob.size[3]));
        assertMsg(blob.depth() == CV_32F, name + ": Expected blob data type to be CV_32F (float).");

        // --- Removed Tests ---
        // All tests related to 'getScaleAndPadding' and coordinate transformations
        // have been removed, as that logic is no longer part of the preprocessor.

        return {name, true, "OK"};
    } catch (const std::exception &ex) {
        return {name, false, ex.what()};
    } catch (...) {
        return {name, false, "An unknown error occurred."};
    }
}

int main() {
    std::vector<std::tuple<std::string,int,int>> cases = {
        {"standard_640x480", 640, 480},
        {"exact_640x640", 640, 640},
        {"smaller_320x240", 320, 240},
        {"wide_1000x200", 1000, 200},
        {"tall_200x1000", 200, 1000},
        {"tiny_1x1", 1, 1},
        {"odd_dims_123x321", 123, 321},
    };

    int total = 0, passed = 0;
    for (auto &c : cases) {
        total++;
        auto res = run_case(std::get<0>(c), std::get<1>(c), std::get<2>(c));
        if (res.ok) {
            std::cout << "[PASS] " << res.name << " : " << res.msg << std::endl;
            passed++;
        } else {
        }
    }

    std::cout << "\n=== Test Summary: " << passed << " / " << total << " passed ===" << std::endl;
    return (passed == total) ? 0 : 1;
}