#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iomanip>
// Include all the corrected and verified headers
#include "../headers/infer_engine.h"
#include "../headers/preprocess.h"
#include "../headers/nms.h"
#include "../headers/frame_queue.h"

// The producer function reads frames from a video source and pushes them into a queue.
void producer(FrameQueue& fq, const std::string& video_path, std::atomic<bool>& running) {
    cv::VideoCapture cap;
    if (video_path.empty()) {
        std::cerr << "Error: empty video path.\n";
        fq.close();
        return;
    }

    if (!cap.open(video_path)) {
        std::cerr << "Error: failed to open video: " << video_path << "\n";
        fq.close();
        return;
    }

    while (running.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;
        }
        if (!fq.push(frame)) {
            break;
        }
    }

    fq.close();
    cap.release();
    std::cerr << "Exiting, queue closed.\n";
}

static inline cv::Mat rows_detections(const cv::Mat& preds) {
    auto in_range = [](int x){ return x >= 10 && x <= 512; };

    if (preds.empty()) return preds;

    const int r = preds.rows;
    const int c = preds.cols;

    if (in_range(c) && r > c) return preds;
    if (in_range(r) && c > r) return preds.t();
    if (r <= 128 && c > r) return preds.t();
    return preds;
}

// The consumer function takes frames from the queue and performs the full inference pipeline.
void consumer(FrameQueue& fq, InferEngine& engine, std::atomic<bool>& running,
              float conf_threshold, float nms_threshold)
{
    Preprocessor pre(engine.getInputWidth(), engine.getInputHeight());

    cv::VideoWriter writer;
    const std::string out_path = "output.mp4";
    const int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
    const double fps_fallback = 25.0;

    while (running.load(std::memory_order_relaxed) || !fq.empty()) {
        cv::Mat frame;
        if (!fq.pop(frame)) break;
        if (frame.empty()) continue;

        if (!writer.isOpened()) {
            if (!writer.open(out_path, fourcc, fps_fallback, frame.size(), true)) {
                std::cerr << "ERROR: could not open writer for " << out_path << "\n";
            } else {
                std::cerr << "Writing annotated video to " << out_path << "\n";
            }
        }

        cv::Mat blob = pre.process(frame);
        if (blob.empty()) {
            std::cerr << "Preprocess returned empty blob so writing raw frame.\n";
            if (writer.isOpened()) writer.write(frame);
            continue;
        }

        cv::Mat preds;
        try {
            preds = engine.infer(blob);
            //check the predictions size and type
            std::cout << "Predictions size: " << preds.size() << ", type: " << preds.type() << std::endl;
            if (preds.rows < preds.cols) {
                preds = preds.t();
            }
        } catch (const std::exception& ex) {
            std::cerr << "[Consumer] Inference error: " << ex.what() << " ; writing raw frame.\n";
            if (writer.isOpened()) writer.write(frame);
            continue;
        }

        if (preds.empty()) {
            if (writer.isOpened()) writer.write(frame);
            continue;
        }

        cv::Mat shaped = rows_detections(preds);

        if (shaped.type() != CV_32F || shaped.cols < 6) {
            std::cerr << "warning: unexpected predictions shape (" << shaped.rows << "x" << shaped.cols << "); writing raw frame.\n";
            if (writer.isOpened()) writer.write(frame);
            continue;
        }

        std::vector<Detection> dets = postprocess(shaped, frame.size(), conf_threshold, nms_threshold);

        //check how many detections are found
        std::cout << "Detections found: " << dets.size() << std::endl;
        if (dets.empty()) {
            std::cerr << "No detections found. Writing raw frame.\n";
            if (writer.isOpened()) writer.write(frame);
            continue;
        }

        //bounding boxes and labels on the frame
        for (const auto& d : dets) {
            std::cout << "Drawing bounding box: " << d.box << ", Confidence: " << d.conf << std::endl;
            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);

            std::ostringstream oss;
            oss << "class " << d.cls << " " << std::fixed << std::setprecision(2) << d.conf;

            int baseline = 0;
            cv::Size label_sz = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point tl((int)d.box.x, std::max(0, (int)d.box.y - label_sz.height - 4));
            cv::Rect bg(tl.x, tl.y, label_sz.width + 6, label_sz.height + 6);
            bg &= cv::Rect(0, 0, frame.cols, frame.rows);

            cv::rectangle(frame, bg, cv::Scalar(0,255,0), cv::FILLED);
            cv::putText(frame, oss.str(),cv::Point(bg.x + 3, bg.y + label_sz.height + 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);
        }

        if (writer.isOpened()) writer.write(frame);
    }

    if (writer.isOpened()) {
        writer.release();
        std::cerr << "Finished writing " << out_path << "\n";
    }

    std::cerr << "Exiting.\n";
}

