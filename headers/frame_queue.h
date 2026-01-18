#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

class FrameQueue {
public:
    explicit FrameQueue(size_t max_size = 10);
    ~FrameQueue();

    /// Push a frame. Returns false if queue is closed or max_size==0.
    bool push(const cv::Mat& frame);

    /// Pop a frame. Blocks until data available or queue closed.
    /// Returns false if queue is empty AND closed.
    bool pop(cv::Mat& frame);

    bool empty() const;
    size_t size() const;

    /// Close queue: no further pushes allowed, unblocks all waiting pops.
    void close();

    /// Whether queue is closed.
    bool isClosed() const;

private:
    mutable std::mutex mtx;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;

    std::queue<cv::Mat> q;
    size_t max_size;
    bool closed;
};
