#include "../headers/frame_queue.h"
#include <iostream>
#include <opencv2/opencv.hpp>

FrameQueue::FrameQueue(size_t max_size)
    : max_size(max_size), closed(false) {}

FrameQueue::~FrameQueue() {
    close();
}

bool FrameQueue::push(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mtx);

    if (closed || max_size == 0) {
        return false;
    }

    //wait if the queue is full
    while (q.size() >= max_size) {
        cv_push.wait(lock);  //wait until space is available in the queue
    }

    q.push(frame);

    //notify one of the waiting threads to pop
    cv_pop.notify_one();
    return true;

}

bool FrameQueue::pop(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mtx);

    while (q.empty() && !closed) {
        cv_pop.wait(lock);  //wait until a frame is pushed or the queue is closed
    }

    if (q.empty() && closed) {
        return false;
    }

    frame = q.front();
    q.pop();  

    cv_push.notify_one();
    return true;
}

bool FrameQueue::empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
}

size_t FrameQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
}

void FrameQueue::close() {
    std::unique_lock<std::mutex> lock(mtx);
    closed = true;  

    //notify all waiting threads to unblock them in case they're waiting
    cv_push.notify_all();
    cv_pop.notify_all();
}

bool FrameQueue::isClosed() const {
    std::lock_guard<std::mutex> lock(mtx);
    return closed;
}
