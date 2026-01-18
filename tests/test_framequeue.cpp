#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "../headers/frame_queue.h"

using namespace std;

#define LOG(...) do { cerr << __VA_ARGS__ << endl; } while(0)
#define RUN_TEST(fn) \
    do { \
        cout << "Running " << #fn << " ... "; \
        bool ok = fn(); \
        if (ok) cout << "[PASS]\n"; else cout << "[FAIL]\n"; \
        total++; if (ok) passed++; \
    } while(0)

vector<cv::Mat> generate_dummy_frames(int n, int w = 640, int h = 480) {
    vector<cv::Mat> frames;
    for (int i = 0; i < n; ++i) {
        cv::Mat frame = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
        cv::putText(frame, "Frame " + to_string(i), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
        frames.push_back(frame);
    }
    return frames;
}

// ---------------- Tests ----------------

bool test_push_pop_basic() {
    FrameQueue fq(10);
    auto frames = generate_dummy_frames(5);

    for (auto &f : frames) {
        if (!fq.push(f)) { LOG("push failed unexpectedly"); return false; }
    }

    for (size_t i = 0; i < frames.size(); ++i) {
        cv::Mat popped;
        if (!fq.pop(popped)) { LOG("pop failed unexpectedly"); return false; }
        if (popped.empty()) { LOG("popped frame empty"); return false; }
    }

    return fq.empty();
}

bool test_pop_on_empty_returns_false() {
    FrameQueue fq(3);
    cv::Mat tmp;
    fq.close();
    return !fq.pop(tmp);
}

bool test_queue_overflow_behavior() {
    const size_t max_size = 3;
    FrameQueue fq(max_size);
    auto frames = generate_dummy_frames(10);

    for (auto &f : frames) {
        while (fq.size() >= max_size) {
            cv::Mat tmp;
            if (!fq.pop(tmp)) break;
        }
        fq.push(f);
    }
    return fq.size() <= max_size;
}

bool test_repeated_push_pop() {
    FrameQueue fq(5);
    auto frames = generate_dummy_frames(5);
    for (int cycle = 0; cycle < 10; ++cycle) {
        for (auto &f : frames) fq.push(f);
        while (!fq.empty()) {
            cv::Mat tmp;
            fq.pop(tmp);
        }
    }
    return fq.empty();
}

// threaded producer/consumer
void threaded_producer(FrameQueue& fq, const vector<cv::Mat>& frames) {
    for (const auto& f : frames) {
        if (!fq.push(f)) break; // closed
        this_thread::sleep_for(chrono::microseconds(50));
    }
}

void threaded_consumer(FrameQueue& fq, atomic<int>& processed) {
    cv::Mat frame;
    while (fq.pop(frame)) {
        processed++;
    }
}

bool test_thread_safety_stress_and_shutdown() {
    FrameQueue fq(10);
    atomic<int> processed{0};

    auto frames1 = generate_dummy_frames(50, 320, 240);
    auto frames2 = generate_dummy_frames(50, 320, 240);

    thread p1(threaded_producer, ref(fq), cref(frames1));
    thread p2(threaded_producer, ref(fq), cref(frames2));
    thread c1(threaded_consumer, ref(fq), ref(processed));
    thread c2(threaded_consumer, ref(fq), ref(processed));

    p1.join();
    p2.join();

    fq.close(); // signal consumers

    auto start = chrono::steady_clock::now();
    c1.join();
    c2.join();
    auto dur = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start).count();

    bool valid_count = (processed > 0 && processed <= 100);
    if (!valid_count) LOG("Processed count invalid: " << processed);
    if (dur > 10) LOG("Consumers took >10s to finish (timeout): " << dur << "s");
    return valid_count;
}

bool test_zero_max_size_behaviour() {
    FrameQueue fq(0);
    cv::Mat f = cv::Mat::zeros(10, 10, CV_8UC3);
    bool pushed = fq.push(f); 
    fq.close();
    cv::Mat tmp;
    bool popped = fq.pop(tmp);
    return (!pushed) && (!popped) && fq.empty();
}

int main() {
    int passed = 0, total = 0;
    RUN_TEST(test_push_pop_basic);
    RUN_TEST(test_pop_on_empty_returns_false);
    RUN_TEST(test_queue_overflow_behavior);
    RUN_TEST(test_repeated_push_pop);
    RUN_TEST(test_thread_safety_stress_and_shutdown);
    RUN_TEST(test_zero_max_size_behaviour);

    cout << "----------------------------------------\n";
    cout << "Test summary: Passed " << passed << " / " << total << " tests\n";
    return (passed == total) ? 0 : 1;
}
