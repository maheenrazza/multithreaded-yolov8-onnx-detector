#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include "infer_engine.h"
#include "frame_queue.h"

// --- Global Running Flag and Signal Handler ---
std::atomic<bool> running(true);

void signalHandler(int signum) {
    std::cout << "\n[INFO] Received signal " << signum << ". Shutting down gracefully..." << std::endl;
    running = false;
}

extern void producer(FrameQueue& fq, const std::string& video_path, std::atomic<bool>& running);
extern void consumer(FrameQueue& fq, InferEngine& engine, std::atomic<bool>& running,
                    float conf_threshold, float nms_threshold);

// --- Argument Parser and Main Execution Logic ---
void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <path> [options]\n\n"
              << "A multi-threaded YOLOv8 object detection application.\n\n"
              << "Required Arguments:\n"
              << "  --model <path>     Path to the ONNX model file.\n\n"
              << "Optional Arguments:\n"
              << "  --video <path>     Path to video file or '0' for webcam. (Default: 0)\n"
              << "  --conf <float>     Confidence threshold for detections. (Default: 0.25)\n"
              << "  --nms <float>      NMS IoU threshold for filtering boxes. (Default: 0.45)\n"
              << "  --queue-size <int> Max number of frames to buffer. (Default: 24)\n"
              << "  --help             Show this help message.\n";
}

int main(int argc, char** argv) {
    // Set up signal handling for graceful shutdown (Ctrl+C).
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Default parameters
    std::string model_path, video_path = "0";
    float conf_threshold = 0.25f, nms_threshold = 0.6f;
    size_t queue_size = 24;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf_threshold = std::stof(argv[++i]);
        else if (arg == "--nms" && i + 1 < argc) nms_threshold = std::stof(argv[++i]);
        else if (arg == "--queue-size" && i + 1 < argc) queue_size = std::stoul(argv[++i]);
        else if (arg == "--help") { printUsage(argv[0]); return 0; }
    }

    if (model_path.empty()) {
        std::cerr << "Model argument is required\n";
        printUsage(argv[0]);
        return 1;
    }

    try {
        InferEngine engine(model_path);
        std::cerr << "Model loaded: " << model_path
                  << " (" << engine.getInputWidth() << "x" << engine.getInputHeight() << ")\n";

        FrameQueue fq(queue_size);

        std::thread prod_thread(producer, std::ref(fq), std::ref(video_path), std::ref(running));
        std::thread cons_thread(consumer, std::ref(fq), std::ref(engine), std::ref(running),
                                conf_threshold, nms_threshold);

        prod_thread.join();
        cons_thread.join();

        std::cerr << "Pipeline completed. Exiting.\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
