# YOLOv8 ONNX Vehicle Detection (C++ · Multi-threaded)

A multi-threaded C++ object detection pipeline using a YOLOv8 **ONNX** model. The system runs a classic **producer–consumer** architecture: one thread reads frames from a video/webcam into a bounded queue, while another thread runs inference and post-processing (confidence filtering + NMS).

## Features

- ✅ **YOLOv8 ONNX inference** in C++
- ✅ **Producer–consumer pipeline** (frame reader thread + inference thread)
- ✅ **Bounded frame queue** to control memory / latency
- ✅ Runs on **video file** or **webcam**
- ✅ Configurable thresholds:
  - Confidence threshold (`--conf`)
  - NMS IoU threshold (`--nms`)
- ✅ Graceful shutdown (`Ctrl+C`)

---

## Demo


- `assets/screenshot.png`

---

## Requirements

Typical dependencies:

- C++17 compiler (GCC/Clang)
- CMake (if you use it)
- OpenCV (for video I/O + drawing)
- ONNX Runtime (for inference)

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
````

## How it Works (Architecture)

### Threads

* **Producer thread:** reads frames from the input source (video/webcam) and pushes them into a bounded `FrameQueue`.
* **Consumer thread:** pops frames from `FrameQueue`, runs YOLOv8 inference via `InferEngine`, applies post-processing (confidence threshold + NMS), and (optionally) draws boxes / outputs results.

### Graceful shutdown

SIGINT/SIGTERM flips a global atomic `running` flag which both threads observe, allowing safe termination without corrupting queue state.

