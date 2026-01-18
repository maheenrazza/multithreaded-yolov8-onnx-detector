#!/bin/bash

echo "=== Video File Checker ==="
echo ""

# Check if data directory exists
if [ -d "data" ]; then
    echo "✓ data directory found"
    echo "Video files in data/:"
    ls -la data/*.{avi,mp4,mov,mkv} 2>/dev/null || echo "  No video files found"
    total_videos=$(ls -1 data/*.{avi,mp4,mov,mkv} 2>/dev/null | wc -l)
    if [ "$total_videos" -gt "0" ]; then
        echo "  Found $total_videos video files"
    fi
    echo ""
else
    echo "✗ data directory not found"
    echo "  Please ensure the data directory exists in the current path"
    echo ""
fi

# Check if model exists
if [ -f "yolov8n.onnx" ]; then
    echo "✓ Default model yolov8n.onnx found"
    size=$(ls -lh yolov8n.onnx | awk '{print $5}')
    echo "  Model size: $size"
else
    echo "✗ Default model yolov8n.onnx not found"
    echo "  Available .onnx files:"
    ls -la *.onnx 2>/dev/null || echo "  No .onnx files found"
fi

echo ""
echo "=== Usage Examples ==="
echo "Test with webcam:"
echo "  ./inference_engine --model yolov8n.onnx --video 0"
echo ""
echo "Test with video file:"
echo "  ./inference_engine --model yolov8n.onnx --video data/sample.avi"
echo ""

# Optional: Show system info
echo "=== System Info ==="
echo "OpenCV version: $(pkg-config --modversion opencv4 2>/dev/null || echo "Not found")"
echo "CPU: $(cat /proc/cpuinfo | grep "model name" | head -n1 | cut -d: -f2 | xargs)"
echo "Memory: $(free -h | grep "Mem:" | awk '{print $2}')"
echo ""