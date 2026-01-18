CXX := g++
OPENCV_CXXFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I./headers -I./onnxruntime-linux-x64-1.17.0/include $(OPENCV_CXXFLAGS) -pthread

SRC_DIR := src
HEADERS_DIR := headers
TESTS_DIR := tests
MODELS_DIR := models

SOURCES := $(SRC_DIR)/main.cpp $(SRC_DIR)/infer_engine.cpp $(SRC_DIR)/preprocess.cpp \
           $(SRC_DIR)/nms.cpp $(SRC_DIR)/frame_queue.cpp $(SRC_DIR)/frame.cpp 
OBJECTS := $(SOURCES:.cpp=.o)
TARGET := inference_engine

# Test sources
TEST_SOURCES := $(wildcard $(TESTS_DIR)/*.cpp)
TEST_TARGETS := $(patsubst $(TESTS_DIR)/%.cpp,$(TESTS_DIR)/%,$(TEST_SOURCES))

IMAGE_NAME := inference_engine
CONTAINER_NAME := engine_container

UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")

ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
    EXT := .exe
    RM := del /Q /F
    MKDIR := mkdir
    HOST_PWD := $(CURDIR)
    ONNX_LIB := -L./onnxruntime-windows-x64-1.17.0/lib -lonnxruntime -ldl -lpthread
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macOS
    EXT :=
    RM := rm -f
    MKDIR := mkdir -p
    HOST_PWD := $(shell pwd)
    ONNX_LIB := -L./onnxruntime-macos-x64-1.17.0/lib -lonnxruntime -ldl -lpthread
else
    PLATFORM := Unix
    EXT :=
    RM := rm -f
    MKDIR := mkdir -p
    HOST_PWD := $(shell pwd)
    ONNX_LIB := -L./onnxruntime-linux-x64-1.17.0/lib -lonnxruntime -ldl -lpthread
endif

.DEFAULT_GOAL := all

# ---------------- Main target ----------------
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) $(ONNX_LIB)

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------------- Tests ----------------
tests: $(TEST_TARGETS)
	@for test in $(TEST_TARGETS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done

$(TESTS_DIR)/test_inferengine: $(TESTS_DIR)/test_inferengine.cpp $(SRC_DIR)/infer_engine.o
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS) $(ONNX_LIB)

$(TESTS_DIR)/test_preprocess: $(TESTS_DIR)/test_preprocess.cpp $(SRC_DIR)/preprocess.o
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)

$(TESTS_DIR)/test_nms: $(TESTS_DIR)/test_nms.cpp $(SRC_DIR)/nms.o
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)


$(TESTS_DIR)/test_framequeue: $(TESTS_DIR)/test_framequeue.cpp $(SRC_DIR)/frame_queue.o
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(OPENCV_LIBS)

test-%: $(TESTS_DIR)/test_%
	./$<

# ---------------- Run ----------------
inference: $(TARGET)
	./$(TARGET) --video data/sample.mp4


# For testing with video files from ../data/
test-video: $(TARGET)
	@echo "Usage: make test-video VIDEO=../data/yourfile.avi"
	@echo "Available videos in ../data/:"
	@ls -la ../data/ 2>/dev/null || echo "Directory ../data/ not found"

# Quick test with specific video
run-cctv: $(TARGET)
	./$(TARGET) --model yolov8n.onnx --video data/Sample_video.mp4 --conf 0.3

# ---------------- Docker ----------------
docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run: docker-build
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-v "$(HOST_PWD):/app" \
		$(IMAGE_NAME)

docker-make: docker-build
	docker run --rm \
		--name $(CONTAINER_NAME) \
		-v "$(HOST_PWD):/app" \
		$(IMAGE_NAME) make $(ARGS)

# ---------------- Clean ----------------
clean:
	@$(RM) $(TARGET) $(OBJECTS) $(TEST_TARGETS) > /dev/null 2>&1 || true

# ---------------- Help ----------------
help:
	@echo "Available targets:"
	@echo "  all           - Build the main executable"
	@echo "  tests         - Build and run all tests"
	@echo "  inference     - Run inference with sample video"
	@echo "  car-counter   - Run car counter (use: make car-counter MODEL=path VIDEO=path)"
	@echo "  demo-webcam   - Demo with webcam"
	@echo "  demo-video    - Demo with sample video"
	@echo "  clean         - Clean build files"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run in Docker container"
	@echo "  help          - Show this help"

.PHONY: all tests inference car-counter demo-webcam demo-video docker-build docker-run docker-make clean help
