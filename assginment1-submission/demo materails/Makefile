# 项目名称
PROJECT = TFLiteMicroProject

# 源文件
SOURCES = part3_micro_deployment.cc model_data.cc

# 目标文件
TARGET = model_inference

# TensorFlow Lite Micro 库路径
TFLITE_MICRO_LIB = /Users/bbfish/develop/assignment1/TFLM/tflite-micro/gen/osx_arm64_default_gcc/lib/libtensorflow-microlite.a

# OpenCV 包的路径
OPENCV_LIBS = `pkg-config --cflags --libs opencv4`

# 编译器
CXX = g++
CXXFLAGS = -std=c++14
CXXFLAGS += -I/Users/bbfish/develop/assignment1/TFLM/tflite-micro/
CXXFLAGS += -I/Users/bbfish/develop/assignment1/TFLM/tflite-micro/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
CXXFLAGS += -I/Users/bbfish/develop/assignment1/TFLM/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gemmlowp

# 目标规则
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(TFLITE_MICRO_LIB) $(OPENCV_LIBS)

# 清理生成的文件
clean:
	rm -f $(TARGET)

.PHONY: all clean
