// model_inference.cc
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"  
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model data
extern const unsigned char mnist_model_quantized_int8_tflite[];
extern const unsigned int mnist_model_quantized_int8_tflite_len;

const int kTensorArenaSize = 60 * 1024;

uint8_t tensor_arena[kTensorArenaSize];



// 使用智能指针管理 interpreter
std::unique_ptr<tflite::MicroInterpreter> interpreter_ptr;

// Function to set up the model
void setup_model() {

    // Create error reporter
    static tflite::MicroErrorReporter error_reporter;

    // Load model from data
    const tflite::Model* model = tflite::GetModel(mnist_model_quantized_int8_tflite);

    // Create ops resolver
    static tflite::MicroMutableOpResolver<6> op_resolver;

    op_resolver.AddConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax(); 
    op_resolver.AddReshape();
    op_resolver.AddRelu();


    // Set up interpreter with memory arena
    interpreter_ptr = std::make_unique<tflite::MicroInterpreter>(
        model, 
        op_resolver, 
        tensor_arena, 
        kTensorArenaSize, 
        nullptr
    );

    TfLiteStatus status = interpreter_ptr->AllocateTensors();
   
    TfLiteTensor* input_tensor = interpreter_ptr->input(0);
    TfLiteTensor* output_tensor = interpreter_ptr->output(0);
    input_tensor->type = kTfLiteInt8;
    output_tensor->type = kTfLiteInt8;

}


//  run_inference() function 
int run_inference(const uint8_t* input_data) {
    TfLiteTensor* input_tensor = interpreter_ptr->input(0);
    TfLiteTensor* output_tensor = interpreter_ptr->output(0);
    
    // - Copy input data to model input tensor
    if (input_tensor->type == kTfLiteInt8) {
        for (int i = 0; i < input_tensor->bytes; ++i) {
            input_tensor->data.int8[i] = static_cast<int8_t>(input_data[i] - 128);
        }
    } else {
        return -1;
    }

    // - Invoke interpreter
    if (interpreter_ptr->Invoke() != kTfLiteOk) {
        return -1;
    }

    // - Read output from output tensor
    // - Return predicted class
    if (output_tensor->type == kTfLiteInt8) {
        float output_scale = output_tensor->params.scale;
        int output_zero_point = output_tensor->params.zero_point;
        
        int predicted_class = 0;
        float max_probability = (output_tensor->data.int8[0] - output_zero_point) * output_scale;
        
        for (int i = 1; i < 10; ++i) { 
            float probability = (output_tensor->data.int8[i] - output_zero_point) * output_scale;
            if (probability > max_probability) {
                max_probability = probability;
                predicted_class = i;
            }
        }
        return predicted_class;
    }
    
    return -1;
}

//Implement main() function
int main() {
  
    // Set up model
    setup_model();


    // - Create test input (28x28 image data)
    cv::Mat img = cv::imread("test_image.png", cv::IMREAD_GRAYSCALE);
    
    if (img.cols != 28 || img.rows != 28) {
        cv::resize(img, img, cv::Size(28, 28));
    }
    double minVal, maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    
    cv::Mat processed_img;
    img.convertTo(processed_img, CV_8UC1);

    uint8_t input_data[28 * 28];
    memcpy(input_data, processed_img.data, sizeof(input_data));
   
    // Run inference
    int predicted_class = run_inference(input_data);

    if (predicted_class >= 0) {
        std::cout << "Predict number: " << predicted_class << std::endl;
    } else {
        std::cout << "fail" << std::endl;
    }

    return 0;
}