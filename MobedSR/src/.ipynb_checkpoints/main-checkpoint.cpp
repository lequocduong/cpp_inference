#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <type_traits>
// OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
//Tflite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

// TORCH
#include <torch/script.h>
#include <torch/torch.h>

void print_img_shape(const cv::Mat& img, const std::string& label);
void print_tensor_shape(const TfLiteTensor* tensor);
cv::Mat load_and_preprocess_image(const std::string& image_path, const std::vector<int>& input_shape, bool fp);
torch::Tensor tf_lite_to_torch_tensor(const TfLiteTensor* tensor);



// Function to convert TfLiteTensor to cv::Mat
cv::Mat convertTfLiteTensorToCvMat(const TfLiteTensor* tensor) {
    // Ensure the tensor shape is [1, 1080, 1920, 3]
    if (tensor->dims->size != 4 || tensor->dims->data[0] != 1 || 
        tensor->dims->data[1] != 1080 || tensor->dims->data[2] != 1920 || tensor->dims->data[3] != 3) {
        std::cerr << "Unexpected tensor shape: ";
        for (int i = 0; i < tensor->dims->size; ++i) {
            std::cout << tensor->dims->data[i] << " ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor shape is not [1, 1080, 1920, 3]");
    }

    cv::Mat mat;

    // Check the data type and convert accordingly
    if (tensor->type == kTfLiteFloat32) {
        // For float32, convert to CV_32FC3 and scale to 8-bit
        float* data = reinterpret_cast<float*>(tensor->data.f);
        cv::Mat float_mat(1080, 1920, CV_32FC3, data);
        float_mat.convertTo(mat, CV_8UC3, 255.0);  // Convert from float to 8-bit (scaling by 255)
    } else if (tensor->type == kTfLiteUInt8) {
        // For uint8, create a Mat directly
        uint8_t* data = tensor->data.uint8;
        mat = cv::Mat(1080, 1920, CV_8UC3, data).clone(); // Clone to create a separate copy
    } else {
        throw std::runtime_error("Unsupported tensor data type");
    }
    
    return mat;
}

int main(int argc, char** argv) {    

    const char *model_path = "../models/evsrnet_x4.tflite";    
    const char *input_image_path = "../input_880_480.jpg";
    const char *output_image_path = "../output_cpp.jpg";
    
    std::cout << "Processing" << std::endl;
    // Load model
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
     if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return -1;
    }    
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    
    // Prepare GPU delegate.
    // GPU delegate options 
    // TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    // options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    // options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    // options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    // options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    // TfLiteDelegate* delegate = TfLiteGpuDelegateV2Create(&options);
    
    // TfLiteDelegate *delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);    
    // builder.AddDelegate(delegate);

    // Build interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;    
    builder(&interpreter);

    // Allocate tensors
    interpreter->AllocateTensors();  
    
    // Get input tensor size + rezie into 1 1080 1920 3 (B,W,H,C)
    const auto &dimensions = interpreter->tensor(interpreter->inputs()[0])->dims;    
    std::vector<int> input_tensor_shape;

    input_tensor_shape.resize(dimensions->size);
    for (auto i = 0; i < dimensions->size; i++)
    {
        input_tensor_shape[i] = dimensions->data[i];        
    }
    auto input_height = input_tensor_shape[1];
    auto input_width = input_tensor_shape[2];

    // // show input shape
    // std::ostringstream input_string_stream;
    // std::copy(input_tensor_shape.begin(), input_tensor_shape.end(), std::ostream_iterator<int>(input_string_stream, " "));
    // std::cout << "Input shape: " << input_string_stream.str() << std::endl;
    
    // input image - resize - cvtColor - normalize
    cv::Mat img = cv::imread(input_image_path);
    cv::resize(img, img, cv::Size(input_width, input_height)); 
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);   
    img.convertTo(img, CV_32F, 1.0 / 255.0); // convert to 32F 
    
    // Set input tensor 
    float *input_ptr = interpreter->typed_input_tensor<float_t>(0);
    // Assign values to input tensor
    std::memcpy(input_ptr, img.data, img.total() * img.elemSize()); 

    // const auto &start_time = std::chrono::steady_clock::now();  
    // // Invoke interpreter - run inference
    // interpreter->Invoke();      
    // const auto &end_time = std::chrono::steady_clock::now();     
    // std::chrono::duration<double> time_span = end_time - start_time;    
    // std::cout << "Time inference: " << time_span.count() << std::endl; 
    
    for (int k=0;k<5;k++){
        // Run inference on the input image
        auto start = std::chrono::high_resolution_clock::now();    
        //Do inference synchronously
        interpreter->Invoke();    
        auto end = std::chrono::high_resolution_clock::now();
        // Print the processing-time
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time inference: " << elapsed.count() << std::endl;
    }
    // Get output data
    int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index); //  get tensor    

    print_tensor_shape(output_tensor); //[1, 1080, 1920, 3]
    
    cv::Mat image_out = convertTfLiteTensorToCvMat(output_tensor);
    cv::Mat image_BGR;
    cv::cvtColor(image_out, image_BGR, cv::COLOR_RGB2BGR);// original : COLOR_BGR2RGB

    // save image
    cv::imwrite(output_image_path, image_BGR); 
    std::cout << "End" << std::endl;
    
    return 0;   
}

cv::Mat load_and_preprocess_image(const std::string& image_path, const std::vector<int>& input_shape, bool fp) {
    // NOT DONE YET
    // Load the image    
    cv::Mat img = cv::imread(image_path);
    // Check if the image was loaded successfully
    if (img.empty()) {
        std::cerr << "Error: Could not load image at " << image_path << std::endl;
        return cv::Mat();
    }

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // print_img_shape(img, "Original Image Shape");    

    // // Add batch dimension (expand dimensions) - error here
    // img = img.reshape(0, cv::Size(img.cols, img.rows * img.channels())); // Flatten into a single channel
    
    // img = img.reshape(1, cv::Size(img.cols, img.rows)); // Add batch dimension

    if (fp) {
        // Convert to float32 and normalize
        img.convertTo(img, CV_32F, 1.0 / 255.0);
    } else {
        // Convert to int8
        img.convertTo(img, CV_8S);
    }

    return img;
}

// Function to convert TfLiteTensor to torch::Tensor
torch::Tensor tf_lite_to_torch_tensor(const TfLiteTensor* tensor) {
    // Extract the shape from TfLiteTensor
    std::vector<int64_t> shape(tensor->dims->size);
    for (int i = 0; i < tensor->dims->size; ++i) {
        shape[i] = tensor->dims->data[i];
    }

    // Create a torch::Tensor from the TfLiteTensor data
    // Assuming the tensor data is float. Change to torch::kUInt8 or other if necessary.
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor torch_tensor = torch::from_blob(tensor->data.f, shape, options);

    return torch_tensor.clone();  // Clone to ensure tensor is not affected by TfLiteTensor's lifetime
}

void print_img_shape(const cv::Mat& img, const std::string& label) {
    std::cout << label << "- Height: " << img.cols
              <<".  Width: " << img.rows              
              << ", Channels: " << img.channels()
              << std::endl;
}

void print_tensor_shape(const TfLiteTensor* tensor) {
    std::cout << "Tensor shape: [";
    for (int i = 0; i < tensor->dims->size; ++i) {
        std::cout << tensor->dims->data[i];
        if (i < tensor->dims->size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}