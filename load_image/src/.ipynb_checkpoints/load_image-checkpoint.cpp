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
// TORCH
#include <torch/script.h>
#include <torch/torch.h>
// Model of mobile platform openvivo
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset8.hpp>

// Function


// PROTOTYPES
std::vector<std::vector<std::vector<float>>> input_image;

// Function to convert ov::Tensor to cv::Mat
cv::Mat convertOvTensorToCvMat(const ov::Tensor& tensor) {
    // Ensure the tensor shape is as expected: [1, 3, 1088, 1920]
    const ov::Shape& shape = tensor.get_shape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != 1088 || shape[3] != 1920) {
        std::cerr << "Unexpected tensor shape: ";
        for (size_t dim : shape) std::cout << dim << " ";
        std::cerr << std::endl;
        throw std::runtime_error("Tensor shape is not [1, 3, 1088, 1920]");
    }

    // Determine the data type of the tensor and map it to OpenCV Mat format
    cv::Mat mat;
    if (tensor.get_element_type() == ov::element::f32) {
        // Data is float, create a Mat with float type
        float* data = tensor.data<float>();
        // Create cv::Mat with dimensions [1088, 1920] and 3 channels with float type
        cv::Mat chw_mat(3, 1088, CV_32FC(1920), data); // Treat data as one continuous float array

        // Convert CHW format to HWC by using OpenCV split and merge
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; ++i) {
            channels[i] = cv::Mat(1088, 1920, CV_32FC1, data + i * 1920 * 1088);
        }

        // Merge into a single 3-channel image
        cv::merge(channels, mat);
        // Convert to 8-bit unsigned integer type if necessary
        mat.convertTo(mat, CV_8UC3, 255.0);
        
    } else if (tensor.get_element_type() == ov::element::u8) {
        // Data is uint8, create a Mat directly from the raw data
        uint8_t* data = tensor.data<uint8_t>();
        cv::Mat chw_mat(3, 1088, CV_8UC(1920), data); // Treat data as one continuous unsigned 8-bit array

        // Convert CHW format to HWC
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; ++i) {
            channels[i] = cv::Mat(1088, 1920, CV_8UC1, data + i * 1920 * 1088);
        }

        // Merge into a single 3-channel image
        cv::merge(channels, mat);
    } else {
        throw std::runtime_error("Unsupported data type in tensor");
    }

    return mat;
}


void check_ov_shape(ov::Tensor torch_tensor);

int main(int argc, char** argv)
{
    // const char *model_xml = "../models/convnext-tiny-llie-sim.xml";
    // const char *model_bin = "../models/convnext-tiny-llie-sim.bin";

    const char *model_xml = "../models/reseffnet_gan_div4_1088_1920.xml";
    const char *model_bin = "../models/reseffnet_gan_div4_1088_1920.bin";
    
    const char *input_image_path = "../input.jpg";
    const char *output_image_path = "../output_cpp.jpg";

    std::cout << "PROCESSING " << std::endl;
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;

    std::cout << "Available " << core.get_available_devices()<< std::endl;
    
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(model_xml);

    // Step 3. Read input image
    cv::Mat image = cv::imread(input_image_path);
    
    // Step 4. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ov::Shape input_shape = {1,image.rows,image.cols,image.channels()};
    // Specify input image format
    ppp.input().tensor().set_shape(input_shape).set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    
    // Specify preprocess pipeline to input image without resizing    
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255., 255., 255.});

    //  Specify model's input layout
    ppp.input().model().set_layout("NCHW");
    // Specify output results format
    ppp.output().tensor().set_element_type(ov::element::f32);
    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    
    // // Step 5. Create tensor from image
    float *input_data = (float*) image.data;
    
    // ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);// fixed dynamic shape by change into staic shape
    ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), input_shape, input_data);


    // Step 6. Create an infer request for model inference 
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();

    for (int k=0;k<5;k++){
        // Run inference on the input image
        auto start = std::chrono::high_resolution_clock::now();    
        //Do inference synchronously
        infer_request.infer();
        auto end = std::chrono::high_resolution_clock::now();
        // Print the processing-time
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "TOTAL TIME: " << elapsed.count() << std::endl;
    }
    
    // Step 7. Retrieve inference results 
    const ov::Tensor &output_tensor = infer_request.get_output_tensor();
    ov::Shape output_shape = output_tensor.get_shape();

    //check_ov_shape(output_tensor);
    
    // Convert ov:Tensor --> torch::Tensor
    // Get data type and shape information
    auto ov_shape  = output_tensor.get_shape();
    std::vector<int64_t> torch_shape(ov_shape.begin(), ov_shape.end());
    auto ov_type = output_tensor.get_element_type();
    torch::Dtype torch_dtype = torch::kFloat32;  // ov::Tensor output type f32
    // Get the data pointer
    void *data_ptr = output_tensor.data();    
    torch::TensorOptions options(torch_dtype);
    torch::Tensor torch_tensor = torch::from_blob(data_ptr, torch_shape, options);
    //Remove batch dimension
    torch_tensor = torch_tensor.squeeze(0); // [3, 1088, 1920]
    
    // Convert tensor from [C, H, W] to [H, W, C]
    torch_tensor = torch_tensor.permute({1, 2, 0}); //[1088, 1920, 3]

    // Ensure the tensor is of type uint8 and convert denormalize the tensor --> to torch 
    if (torch_tensor.scalar_type() != torch::kUInt8) {
        torch_tensor = torch_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    }
      // Convert torch::tensor to cv::Mat
    cv::Mat image_out(torch_tensor.size(0), torch_tensor.size(1), CV_8UC3, torch_tensor.contiguous().data_ptr());
    

    // cv::Mat image_out = convertOvTensorToCvMat(output_tensor);
    cv::Mat image_BGR;
    cv::cvtColor(image_out, image_BGR, cv::COLOR_RGB2BGR);// original : COLOR_BGR2RGB 
   
    // // save image
    cv::imwrite(output_image_path, image_BGR);  
    std::cout << "END " << std::endl;
    
    return 0;
}

    // // print dim tensor
    // std::cout << torch::_shape_as_tensor(tensor) << std::endl;

    // Check the height and weight of image
    // std::cout << "Width : " << image.size() << std::endl;
    // std::cout << "Height: " << image.rows/cols/channels() << std::endl;

  // // Access the data from the ov::Tensor
    // unsigned char* tensor_data = input_tensor.data<unsigned char>();

    
    // // Print the first few pixels to verify
    // for (size_t i = 0; i < std::min<size_t>(10, input_tensor.get_size()); ++i) {
    //     std::cout << static_cast<int>(tensor_data[i]) << " ";
    // }
    // std::cout << std::endl;

void check_ov_shape(ov::Tensor torch_tensor){
     std::cout << "tensor_shape" << std::endl;
    // Retrieve and print the shape from the ov::Tensor
    auto shape = torch_tensor.get_shape();
    std::cout << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}
cv::Mat preprocess(ov::Tensor output_tensor){
    // Convert ov:Tensor --> torch::Tensor
    // Get data type and shape information
    auto ov_shape  = output_tensor.get_shape();
    std::vector<int64_t> torch_shape(ov_shape.begin(), ov_shape.end());
    auto ov_type = output_tensor.get_element_type();
    torch::Dtype torch_dtype = torch::kFloat32;  // ov::Tensor output type f32
    // Get the data pointer
    void *data_ptr = output_tensor.data();    
    torch::TensorOptions options(torch_dtype);
    torch::Tensor torch_tensor = torch::from_blob(data_ptr, torch_shape, options);
    //Remove batch dimension
    torch_tensor = torch_tensor.squeeze(0); // [3, 1088, 1920]
    
    // Convert tensor from [C, H, W] to [H, W, C]
    torch_tensor = torch_tensor.permute({1, 2, 0}); //[1088, 1920, 3]
    
    // Ensure the tensor is of type uint8 and convert denormalize the tensor --> to torch 
    if (torch_tensor.scalar_type() != torch::kUInt8) {
        torch_tensor = torch_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    }
      // Convert torch::tensor to cv::Mat
    cv::Mat image_out(torch_tensor.size(0), torch_tensor.size(1), CV_8UC3, torch_tensor.contiguous().data_ptr());
    cv::Mat image_BGR;
    cv::cvtColor(image_out, image_BGR, cv::COLOR_RGB2BGR);// original : COLOR_BGR2RGB
}



void check_torch_shape(torch::Tensor torch_tensor){
    /*
    Check Shape of torch :: Tensor 
    */
    auto sizes = torch_tensor.sizes();
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i != sizes.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}



  // cv::Mat image_out(img_height, img_width, (image.channels() == 1) ? CV_32FC1 : CV_32FC3, torch_tensor.data_ptr());

// text
    // auto ov_shape  = output_tensor.get_shape();
    // std::cout <<"ov::Tensor output shape " << ov_shape << std::endl;
    // std::vector<int64_t> torch_shape(ov_shape.begin(), ov_shape.end());
    // std::cout <<"torch::Tensor output shape " << torch_shape << std::endl;
    // auto ov_type = output_tensor.get_element_type();
    // std::cout <<"ov::Tensor output type " << ov_type << std::endl;
    