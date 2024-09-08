#include "ir_model.h"
#include <opencv2/opencv.hpp>

#if defined(NCNN)
#include "net.h"
#endif

using namespace AIBox;

IRModel::IRModel(int upscale, int tile)
    : BaseModel({0, 0, 0}, {1, 1, 1}),
      upscale_(upscale),
      window_size_(64),
      tile_(tile),
      tile_pad_(32)
{
}

cv::Mat IRModel::preprocess(const cv::Mat &im0, cv::Size &size)
{
    cv::Mat result;
    int im_wid = im0.cols;
    int im_hei = im0.rows;

    cv::Mat im = im0;

    if (!is_dynamic_) {
        float ratio = static_cast<float>(im_wid) / static_cast<float>(im_hei);
        if (ratio > 1.0) {
            im_wid = input_size_.width;
            im_hei = static_cast<int>(input_size_.width / ratio);
        } else {
            im_hei = input_size_.height;
            im_wid = static_cast<int>(input_size_.height * ratio);
        }
        cv::resize(im0, im, cv::Size(im_wid, im_hei));
    }

    size = {im_wid, im_hei};

    int pad_w = ceil(1.f * im_wid / window_size_) * window_size_ - im_wid;
    int pad_h = ceil(1.f * im_hei / window_size_) * window_size_ - im_hei;
    cv::copyMakeBorder(im, result, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, {0});
    return result;
}

cv::Mat IRModel::predict(const cv::Mat &im0, TimeCost &time_cost)
{
    // timer_.reset();

    // int im_wid = im0.cols;
    // int im_hei = im0.rows;

    // std::vector<cv::Mat> inputs;
    // cv::Size size;
    // inputs.push_back(preprocess(im0, size));
    // im_wid = size.width;
    // im_hei = size.height;

    // forward(inputs);

    // int out_im_hei = im_hei * upscale_;
    // int out_im_wid = im_wid * upscale_;

#if defined(ONNXRUNTIME)
    const auto *raw_output = pred_[0].GetTensorData<float>();
    auto output_shape = pred_[0].GetTensorTypeAndShapeInfo().GetShape();
    int ow = output_shape[3];
    int oh = output_shape[2];
    int owh = ow * oh;

    cv::Mat channelR(oh, ow, CV_32FC1, (void*)(raw_output));
    cv::Mat channelG(oh, ow, CV_32FC1, (void*)(raw_output + owh));
    cv::Mat channelB(oh, ow, CV_32FC1, (void*)(raw_output + 2 * owh));

    std::vector<cv::Mat> channels = {channelB, channelG, channelR};
    cv::Mat result;
    cv::merge(channels, result);
    result({0, out_im_hei}, {0, out_im_wid}).convertTo(result, CV_8UC3, 255.0, 0);

    post_cost_ = timer_.elapsed();
    time_cost.pre = pre_cost_;
    time_cost.post = post_cost_;
    time_cost.infer = infer_cost_;

#elif defined(OPENVINO)
    const ov::Tensor &output_tensor = infer_request_.get_output_tensor();
    const auto *raw_output = output_tensor.data<float>();
    auto output_shape = output_tensor.get_shape();

#elif defined(NCNN)
    ncnn::Net net;    
    const char *model_param = "../models/efficientunet_v2_nano_llie.ncnn.param";
    const char *model_bin = "../models/efficientunet_v2_nano_llie.ncnn.bin";

    int ret = net.load_param(model_param);
    
    ret = net.load_model(model_bin);
        
   cv::cvtColor(im0, im0, cv::COLOR_BGR2RGB);
    
    // Convert image data to ncnn format
    ncnn::Mat in = ncnn::Mat::from_pixels(im0.data,
                                             ncnn::Mat::PIXEL_RGB,
                                             im0.cols, im0.rows);

    // Data preprocessing (normalization)

    const float scal[] = {0.003915, 0.003915, 0.003915};
    in.substract_mean_normalize(0, scal); // 0-255  -->  0-1

    // Perform inference
    ncnn::Extractor ex = net.create_extractor();
    // Use the NPU (if available)
//    net.opt.use_vulkan_compute = true; // Enable Vulkan (NPU) 

    auto start = std::chrono::high_resolution_clock::now();
    ex.input("in0", in);
    // ex.input("in0", in0);
    ncnn::Mat out;
    ex.extract("out0", out);
    // ex.extract("out0", out0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
   std::cout << "TOTAL TIME: " << elapsed.count() << std::endl;

    // Denormalize the ncnn output to the range of 0-255
    const float scal2[] = {255.0f, 255.0f, 255.0f};
    out.substract_mean_normalize(0, scal2);

    // Convert ncnn output back to OpenCV format
    cv::Mat enhanced_img(im0.rows, im0.cols, CV_8UC3);
    out.to_pixels(enhanced_img.data, ncnn::Mat::PIXEL_RGB);
    enhanced_img.convertTo(enhanced_img, CV_8UC3);

    // cv::cvtColor(enhanced_img, enhanced_img, cv::COLOR_RGB2RGBA);
    // cv::cvtColor(enhanced_img, enhanced_img, cv::COLOR_RGB2GRB);
    cv::Mat result = enhanced_img;
#endif


    // int ow = output_shape[3];
    // int oh = output_shape[2];
    // int owh = ow * oh;

    // cv::Mat channelR(oh, ow, CV_32FC1, (void*)(raw_output));
    // cv::Mat channelG(oh, ow, CV_32FC1, (void*)(raw_output + owh));
    // cv::Mat channelB(oh, ow, CV_32FC1, (void*)(raw_output + 2 * owh));

    // std::vector<cv::Mat> channels = {channelB, channelG, channelR};
    // cv::Mat result;
    // cv::merge(channels, result);
    // result({0, out_im_hei}, {0, out_im_wid}).convertTo(result, CV_8UC3, 255.0, 0);

    // post_cost_ = timer_.elapsed();
    // time_cost.pre = pre_cost_;
    // time_cost.post = post_cost_;
    // time_cost.infer = infer_cost_;

    return result;
}