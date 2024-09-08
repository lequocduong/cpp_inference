#ifndef _AI_BOX_BASE_MODEL_H_
#define _AI_BOX_BASE_MODEL_H_

#include "timer.h"
#include <opencv2/opencv.hpp>

#ifdef ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#ifdef OPENVINO
#include <openvino/openvino.hpp>
#endif

namespace AIBox {

enum class Device
{
	CPU,
	GPU
};

enum class Precision
{
	FP32,
	FP16,
    INT8
};

class BaseModel
{
public:
	BaseModel(const cv::Scalar &mean, const cv::Scalar &norm);
	virtual ~BaseModel();

    bool load(const std::string &model_path, Device device, Precision precision);

protected:
    void forward(cv::Mat &im);
    void forward(std::vector<cv::Mat> &ims);

	template<typename T>
	void forward(const std::vector<std::vector<T>> &blob_list, const std::vector<std::vector<int64_t>> &input_shape_list);

#ifdef ONNXRUNTIME
    Ort::Env env{nullptr};
    Ort::Session session_{nullptr};
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<Ort::Value> pred_;

#elif defined(OPENVINO)
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
#endif

	Device device_;
    bool is_dynamic_;
    cv::Size input_size_;
    int num_outputs_;
    cv::Scalar mean_;
    cv::Scalar std_;

    double pre_cost_ = 0.;
    double infer_cost_ = 0.;
    double post_cost_ = 0.;
    Timer timer_;
};

} // namespace AIBox

#endif // _AI_BOX_BASE_MODEL_H_
