#include "base_model.h"
#ifdef ONNXRUNTIME
// #include <providers/nnapi/nnapi_provider_factory.h>
#endif

using namespace cv;
using namespace AIBox;

BaseModel::BaseModel(const Scalar &mean, const Scalar &std)
    : device_(Device::CPU),
      is_dynamic_(false),
      num_outputs_(0),
      mean_(mean),
      std_(std)
{
}

BaseModel::~BaseModel()
{
}

bool BaseModel::load(const std::string &model_path, Device device, Precision precision)
{
    device_ = device;

#ifdef ONNXRUNTIME
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_MOBILE");
    Ort::SessionOptions so;
    if (device == Device::GPU) {
        // For Android
        // uint32_t nnapi_flags = 0;
        // nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
        // if (precision == Precision::FP16)
        //     nnapi_flags |= NNAPI_FLAG_USE_FP16;
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
        // For CUDA
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0));
    }
    session_ = Ort::Session(env, model_path.c_str(), so);

    Ort::AllocatorWithDefaultOptions allocator;

    int input_count = session_.GetInputCount();
    input_names_.reserve(input_count);
    for (int i = 0; i < input_count; ++i) {
        auto name_ptr = session_.GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name_ptr.get());

        Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

        printf("Input shape: ");
        for (auto shape : inputTensorShape)
            printf(" %ld", shape);
        printf("\n");

        if (i == 0) {
            if (inputTensorShape.size() >= 4) {
                is_dynamic_ = inputTensorShape[2] == -1 && inputTensorShape[3] == -1;
                if (!is_dynamic_)
                    input_size_ = {(int)inputTensorShape[3], (int)inputTensorShape[2]};
            }
        }
    }

    int output_count = session_.GetOutputCount();
    num_outputs_ = output_count;
    output_names_.reserve(output_count);
    for (int i = 0; i < output_count; ++i) {
        auto name_ptr = session_.GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name_ptr.get());
    }

#elif defined(OPENVINO)
    ov::Core core;
    auto model = core.read_model(model_path);
    compiled_model_ = core.compile_model(model, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    infer_request_ = compiled_model_.create_infer_request();
#endif

    return true;
}

void BaseModel::forward(std::vector<Mat> &ims)
{
    std::vector<std::vector<float>> blob_list;
    std::vector<std::vector<int64_t>> input_shape_list;

    for (auto &im : ims) {
        std::vector<Mat> ims;
        cvtColor(im, im, COLOR_BGR2RGB);
        im.convertTo(im, CV_32F, 1.0f / 255.0f);
        im = (im - mean_) / std_;
        ims.push_back(im);

        Mat ims_blob;
        dnn::blobFromImages(ims, ims_blob);
        cv::Mat flat = ims_blob.reshape(1, ims_blob.total() * ims_blob.channels());
        std::vector<float> blob = ims_blob.isContinuous()? flat : flat.clone();
        blob_list.push_back(blob);

        std::vector<int64_t> input_shape = {1, im.channels(), im.rows, im.cols};
        input_shape_list.push_back(input_shape);
    }
    pre_cost_ = timer_.elapsed();
	forward<float>(blob_list, input_shape_list);
    infer_cost_ = timer_.elapsed();
}

void BaseModel::forward(Mat &im)
{
    std::vector<Mat> ims{im};
    forward(ims);
}

template<typename T>
void BaseModel::forward(const std::vector<std::vector<T>> &blob_list, const std::vector<std::vector<int64_t>> &input_shape_list)
{
#ifdef ONNXRUNTIME
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> input_tensors;
    for (int i = 0; i < blob_list.size(); ++i) {
        size_t input_tensor_size = 1;
        const auto &input_shape = input_shape_list[i];
        for (const auto element : input_shape)
            input_tensor_size *= element;

        input_tensors.emplace_back(std::move(
            Ort::Value::CreateTensor<T>(
                memory_info, (T*)blob_list[i].data(), input_tensor_size,
                input_shape.data(), input_shape.size()
        )));
    }

    std::vector<const char*> input_names_cstr;
    std::vector<const char*> output_names_cstr;

    for (auto &name : input_names_)
        input_names_cstr.push_back(name.data());
    for (auto &name : output_names_)
        output_names_cstr.push_back(name.data());

    session_.Run(Ort::RunOptions{nullptr},
                 input_names_cstr.data(),
                 input_tensors.data(),
                 input_tensors.size(),
                 output_names_cstr.data(),
                 output_names_cstr.size()).swap(pred_);

#elif defined(OPENVINO)
    ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), blob.data());
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();
#endif
}

// Explicit template instantiation
template void BaseModel::forward(const std::vector<std::vector<int32_t>> &blob_list, const std::vector<std::vector<int64_t>> &input_shape_list);
