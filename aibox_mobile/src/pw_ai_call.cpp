#include "pw_ai_call.h"
#include "ir_model.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

const std::set<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv"};
const std::set<std::string> image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"};

namespace {

bool hasValidExtension(const std::string &filename, const std::set<std::string> &exts) 
{
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos)
        return false;

    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return exts.find(ext) != exts.end();
}

bool isVideoFile(const std::string &filename) 
{
    return hasValidExtension(filename, video_exts);
}

bool isImageFile(const std::string &filename) 
{
    return hasValidExtension(filename, image_exts);
}

} // namespace

int AIBox::imageRestore(const std::string &model_path, const std::string &input_path, const std::string &save_path, 
                        const InferOptions &opt, InferStatistic &data)
{
    std::unique_ptr<BaseModel> model = std::make_unique<IRModel>(1, 0);
    Timer timer;
    if (!model->load(model_path, opt.device, opt.precision)) {
        printf("Load model failed: %s\n", model_path.c_str());
        return -1;
    }
    data.model_load_time = timer.elapsed();

    if (isImageFile(input_path)) {
        cv::Mat result;
        cv::Mat image = cv::imread(input_path);

        int iter_count = std::max(opt.iter_count, 1);
        std::vector<double> times;
        times.reserve(iter_count);
        Timer timer;
        for (int i = 0; i < iter_count; ++i) {
            TimeCost time_cost;
            result = ((IRModel *)model.get())->predict(image, time_cost);
            times.push_back(time_cost.infer);
            data.pre_cost = time_cost.pre;
            data.post_cost = time_cost.post;
        }
        data.total_cost = timer.elapsed();
        cv::imwrite(save_path, result);

        data.frame_width = image.cols;
        data.frame_height = image.rows;
        data.frame_count = times.size();
        data.min_infer_cost = *std::min_element(times.begin(), times.end());
        data.max_infer_cost = *std::max_element(times.begin(), times.end());
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        data.aver_infer_cost = iter_count > 1 ? (sum - data.max_infer_cost) / (times.size() - 1) : sum / times.size();
        data.one_frame_cost = data.pre_cost + data.post_cost + data.aver_infer_cost;
    } else if (isVideoFile(input_path)) {
        /* 输入 */
        AVFormatContext* formatContext = avformat_alloc_context();
        if (avformat_open_input(&formatContext, input_path.c_str(), nullptr, nullptr) != 0) {
            printf("Could not open file: %s\n", input_path.c_str());
            return -1;
        }

        if (avformat_find_stream_info(formatContext, nullptr) < 0) {
            printf("Could not find stream information\n");
            return -1;
        }

        // Find the first video stream
        int videoStreamIndex = -1;
        for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
            if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                videoStreamIndex = i;
                break;
            }
        }
        if (videoStreamIndex == -1) {
            printf("Could not find a video stream\n");
            return -1;
        }

        // Get a pointer to the codec context for the video stream
        AVCodecParameters* codecParameters = formatContext->streams[videoStreamIndex]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codecParameters->codec_id);
        if (!codec) {
            printf("Unsupported codec!\n");
            return -1;
        }

        AVCodecContext* codecContext = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codecContext, codecParameters) < 0) {
            printf("Could not copy codec context\n");
            return -1;
        }

        if (avcodec_open2(codecContext, codec, nullptr) < 0) {
            printf("Could not open codec\n");
            return -1;
        }

        AVFrame* frame = av_frame_alloc();
        AVPacket packet;

        SwsContext* swsContext = sws_getContext(
            codecContext->width, codecContext->height, codecContext->pix_fmt,
            codecContext->width, codecContext->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        /* 输入 */

        /* 输出 */
        AVFormatContext* outputFormatContext = nullptr;
        avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, save_path.c_str());
        if (!outputFormatContext) {
            printf("Could not create output context\n");
            return -1;
        }

        // 创建输出流
        AVStream* outStream = avformat_new_stream(outputFormatContext, nullptr);
        if (!outStream) {
            printf("Failed allocating output stream\n");
            return -1;
        }

        // 设置输出流的编码参数
        AVCodecParameters* outCodecParameters = outStream->codecpar;
        outCodecParameters->codec_id = outputFormatContext->oformat->video_codec;
        outCodecParameters->codec_type = AVMEDIA_TYPE_VIDEO;
        outCodecParameters->width = codecContext->width;
        outCodecParameters->height = codecContext->height;
        outCodecParameters->format = AV_PIX_FMT_YUV420P; // 输出为YUV420P格式
        outCodecParameters->bit_rate = 400000;

        const AVCodec* outCodec = avcodec_find_encoder(outCodecParameters->codec_id);
        if (!outCodec) {
            printf("Necessary encoder not found\n");
            return -1;
        }

        AVCodecContext* outCodecContext = avcodec_alloc_context3(outCodec);
        if (!outCodecContext) {
            printf("Could not allocate video codec context\n");
            return -1;
        }

        if (avcodec_parameters_to_context(outCodecContext, outCodecParameters) < 0) {
            printf("Could not initialize the video codec context\n");
            return -1;
        }
        
        AVRational input_framerate = formatContext->streams[videoStreamIndex]->r_frame_rate;
        outCodecContext->time_base = av_inv_q(input_framerate);
        outStream->time_base = outCodecContext->time_base;

        if (avcodec_open2(outCodecContext, outCodec, nullptr) < 0) {
            printf("Cannot open video encoder for output stream\n");
            return -1;
        }

        // 打开输出文件
        if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
            if (avio_open(&outputFormatContext->pb, save_path.c_str(), AVIO_FLAG_WRITE) < 0) {
                printf("Could not open output file '%s'\n", save_path.c_str());
                return -1;
            }
        }

        // 写入文件头
        if (avformat_write_header(outputFormatContext, nullptr) < 0) {
            printf("Error occurred when opening output file\n");
            return -1;
        }

        // 分配用于转换的帧
        AVFrame* outFrame = av_frame_alloc();
        outFrame->format = outCodecContext->pix_fmt;
        outFrame->width = outCodecContext->width;
        outFrame->height = outCodecContext->height;
        if (av_frame_get_buffer(outFrame, 0) < 0) {
            printf("Could not allocate the video frame data\n");
            return -1;
        }

        // 初始化用于颜色空间转换的SwsContext
        SwsContext* swsContextOutput = sws_getContext(
            codecContext->width, codecContext->height, AV_PIX_FMT_BGR24,
            outCodecContext->width, outCodecContext->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        /* 输出 */

        cv::Mat img(codecContext->height, codecContext->width, CV_8UC3);
        Timer timer;
        std::vector<double> times;

        while (av_read_frame(formatContext, &packet) >= 0) {
            if (packet.stream_index == videoStreamIndex) {
                int response = avcodec_send_packet(codecContext, &packet);
                if (response < 0) {
                    printf("Error while sending a packet to the decoder: %d\n", response);
                    break;
                }

                while (response >= 0) {
                    response = avcodec_receive_frame(codecContext, frame);
                    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                        break;
                    } else if (response < 0) {
                        printf("Error while receiving a frame from the decoder: %d\n", response);
                        return -1;
                    }

                    // Convert the frame to BGR format for OpenCV
                    uint8_t* img_data[1] = { img.data };
                    int linesize[1] = { static_cast<int>(img.step) };
                    sws_scale(swsContext, frame->data, frame->linesize, 0, codecContext->height, img_data, linesize);

                    TimeCost time_cost;
                    auto result = ((IRModel *)model.get())->predict(img, time_cost);
                    times.push_back(time_cost.infer);
                    data.pre_cost = time_cost.pre;
                    data.post_cost = time_cost.post;

                    // Convert the processed image back to YUV420P format for output
                    const uint8_t *out_img_data[] = { result.data };
                    int out_linesize[] = { static_cast<int>(result.step) };
                    sws_scale(swsContextOutput, out_img_data, out_linesize, 0, outCodecContext->height, outFrame->data, outFrame->linesize);

                    outFrame->pts = times.size();

                    // Encode the frame
                    response = avcodec_send_frame(outCodecContext, outFrame);
                    if (response < 0) {
                        printf("Error while sending a frame to the encoder: %d\n", response);
                        return -1;
                    }

                    while (response >= 0) {
                        response = avcodec_receive_packet(outCodecContext, &packet);
                        if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                            break;
                        } else if (response < 0) {
                            printf("Error while receiving a packet from the encoder: %d\n", response);
                            return -1;
                        }

                        packet.stream_index = outStream->index;
                        av_packet_rescale_ts(&packet, outCodecContext->time_base, outStream->time_base);
                        response = av_interleaved_write_frame(outputFormatContext, &packet);
                        if (response < 0) {
                            printf("Error while writing packet: %d\n", response);
                            return -1;
                        }
                        av_packet_unref(&packet);
                    }
                }
            }
            av_packet_unref(&packet);
        }

        data.total_cost = timer.elapsed();
        data.frame_width = codecContext->width;
        data.frame_height = codecContext->height;
        data.fps = (double)input_framerate.num / (double)input_framerate.den;
        data.frame_count = times.size();
        data.min_infer_cost = *std::min_element(times.begin(), times.end());
        data.max_infer_cost = *std::max_element(times.begin(), times.end());
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        data.aver_infer_cost = times.size() > 1 ? (sum - data.max_infer_cost) / (times.size() - 1) : sum / times.size();
        data.one_frame_cost = data.pre_cost + data.post_cost + data.aver_infer_cost;

        av_write_trailer(outputFormatContext);

        sws_freeContext(swsContextOutput);
        av_frame_free(&outFrame);
        avcodec_close(outCodecContext);
        avio_closep(&outputFormatContext->pb);
        avformat_free_context(outputFormatContext);

        sws_freeContext(swsContext);
        av_frame_free(&frame);
        avcodec_close(codecContext);
        avformat_close_input(&formatContext);
    } else {
        printf("Unsupported input file: %s\n", input_path.c_str());
        return -1;
    }

    return 0;
}