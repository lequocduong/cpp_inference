#include <algorithm>
#include <set>
#include <string>
#include "pw_base_api.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

const std::set<std::string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv"};

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

} // namespace

int main(int argc, char **argv)
{
    std::string input_path = argv[1];
    std::string save_path = argv[2];
    if (isVideoFile(input_path)) {
        /* INPUT */
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
        /* INPUT */

        /* OUTPUT */
        AVFormatContext* outputFormatContext = nullptr;
        avformat_alloc_output_context2(&outputFormatContext, nullptr, nullptr, save_path.c_str());
        if (!outputFormatContext) {
            printf("Could not create output context\n");
            return -1;
        }

        AVStream* outStream = avformat_new_stream(outputFormatContext, nullptr);
        if (!outStream) {
            printf("Failed allocating output stream\n");
            return -1;
        }

        AVCodecParameters* outCodecParameters = outStream->codecpar;
        outCodecParameters->codec_id = outputFormatContext->oformat->video_codec;
        outCodecParameters->codec_type = AVMEDIA_TYPE_VIDEO;
        outCodecParameters->width = codecContext->width;
        outCodecParameters->height = codecContext->height;
        outCodecParameters->format = AV_PIX_FMT_YUV420P;
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

        if (!(outputFormatContext->oformat->flags & AVFMT_NOFILE)) {
            if (avio_open(&outputFormatContext->pb, save_path.c_str(), AVIO_FLAG_WRITE) < 0) {
                printf("Could not open output file '%s'\n", save_path.c_str());
                return -1;
            }
        }

        if (avformat_write_header(outputFormatContext, nullptr) < 0) {
            printf("Error occurred when opening output file\n");
            return -1;
        }

        AVFrame* outFrame = av_frame_alloc();
        outFrame->format = outCodecContext->pix_fmt;
        outFrame->width = outCodecContext->width;
        outFrame->height = outCodecContext->height;
        if (av_frame_get_buffer(outFrame, 0) < 0) {
            printf("Could not allocate the video frame data\n");
            return -1;
        }
        /* OUTPUT */

        int pts = 0;
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
                    // HERE
                    int width = frame->width;
		            int height = frame->height;

	                int type = (frame->format == AV_PIX_FMT_YUV420P || frame->format == AV_PIX_FMT_YUVJ420P) ? VM_FORMAT_YUV420_P : 
                               (frame->format == AV_PIX_FMT_NV12 ? VM_FORMAT_NV12 : VM_FORMAT_YUV420_P);

                    GVM_SCREEN *yuv = pwbCreateImage(width, height, type);

                    int offset = 0;
                    for (int i = 0; i < height; i++) {
                        memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[0] + offset, 
                                frame->data[0] + i * frame->linesize[0], width);
                        offset += width;
                    }

                    if (type == VM_FORMAT_NV12) {
                        offset = 0;
                        for (int i = 0; i < height / 2; i++) {
                            memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[1] + offset,
                                frame->data[1] + i * frame->linesize[1], width);
                            offset += width;
                        }
                    } else if (type == VM_FORMAT_YUV420_P) {
                        offset = 0;
                        for (int i = 0; i < height / 2; i++) {
                            memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[1] + offset,
                                frame->data[1] + i * frame->linesize[1], width / 2);
                            offset += width / 2;
                        }

                        offset = 0;
                        for (int i = 0; i < height / 2; i++) {
                            memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[2] + offset,
                                frame->data[2] + i * frame->linesize[2], width / 2);
                            offset += width / 2;
                        }
                    } else {
                        printf("Unknown type: %d", type);
                        exit(0);
                    }

                    /* Enhancing */                    
                    GVM_SCREEN *enhanced_yuv;
                    // enhancer->process(/* GVM_SCREEN *src */ yuv, /* GVM_SCREEN **dst */ enhanced_yuv);
                    pwbReleaseImage(yuv);

                    width = enhanced_yuv->lWidth;
                    height = enhanced_yuv->lHeight;

                    /* Rendering */
                    // ...

                    memcpy(outFrame->data[0], (uint8_t *)enhanced_yuv->PixelArray.planar.pPixelArray[0], width * height);
                    if (type == VM_FORMAT_NV12) {
                        memcpy(outFrame->data[1], (uint8_t *)enhanced_yuv->PixelArray.planar.pPixelArray[1], width * height / 2);
                    } else if (type == VM_FORMAT_YUV420_P) {
                        memcpy(outFrame->data[1], (uint8_t *)enhanced_yuv->PixelArray.planar.pPixelArray[1], width * height / 4);
                        memcpy(outFrame->data[2], (uint8_t *)enhanced_yuv->PixelArray.planar.pPixelArray[2], width * height / 4);
                    } else {
                        printf("Unknown type: %d", type);
                        exit(0);
                    }
    
                    
                    
                    outFrame->pts = pts++;
                    pwbReleaseImage(enhanced_yuv);
                    
                    // HERE
                    
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

        av_write_trailer(outputFormatContext);

        av_frame_free(&outFrame);
        avcodec_close(outCodecContext);
        avio_closep(&outputFormatContext->pb);
        avformat_free_context(outputFormatContext);

        av_frame_free(&frame);
        avcodec_close(codecContext);
        avformat_close_input(&formatContext);
    } else {
        printf("Unsupported input file: %s\n", input_path.c_str());
        return -1;
    }

    return 0;
}