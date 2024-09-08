#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"


// Function to convert ncnn::Mat to cv::Mat
cv::Mat ncnnMatToCvMat(const ncnn::Mat& ncnnMat);
void pixel_print(cv::Mat &cvMat, ncnn::Mat &img) ;
// cv::Mat cvMat = ncnnMatToCvMat(out);// Convert ncnn::Mat to cv::Mat
// pixel_print(cvMat,out); // print 
std::tuple<ncnn::Mat,cv::Mat> image_digestion(const std::string &input_image_path);
ncnn::Mat perform_inference(ncnn::Net &net, ncnn::Mat &in);
cv::Mat preprocessing(ncnn::Mat out, cv::Mat img);
void image_inference(const std::string &input_image_path,ncnn::Net &net, const std::string &output_image_path);

std::tuple<cv::VideoCapture, double, int, int> video_digestion(const std::string &input_video_path);
void video_inference(const std::string &input_video_path,ncnn::Net &net,const std::string &output_video_path);

int main(int argc, char** argv)
{
    
    const char *model_param = "../models/convnext_tiny.ncnn.param";
    const char *model_bin = "../models/convnext_tiny.ncnn.bin";
    
    const char *input_image_path = "../input_1080.jpg";
    const char *output_image_path = "../output__1080_23_8.jpg";
    // const char *input_video_path = "../1335551007851158_0.mp4";
    // const char *output_video_path = "../1335551007851158_0_out.mp4";
    const char *input_video_path = "../1335551007851158_0_2.mp4";
    const char *output_video_path = "../1335551007851158_0_2_out.mp4";     
    
    std::cerr << "Hello world: " << std::endl;
    
    // Load the ncnn model
    ncnn::Net net;
    int ret = net.load_param(model_param);
    if (ret) std::cerr << "Failed to load model parameters" << std::endl; 
    ret = net.load_model(model_bin);
    if (ret) std::cerr << "Failed to load model weights" << std::endl;

    // image inference procedure
    image_inference(input_image_path, net, output_image_path);

    // video inference procedure
    // video_inference(input_video_path,net,output_video_path);
    
    return 0;
}


void video_inference(const std::string &input_video_path,ncnn::Net &net,const std::string &output_video_path){
    /*
    // // video digestion 
    // auto [cap, fps, frameWidth, frameHeight] = video_digestion(input_video_path);
    // // Define the codec and create VideoWriter object
    // cv::VideoWriter videoWriter;
    // videoWriter.open(output_video_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight));    
    
    // cv::Mat frame;
    // while (true) {
    //     // Read a new frame
    //     bool success = cap.read(frame);
    //     if (!success) {
    //         std::cout << "Finished reading video file." << std::endl;
    //         break;
    //     }
    //     // Process the frame (e.g., convert color space)
    //     cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);        
    //     ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);  
    //     const float scal[] = {0.003915, 0.003915, 0.003915};
    //     in.substract_mean_normalize(0, scal); // 0-255  -->  0-1

    //     // inference        
    //     ncnn::Mat out;
    //     out = perform_inference(net,in);
        
    //     // Postprocess
    //     cv::Mat processedFrame = preprocessing(out,frame);
        
    //     // Write the processed frame to the output video
    //     videoWriter.write(processedFrame);       
    //     }
    // // Release the video capture and writer objects
    // cap.release();
    // videoWriter.release();
    // std::cout << "Video saved successfully to " << output_video_path << std::endl;

    */
    auto [cap, fps, frameWidth, frameHeight] = video_digestion(input_video_path);
    // Define the codec and create VideoWriter object
    cv::VideoWriter videoWriter;
    // videoWriter.open(output_video_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(frameWidth, frameHeight));
    videoWriter.open(output_video_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), 5, cv::Size(frameWidth, frameHeight));
    cv::Mat frame;
    auto start = std::chrono::high_resolution_clock::now();    
    while (true) {
        // Read a new frame
        bool success = cap.read(frame);
        if (!success) {
            std::cout << "Finished reading video file." << std::endl;
            break;
        }
        // Process the frame (e.g., convert color space)
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);        
        ncnn::Mat in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);  
        const float scal[] = {0.003915, 0.003915, 0.003915};
        in.substract_mean_normalize(0, scal); // 0-255  -->  0-1

        // inference        
        ncnn::Mat out;
        out = perform_inference(net,in);
        
        // Postprocess
        
        cv::Mat processedFrame = preprocessing(out,frame);      
        
        // Write the processed frame to the output video
        videoWriter.write(processedFrame);       
        }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "TOTAL TIME VIDEO PROCESSING: " << elapsed.count() << std::endl;
    // Release the video capture and writer objects
    cap.release();
    videoWriter.release();
    std::cout << "Video saved successfully to " << output_video_path << std::endl;
}

std::tuple<cv::VideoCapture, double, int, int> video_digestion(const std::string &input_video_path){
    // Open the video file
    cv::VideoCapture cap(input_video_path);    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return std::make_tuple(cv::VideoCapture(), 0.0, 0, 0);  // Return default values on failure

    }
    // Get the frame rate of the video
    double fps = cap.get(cv::CAP_PROP_FPS);    
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frames per second: " << fps << std::endl;
    std::cout << "Frames Width: " << frameWidth << std::endl;
    std::cout << "Frames Height: " << frameHeight << std::endl;
    return std::make_tuple(cap, fps, frameWidth, frameHeight);        
    }

void image_inference(const std::string &input_image_path,ncnn::Net &net, const std::string &output_image_path){
    /*
    // image injection
    ncnn::Mat in;
    cv::Mat img;
    std::tie(in, img) = image_digestion(input_image_path);
    // inference
    ncnn::Mat out;
    out = perform_inference(net,in);
    // Preprocessing
    cv::Mat enhanced_img = preprocessing(out,img);
    // Save the enhanced images
    if (!cv::imwrite(output_image_path, enhanced_img))
    {
        std::cerr << "Failed to save image: " << output_image_path << std::endl;
        // return -1;
    }
    std::cout << "Enhanced image saved to: " << output_image_path << std::endl;
    */
    // image injection
    ncnn::Mat in;
    cv::Mat img;
    std::tie(in, img) = image_digestion(input_image_path);
    // inference
    const auto &start_time = std::chrono::steady_clock::now();  
    ncnn::Mat out;    
    out = perform_inference(net,in);
    const auto &end_time = std::chrono::steady_clock::now();     
    std::chrono::duration<double> time_span = end_time - start_time;    
    std::cout << "Time inference: " << time_span.count() << std::endl;    
    
    // Postprocess
    cv::Mat enhanced_img = preprocessing(out,img);
    // Save the enhanced images
    if (!cv::imwrite(output_image_path, enhanced_img))
    {
        std::cerr << "Failed to save image: " << output_image_path << std::endl;
        // return -1;
    }
    std::cout << "Enhanced image saved to: " << output_image_path << std::endl;
}

cv::Mat preprocessing(ncnn::Mat out, cv::Mat img){
    /*
    // Convert ncnn output back to OpenCV format
    cv::Mat enhanced_img(img.rows, img.cols, CV_8UC3);
    out.to_pixels(enhanced_img.data, ncnn::Mat::PIXEL_RGB);
    enhanced_img.convertTo(enhanced_img, CV_8UC3);   
    cv::cvtColor(enhanced_img, enhanced_img, cv::COLOR_RGB2BGR);
    */
    cv::Mat enhanced_img(img.rows, img.cols, CV_8UC3);
    out.to_pixels(enhanced_img.data, ncnn::Mat::PIXEL_RGB);
    enhanced_img.convertTo(enhanced_img, CV_8UC3);   
    cv::cvtColor(enhanced_img, enhanced_img, cv::COLOR_RGB2BGR);
    return enhanced_img;
}

ncnn::Mat perform_inference(ncnn::Net &net, ncnn::Mat &in){
    /* old code
    // Perform inference
    ncnn::Extractor ex = net.create_extractor(); 
    auto start = std::chrono::high_resolution_clock::now();    
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "TOTAL TIME: " << elapsed.count() << std::endl;  
    // Denormalize the ncnn output to the range of 0-255
    const float scal2[] = {255.0f, 255.0f, 255.0f};
    out.substract_mean_normalize(0, scal2);
    */
    ncnn::Extractor ex = net.create_extractor(); 
    auto start = std::chrono::high_resolution_clock::now();    
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "TOTAL TIME: " << elapsed.count() << std::endl;
     // Denormalize the ncnn output to the range of 0-255
    const float scal2[] = {255.0f, 255.0f, 255.0f};
    out.substract_mean_normalize(0, scal2);
    return out;
}

std::tuple<ncnn::Mat,cv::Mat> image_digestion(const std::string &input_image_path){
     /*Old file without function
           // Read the input image using OpenCV
            cv::Mat img = cv::imread(input_image_path, cv::IMREAD_COLOR);
            if (img.empty())
            {
                std::cerr << "Failed to read image: " << input_image_path << std::endl;
                return -1;
            }
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.cols, img.rows);   
            const float scal[] = {0.003915, 0.003915, 0.003915};
            in.substract_mean_normalize(0, scal); // 0-255  -->  0-1
    */
    
    // Read the input image using OpenCV
    cv::Mat img = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << input_image_path << std::endl;
        // return -1; - return empty Mats on failure
        return std::make_tuple(ncnn::Mat(), cv::Mat()); 
    }
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_RGB, img.cols, img.rows);  
    const float scal[] = {0.003915, 0.003915, 0.003915};
    in.substract_mean_normalize(0, scal); // 0-255  -->  0-1
    return std::make_tuple(in,img);
}

cv::Mat ncnnMatToCvMat(const ncnn::Mat& ncnnMat) {
    int width = ncnnMat.w;
    int height = ncnnMat.h;
    int channels = ncnnMat.c;

    // Determine the type of the cv::Mat
    int type = CV_32FC1; // Default to float32 single channel

    if (channels == 1) {
        type = CV_32FC1;
    } else if (channels == 3) {
        type = CV_32FC3;
    } else if (channels == 4) {
        type = CV_32FC4;
    }

    // Create cv::Mat without copying data, ncnn::Mat data is in float
    cv::Mat cvMat(height, width, type, (void*)ncnnMat.data);

    // Note: OpenCV's cv::Mat doesn't own the data buffer in this case.
    // If you want OpenCV to manage the data, you may need to copy the data:
    // cv::Mat cvMatCopy = cvMat.clone(); // Deep copy if needed

    return cvMat;
}
void pixel_print(cv::Mat &cvMat, ncnn::Mat &img){
// Print the pixel values
    int pixelCount = 0;
    for (int i = 0; i < img.h; ++i) {
        for (int j = 0; j < img.w; ++j) {
            // For a 3-channel color image
            cv::Vec3b pixel = cvMat.at<cv::Vec3b>(i, j);
            std::cout << "Pixel at (" << i << ", " << j << "): ";
            std::cout << "B=" << (int)pixel[0] << ", "; // Blue channel
            std::cout << "G=" << (int)pixel[1] << ", "; // Green channel
            std::cout << "R=" << (int)pixel[2] << std::endl; // Red channel
            pixelCount++;
            if (pixelCount == 10) {
                break;  // Stop after printing 10 pixels
            }
        }
        if (pixelCount == 10) {
            break;  // Stop after printing 10 pixels
        }
    }
}
    