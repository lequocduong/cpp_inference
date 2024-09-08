cmake_minimum_required(VERSION 3.12)
project(aibox-mobile)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)



#set(OPENCV_PATH /opt/opencv)
#include_directories(${OPENCV_PATH}/include/opencv4)
#link_directories(${OPENCV_PATH}/lib)

find_package(OpenCV REQUIRED)

set(FRAMEWORK "ncnn")

if (${FRAMEWORK} STREQUAL "ort")
  #set(ONNXRT_PATH /opt/onnxruntime)
  #include_directories(${ONNXRT_PATH}/include)
  #link_directories(${ONNXRT_PATH}/lib)
  set(ONNXRT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third) 
  #include_directories(${ONNXRT_PATH}/include/onnxruntime)
  
  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third/lib/onnxruntime/session)
  add_definitions(-DONNXRUNTIME)
elseif (${FRAMEWORK} STREQUAL "ov")
  set(OPENVINO_PATH /opt/intel/openvino/runtime)
  include_directories(${OPENVINO_PATH}/include)
  link_directories(${OPENVINO_PATH}/lib/intel64)
  add_definitions(-DOPENVINO)
elseif (${FRAMEWORK} STREQUAL "ncnn")
  #set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
  #find_package(ncnn REQUIRED)
  
  include_directories (${OpenCV_DIR}/include/
                     /home/lqd159753/C++/ncnn_Inference/ncnn/build/install/include/ncnn)
    
  link_directories (${OpenCV_DIR}/lib/
    					/home/lqd159753/C++/ncnn_Inference/ncnn/build/install/lib)  
  add_definitions(-DNCNN)
  
else()
  message(FATAL_ERROR "This FRAMEWORK is not supported.")
endif()

aux_source_directory(src SRC_LIST)
add_executable(aibox ${SRC_LIST})
target_link_libraries(
  aibox
  opencv_dnn
  opencv_imgcodecs
  opencv_core
  opencv_imgproc
  opencv_videoio
  avcodec
  swscale
  avformat
  avutil
)

if (${FRAMEWORK} STREQUAL "ort")
  target_link_libraries(aibox onnxruntime)
elseif (${FRAMEWORK} STREQUAL "ov")
  target_link_libraries(aibox openvino)
elseif (${FRAMEWORK} STREQUAL "ncnn")
  target_link_libraries(aibox ncnn)
endif()