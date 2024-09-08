cmake_minimum_required(VERSION 3.12)
project(test)

set(CMAKE_CXX_STANDARD 11)

set(OPENCV_PATH /opt/opencv)
include_directories(${OPENCV_PATH}/include/opencv4)
link_directories(${OPENCV_PATH}/lib)

aux_source_directory(. SRC_LIST)
add_executable(test ${SRC_LIST})
target_link_libraries(
  test
  opencv_dnn
  opencv_imgcodecs
  opencv_core
  opencv_imgproc
  opencv_videoio
  avcodec
  avformat
  avutil
)