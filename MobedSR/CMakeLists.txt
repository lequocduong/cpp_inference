cmake_minimum_required(VERSION 3.16)
project(tf_lite_inference C CXX) # tf_lite_inference
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_PREFIX_PATH /home/lqd159753/cpp_package/torch/libtorch)
set(TENSORFLOW_SOURCE_DIR /home/lqd159753/cpp_package/tflite/tensorflow_src)
set(OPENCV_EXTRA_MODULES_PATH /home/lqd159753/cpp_package/opencv/opencv_contrib-4.x/modules)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "TENSORFLOW_SOURCE_DIR: ${TENSORFLOW_SOURCE_DIR}")
message(STATUS "OPENCV_EXTRA_MODULES_PATH: ${OPENCV_EXTRA_MODULES_PATH}")


if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

include_directories( ${OpenCV_INCLUDE_DIRS} )



add_executable(tf_lite_inference src/main.cpp )

target_link_libraries(tf_lite_inference 
						tensorflow-lite                                               
						${OpenCV_LIBS}
						${TORCH_LIBRARIES}
						)
# Include TensorFlow Lite headers
target_include_directories(tf_lite_inference
    PRIVATE
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite
    ${TENSORFLOW_SOURCE_DIR}/tensorflow/lite/delegates/gpu
)
						

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)


