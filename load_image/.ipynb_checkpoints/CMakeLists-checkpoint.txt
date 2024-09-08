cmake_minimum_required(VERSION 2.8)
project( load_image )
set(CMAKE_PREFIX_PATH /home/lqd159753/cpp_package/torch/libtorch/)
set(CMAKE_BUILD_TYPE Release)

set(OPENCV_EXTRA_MODULES_PATH /home/lqd159753/cpp_package/opencv/opencv_contrib-4.x/modules)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenVINO REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# If the package has been found, several variables will be set, 
# you can find the full list with descriptions 
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status: ")
message(STATUS "config: ${OpenCV_DIR}")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(load_image src/load_image.cpp )
target_link_libraries(load_image PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES} openvino::runtime)
set_property(TARGET load_image PROPERTY CXX_STANDARD 17)

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