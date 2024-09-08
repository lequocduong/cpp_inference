cmake_minimum_required(VERSION 2.8)

project(model_Inference)

add_definitions( -std=c++11) 
add_definitions("-Wall")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# ncnn_root_dir is the path to NCNN downloaded. Please change it accordingly. 
include_directories (${OpenCV_DIR}/include/
                     /home/lqd159753/C++/ncnn_Inference/ncnn/build/install/include/ncnn)

link_directories (${OpenCV_DIR}/lib/
					/home/lqd159753/C++/ncnn_Inference/ncnn/build/install/lib)

set(OpenCV_LIBS opencv_core
		opencv_highgui
	    opencv_imgproc
 	 	opencv_imgcodecs
		opencv_videoio)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

add_executable(model_Inference model_inference.cpp video_inference.cpp) 
target_link_libraries (model_Inference ncnn ${OpenCV_LIBS})


#if(True)
#	find_package(OpenMP)
#	if (NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
#		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#		set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 	
#	endif()
#endif()