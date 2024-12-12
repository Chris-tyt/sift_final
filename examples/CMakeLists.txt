add_executable(sift_test sift_test.cpp)

target_include_directories(sift_test PRIVATE ../src)

target_link_libraries(sift_test PRIVATE
                      img
                      sift
)

# 添加 gpu.cu 的编译
# add_executable(gpu_executable cuda_test.cpp)

# 链接 CUDA 库
# target_link_libraries(gpu_executable PRIVATE gpu)

# 包含头文件路径（如需要）
# target_include_directories(gpu_executable PRIVATE ${CMAKE_SOURCE_DIR}/src)

SET_TARGET_PROPERTIES(sift_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# SET_TARGET_PROPERTIES(gpu_executable PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)