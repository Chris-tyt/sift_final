add_library(stb_image STATIC
            stb_image.h
            stb_image_write.h
)

set_target_properties(stb_image PROPERTIES LINKER_LANGUAGE CXX)

add_library(img STATIC
            image.hpp
            image.cpp
)

target_link_libraries(img PRIVATE stb_image)

add_library(sift STATIC
            sift.cpp
            sift.hpp
)

target_link_libraries(sift PRIVATE
                      img
                      OpenMP::OpenMP_CXX
)

add_library(gpu STATIC
            gpu.cu
)

target_link_libraries(gpu PRIVATE
                      cuda
                      cudart
)

# 设置 CUDA 架构（根据您的 GPU 调整）
set_target_properties(gpu PROPERTIES
    CUDA_ARCHITECTURES "52;60;70;75;86" # 替换为适合您硬件的架构
    CUDA_SEPARABLE_COMPILATION ON      # 启用分离编译
)

# # 添加库依赖（如有）
# target_link_libraries(gpu
#     OpenMP::OpenMP_CXX
#     ${CUDA_LIBRARIES}
# )

# 包含头文件路径
target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})