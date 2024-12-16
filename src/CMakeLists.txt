add_library(stb_image STATIC
            stb_image.h
            stb_image_write.h
)

set_target_properties(stb_image PROPERTIES LINKER_LANGUAGE CXX)

add_library(img STATIC
            image.hpp
            image.cpp
)

target_link_libraries(img PRIVATE 
    stb_image
)

add_library(cuda_image STATIC  
            image_cu.h
            image.cu
)

# 设置 CUDA 架构（根据您的 GPU 调整）
set_target_properties(cuda_image PROPERTIES
    CUDA_ARCHITECTURES "52;60;70;75;86"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(cuda_image PRIVATE
    ${CUDA_LIBRARIES}
    cuda
    cudart
    img
)

# 包含头文件路径
# target_include_directories(cuda_image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(sift STATIC
            sift.cpp
            sift.hpp
)

target_link_libraries(sift PRIVATE
    img
    OpenMP::OpenMP_CXX
    cuda_image
)