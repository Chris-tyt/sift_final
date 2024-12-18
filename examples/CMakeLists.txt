add_executable(sift_test sift_test.cpp)

target_include_directories(sift_test PRIVATE ../src)

target_link_libraries(sift_test PRIVATE
                      img
                      sift
)

SET_TARGET_PROPERTIES(sift_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
