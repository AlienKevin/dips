#
# bert.cpp main
#

# ggml includes
include_directories(${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include/ggml)

# bert (shared) library
add_library(bert bert.cpp bert.h)
target_compile_features(bert PUBLIC cxx_std_20)
set_target_properties(bert PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH "$ORIGIN")
target_link_libraries(bert PRIVATE ggml)

# main entry
add_executable(main main.cpp)
target_link_libraries(main PRIVATE bert ggml)

# quantization
add_executable(quantize quantize.cpp)
target_link_libraries(quantize PRIVATE bert ggml)

# install
install(TARGETS bert LIBRARY DESTINATION bert_cpp)
