cmake_minimum_required(VERSION 3.12)
project("bert.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#
# Option list
#

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                  OFF)
option(GGML_PERF                   "ggml: enable performance logging"                    OFF)

# instruction set specific
option(GGML_AVX                    "ggml: enable AVX"                                     ON)
option(GGML_AVX2                   "ggml: enable AVX2"                                    ON)
option(GGML_FMA                    "ggml: enable FMA"                                     ON)

# backend specific
option(GGML_CUBLAS                 "ggml: use cuBLAS"                                    OFF)
option(GGML_METAL                  "ggml: use Metal"                                     OFF)

# emscripten specific
option(EMSCRIPTEN_WASM_ONLY        "emscipten: output wasm file only"                    ON)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (GGML_ALL_WARNINGS)
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wno-unused-function
    )
endif()

add_compile_options(-Wno-format)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# architecture specific
if (NOT EMSCRIPTEN)
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
        message(STATUS "x86 detected")
        add_compile_options(-mf16c)
        if (GGML_FMA)
            add_compile_options(-mfma)
        endif()
        if (GGML_AVX)
            add_compile_options(-mavx)
        endif()
        if (GGML_AVX2)
            add_compile_options(-mavx2)
        endif()
    else()
        message(STATUS "Unknown architecture")
    endif()
else()
    message(STATUS "Emscripten detected")
    if (EMSCRIPTEN_WASM_ONLY)
        set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s WASM=1 -s USE_PTHREADS=1 -fPIC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1 -s USE_PTHREADS=1 -fPIC")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s WASM=1 -s SIDE_MODULE=2 -s USE_PTHREADS=1 -s INITIAL_MEMORY=15MB -s TOTAL_MEMORY=15MB -s FORCE_FILESYSTEM=1 -s SINGLE_FILE=1 -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\"")
    else()
        set(CMAKE_EXECUTABLE_SUFFIX ".js")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s WASM=1 -s USE_PTHREADS=1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1 -s USE_PTHREADS=1")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s WASM=1 -s USE_PTHREADS=1 -s EXPORT_ES6=1 -s PTHREAD_POOL_SIZE_STRICT=0 -s INITIAL_MEMORY=15MB -s TOTAL_MEMORY=15MB -s FORCE_FILESYSTEM=1 -s SINGLE_FILE=1 -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\"")
    endif()
endif()

#
# backends
#

if (GGML_CUBLAS)
    add_compile_definitions(GGML_USE_CUBLAS)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)

    # copy ggml-metal.metal to bin directory
    configure_file(ggml/src/ggml-metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)
endif()

#
# build it
#

# add ggml
add_subdirectory(ggml EXCLUDE_FROM_ALL)
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS ggml LIBRARY DESTINATION bert_cpp)

# add bert
if (EMSCRIPTEN)
    set(TARGET libbert)
    add_executable(${TARGET}
        src/wasm.cpp
    )
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_sources(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/bert.cpp)
    target_link_libraries(${TARGET} PRIVATE ggml)
    target_link_libraries(${TARGET} PRIVATE embind)
else()
    add_subdirectory(src)
endif()
