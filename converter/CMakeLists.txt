cmake_minimum_required(VERSION 2.8)

project(converter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# generate protos
find_package(Protobuf REQUIRED)
find_package(glog REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(GLOB ProtoFiles ${CMAKE_CURRENT_LIST_DIR}/converter/**/*.proto)
protobuf_generate_cpp(ProtoSources ProtoHeaders ${ProtoFiles})


set(ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/converter)
set(SRC_DIR ${ROOT_PATH}/core
    ${ROOT_PATH}/onnx
    ${ROOT_PATH}/onnx/ops
    ${ROOT_PATH}/tf
    ${ROOT_PATH}/graph/
    ${ROOT_PATH}/optimizers/
    ${ROOT_PATH}
    )

set(INCLUDE_DIR ${SRC_DIR})
include_directories(${INCLUDE_DIR})

# *.c
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_c ${file_path}/*.c ${file_path}/*.h)
    set(SRC_FILES ${SRC_FILES} ${file_source_c})
endforeach()
message(STATUS "[*] searching *.c")
 
# *.cc
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_cc ${file_path}/*.cc ${file_path}/*.hpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cc})
endforeach()
message(STATUS "[*] searching *.cc")
 
# *.cpp
foreach(file_path ${SRC_DIR})
    file(GLOB file_source_cpp ${file_path}/*.cpp)
    set(SRC_FILES ${SRC_FILES} ${file_source_cpp})
endforeach()
message(STATUS "[*] searching *.cpp")


add_executable(Converter ${SRC_FILES} ${ProtoSources})


target_link_libraries(Converter ${Protobuf_LIBRARIES} 
    ${PROTOBUF_LIBRARIES} glog::glog)
