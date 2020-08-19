cmake_minimum_required(VERSION 3.2.0)

project(opengl VERSION 1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++14")

option(BUILD_EXAMPLES "BUILD OPENGL EXAMPLES" ON)
option(BUILD_NN "BUILD NERTUAL NETWORK" ON)
option(BUILD_TEST "BUILD OPENGL TEST" ON)
option(BUILD_ARM "USE GLES" OFF)
option(BUILD_DEBUG "ENABLE DEBUG MODE" ON)
include(cmake/Protobuf.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

add_definitions(-DROOT_DIR="${CMAKE_BINARY_DIR}")

if(BUILD_ARM)
    add_definitions(-DARM_PLATFORM)
endif()

if(BUILD_DEBUG)
    add_definitions(-DENABLE_OPENGL_CHECK_ERROR)
endif()

find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
file(GLOB ProtoFiles opengl/**/*.proto)

CUSTOM_PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

set(CORE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/core)
set(UTIL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/utils)

set(PROFILER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/nn/profiler)
set(PROFILER_SRC_FNAMES profiler.cc traceme_recorder.cc
    host_tracer.cc host_tracer_utils.cc profiler_session.cc
    profiler_utils.cc profiler_interface.cc device_tracer.cc)

foreach(file_path ${PROFILER_SRC_FNAMES})
    set(PROFILER_SRC_FILES ${PROFILER_SRC_FILES} ${PROFILER_SRC_DIR}/${file_path})
endforeach()

# lib/monitor
set(CORE_LIB_MONITOR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opengl/core/lib/monitor)
set(CORE_LIB_MONITOR_SRC_FILES collection_registry.cc)

set(CORE_SRC_NAMES
    program.cc
    context.cc
    buffer.cc
    texture.cc
    session.cc
    fbo_session.cc
    scope.cc
    tensor_format.cc
    kernel.cc
    kernel_registry.cc
    init.cc
    driver.cc
    tensor.cc
    allocator.cc
    tensor_allocator.cc
    ogl_allocator.cc
    functor.cc
    cpu_functor.cc
    buffer_base.cc
    cpu_allocator_impl.cc
    metric.cc
    step_stats_collector.cc

    # some protos
    # dlxnet.pb.cc
    # tensor_description.pb.cc
    # step_stats.pb.cc
    )

set(UTIL_SRC_NAMES macros.cc util.cc protobuf.cc env.cc env_time.cc
    port.cc strings.cc status.cc abi.cc)

foreach(fname ${CORE_SRC_NAMES})
    set(CORE_SRC_FILES ${CORE_SRC_FILES} ${CORE_SRC_DIR}/${fname})
endforeach()

foreach(fname ${UTIL_SRC_NAMES})
    set(UTIL_SRC_FILES ${UTIL_SRC_FILES} ${UTIL_SRC_DIR}/${fname})
endforeach()

# lib/monitor
foreach(fname ${CORE_LIB_MONITOR_SRC_FILES})
    set(LIB_SRC_FILES ${LIB_SRC_FILES} ${CORE_LIB_MONITOR_SRC_DIR}/${fname})
endforeach()

# root directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(OpenGL_LIB -lglfw)
if(BUILD_ARM)
    set(OpenGL_LIB ${OpenGL_LIB} -lGLESv2)
else()
    set(OpenGL_LIB ${OpenGL_LIB} -lGL)
endif()

set(GL_CORE gl_core)

# build core library
add_library(${GL_CORE} SHARED ${CORE_SRC_FILES} ${UTIL_SRC_FILES}
    ${PROFILER_SRC_FILES} ${LIB_SRC_FILES} ${ProtoSources}
    ${CMAKE_CURRENT_SOURCE_DIR}/opengl/nn/all_shaders.cc)
# use both in case anyone of them not found
target_link_libraries(${GL_CORE} ${OpenGL_LIB} glog::glog ${Protobuf_LIBRARIES}
    ${PROTOBUF_LIBRARY} -lstdc++fs)


# deeplearning framework using opengl backend
if (BUILD_NN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/nn)
    set(GL_NN gl_nn)
endif()

# examples demo
if (BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/examples)
endif()

if(BUILD_TEST)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/opengl/test)
endif()




