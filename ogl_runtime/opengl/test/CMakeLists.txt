

find_package(GTest REQUIRED)

# opengl core test
file(GLOB_RECURSE TEST_CORE_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/core/*_test.cc)

# nn ops test
file(GLOB_RECURSE TEST_NN_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/nn/kernels/*_test.cc)

file(GLOB_RECURSE TEST_PROFILER_SRC_FILES ${CMAKE_SOURCE_DIR}/opengl/nn/profiler/*_test.cc)

add_executable(test_main ${CMAKE_CURRENT_LIST_DIR}/test_main.cc
    ${TEST_CORE_SRC_FILES} ${TEST_NN_SRC_FILES} ${TEST_PROFILER_SRC_FILES}
    ${CMAKE_SOURCE_DIR}/opengl/nn/kernels/kernel_test_utils.cc)
set_target_properties(test_main PROPERTIES LINK_FLAGS -Wl,--no-as-needed)
target_link_libraries(test_main ${GL_NN} ${GTEST_LIBRARIES})
