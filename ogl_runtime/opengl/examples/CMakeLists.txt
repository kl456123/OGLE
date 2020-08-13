

if(BUILD_NN)
    # all examples about nn library
    # classification task
    add_executable(label_image label_image/main.cc)
    set_target_properties(label_image PROPERTIES LINK_FLAGS -Wl,--no-as-needed)
    target_link_libraries(label_image ${GL_NN} ${OpenCV_LIBS})

    # object detection task
    add_executable(ssd_detector ssd/main.cc ssd/detector.cc ssd/common.cc)
    set_target_properties(ssd_detector PROPERTIES LINK_FLAGS -Wl,--no-as-needed)
    target_link_libraries(ssd_detector ${GL_NN} ${OpenCV_LIBS})

    # segmentation task

    # keypoint task

endif()
