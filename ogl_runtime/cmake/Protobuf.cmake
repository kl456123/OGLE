# Example:
#
# .. code-block:: cmake
#
#   find_package(Protobuf REQUIRED)
#   include_directories(${Protobuf_INCLUDE_DIRS})
#   include_directories(${CMAKE_CURRENT_BINARY_DIR})
#   protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS foo.proto)
#   protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS EXPORT_MACRO DLL_EXPORT foo.proto)
#   protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS DESCRIPTORS PROTO_DESCS foo.proto)
#   protobuf_generate_python(PROTO_PY foo.proto)
#   add_executable(bar bar.cc ${PROTO_SRCS} ${PROTO_HDRS})
#   target_link_libraries(bar ${Protobuf_LIBRARIES})

function(CUSTOM_PROTOBUF_GENERATE_CPP SRCS HDRS)
  cmake_parse_arguments(protobuf "" "EXPORT_MACRO;DESCRIPTORS" "" ${ARGN})

  set(PROTO_FILES "${protobuf_UNPARSED_ARGUMENTS}")
  if(NOT PROTO_FILES)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()
  foreach(FIL ${PROTO_FILES})
    FILE(RELATIVE_PATH RELA_FIL ${CMAKE_CURRENT_SOURCE_DIR} ${FIL})
    set(_proto_files ${_proto_files} ${RELA_FIL})
  endforeach()
  set(PROTO_FILES ${_proto_files})

  if(protobuf_EXPORT_MACRO)
    set(DLL_EXPORT_DECL "dllexport_decl=${protobuf_EXPORT_MACRO}:")
  endif()

  set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  if (protobuf_DESCRIPTORS)
    set(${protobuf_DESCRIPTORS})
  endif()

  foreach(FIL ${PROTO_FILES})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${FIL} DIRECTORY)
    if(FIL_DIR)
        set(FIL_WE "${FIL_DIR}/${FIL_WE}")
    endif()

    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    if(protobuf_DESCRIPTORS)
      set(_protobuf_protoc_desc "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.desc")
      set(_protobuf_protoc_flags "--descriptor_set_out=${_protobuf_protoc_desc}")
      list(APPEND ${protobuf_DESCRIPTORS} "${_protobuf_protoc_desc}")
    else()
      set(_protobuf_protoc_desc "")
      set(_protobuf_protoc_flags "")
    endif()
    set(PROTOBUF_PROTOC_EXECUTABLE /usr/local/bin/protoc)

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
             "${_protobuf_protoc_hdr}"
             ${_protobuf_protoc_desc}
             COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
               "--cpp_out=${DLL_EXPORT_DECL}${CMAKE_CURRENT_BINARY_DIR}"
               ${_protobuf_protoc_flags}
               ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
  set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
  if(protobuf_DESCRIPTORS)
    set(${protobuf_DESCRIPTORS} "${${protobuf_DESCRIPTORS}}" PARENT_SCOPE)
  endif()
endfunction()
