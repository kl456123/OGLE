#!/bin/bash

PROTOC=protoc
ROOT_DIR=${PWD}
CPP_OUT=${PWD}

PROTOS_PATH=`find ${ROOT_DIR}/opengl -name '*.proto'`

# echo ${PROTOS_PATH}
echo "--- compile proto files ---"
${PROTOC} --cpp_out=${CPP_OUT} \
    -I${CPP_OUT} \
    ${PROTOS_PATH}
