#include "core/op_converter.h"


OpConverter::OpConverter(){
}

OpConverter::~OpConverter(){
}

void OpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor, int tensor_index){
    // set default dformat in onnx, nchw is common used by torch
    dlcl_tensor->set_data_format(dlxnet::TensorProto::ANY);
    dlcl_tensor->set_target_data_format(dlxnet::TensorProto::ANY4);
}
