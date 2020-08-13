#include "core/op_converter.h"
#include "onnx.pb.h"


DECLARE_OP_CONVERTER(Relu);

void ReluOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){}

void ReluOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::ReluAttribute* dst_attr = dst_node->mutable_attr()->mutable_relu_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    // set default value first
    dst_attr->set_slope(0.0);
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        // 1. alpha
        if(attr.name()=="alpha"){
            dst_attr->set_slope(attr.f());
        }
    }
}


REGISTER_OP_WITH_NAME(ReluOpConverter, "Relu");

