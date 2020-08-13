#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Binary);

void BinaryOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for binary due to the only one input tensor";
}

void BinaryOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::BinaryAttribute* dst_attr = dst_node->mutable_attr()->mutable_binary_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        ParseAttrValueToString(attr, &res);
        LOG(INFO)<<res;
    }
}


REGISTER_OP_WITH_NAME(BinaryOpConverter, "Add");
