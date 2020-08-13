#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Transpose);

void TransposeOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for transpose due to the only one input tensor";
}

void TransposeOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::TransposeAttribute* dst_attr = dst_node->mutable_attr()->mutable_transpose_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="perm"){
            for(auto item: attr.ints()){
                dst_attr->add_perm(item);
            }
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(TransposeOpConverter, "Transpose");
