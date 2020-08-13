#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Unsqueeze);

void UnsqueezeOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for unsqueeze due to the only one input tensor";
}

void  UnsqueezeOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::UnsqueezeAttribute* dst_attr = dst_node->mutable_attr()->mutable_unsqueeze_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="axes"){
            for(auto item:attr.ints()){
                dst_attr->add_axes(item);
            }
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(UnsqueezeOpConverter, "Unsqueeze");
