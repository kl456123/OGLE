#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Clip);

void ClipOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for clip due to the only one input tensor";
}

void ClipOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::ClipAttribute* dst_attr = dst_node->mutable_attr()->mutable_clip_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="min"){
            dst_attr->set_min(attr.f());
        }else if(attr.name()=="max"){
            dst_attr->set_max(attr.f());
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(ClipOpConverter, "Clip");
