#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"


DECLARE_OP_CONVERTER(MaxPool);
void MaxPoolOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    LOG(FATAL)<<"Cannot set tensor info for maxpool due to the only one input tensor";
}

void MaxPoolOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::MaxPoolAttribute* dst_attr = dst_node->mutable_attr()->mutable_maxpool_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="kernel_shape"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_kernel_shape(attr.ints(i));
            }
        }else if(attr.name()=="strides"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_strides(attr.ints(i));
            }
        }else if(attr.name()=="pads"){
            for(int i=0;i<attr.ints_size();++i){
                dst_attr->add_pads(attr.ints(i));
            }
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(MaxPoolOpConverter, "MaxPool");
