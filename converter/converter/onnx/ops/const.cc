#include "core/op_converter.h"
#include "onnx/onnx_utils.h"
#include "onnx.pb.h"

DECLARE_OP_CONVERTER(Const);


void  ConstOpConverter::SetTensorInfo(dlxnet::TensorProto* dlcl_tensor,
        int tensor_index){
    OpConverter::SetTensorInfo(dlcl_tensor, tensor_index);
}

void ConstOpConverter::Run(dlxnet::NodeProto* dst_node, const void* src_node){
    dlxnet::ConstAttribute* dst_attr = dst_node->mutable_attr()->mutable_const_attr();
    const auto src_node_onnx = reinterpret_cast<const onnx::NodeProto*>(src_node);
    std::string res;
    for(int i=0;i<src_node_onnx->attribute_size();i++){
        const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        if(attr.name()=="value"){
            dlxnet::TensorProto* tensor = dst_attr->mutable_value();
            MakeTensorFromProto(attr.t(), tensor);

            // set tensor info here for constant node
            SetTensorInfo(tensor, 0);
        }else{
            ParseAttrValueToString(attr, &res);
            LOG(INFO)<<res;
        }
    }
}


REGISTER_OP_WITH_NAME(ConstOpConverter, "Constant");
