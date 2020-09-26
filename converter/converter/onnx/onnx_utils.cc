#include "onnx/onnx_utils.h"
#include <glog/logging.h>
#include <sstream>



void ParseAttrValueToString(const onnx::AttributeProto& attr,
        std::string* str){
    onnx::AttributeProto::AttributeType type = attr.type();
    std::stringstream ss;
    ss<<"name: "<<attr.name()<<"; ";
    ss<<"type: "<<attr.type()<<"; ";
    switch(type){
        case onnx::AttributeProto::INTS:
            for(int i=0;i<attr.ints_size();++i){
                ss<<attr.ints(i)<<", ";
            }
            break;
        case onnx::AttributeProto::INT:
            ss<<attr.i();
            break;
        case onnx::AttributeProto::FLOAT:
            ss<<attr.f();
            break;
        case onnx::AttributeProto::FLOATS:
            for(int i=0;i<attr.ints_size();++i){
                ss<<attr.floats(i)<<", ";
            }
            break;
        default:
            LOG(WARNING)<<"unknown types: "<<type;
    }
    *str = ss.str();
}

void ParseAttrListToString(const AttributeProtoList& attr_list, std::string* pieces){
    std::string res;
    for(int i=0;i<attr_list.size();i++){
        std::string tmp;
        ParseAttrValueToString(attr_list[i], &tmp);
        res+=tmp;
        // const onnx::AttributeProto& attr = src_node_onnx->attribute(i);
        // ParseAttrValueToString(attr, &res);
        // LOG(INFO)<<res;
    }
    *pieces = res;
}

void MakeTensorFromProto(const onnx::TensorProto& onnx_tensor,
        dlxnet::TensorProto* dlcl_tensor){
    // switch data transfer according to its data type
    auto data_type = static_cast<onnx::TensorProto::DataType>(
            onnx_tensor.data_type());
    // handle tensor shape
    size_t num_elements =1;
    size_t dim_size = onnx_tensor.dims().size();

    // if(dim_size==0){
        // dlcl_tensor->add_dims(1);
    // }
    for (int i = 0; i < dim_size; ++i) {
        dlcl_tensor->add_dims(onnx_tensor.dims(i));
        num_elements  *= onnx_tensor.dims(i);
    }



    // handle tensor value
    const void* tensor_content = nullptr;
    switch(data_type){
        #define CASE_DATA_TYPE(src, dst)                            \
      case src:                                               \
          tensor_content = onnx_tensor.dst##_data().data();   \
          if (onnx_tensor.dst##_data_size() == 0) {           \
              tensor_content = onnx_tensor.raw_data().data(); \
          }                                                   \
          break
        CASE_DATA_TYPE(onnx::TensorProto_DataType_FLOAT, float);
        CASE_DATA_TYPE(onnx::TensorProto_DataType_INT64, int64);
        default:
              LOG(INFO) << "[TODO]ONNX data type to support: " << onnx_tensor.data_type();
    }

    CHECK_NOTNULL(tensor_content);
    switch(data_type){
        case onnx::TensorProto::FLOAT:
            {
                auto source = (float*)tensor_content;
                for(int i=0;i<num_elements;++i){
                    dlcl_tensor->add_float_data(source[i]);
                }
                dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
                break;
            }
        case onnx::TensorProto::INT64:
            {
                auto source = (int64_t*)tensor_content;
                for(int i=0;i<num_elements;++i){
                    dlcl_tensor->add_float_data(source[i]);
                }
                dlcl_tensor->set_data_type(dlxnet::TensorProto::FLOAT32);
                break;
            }
        default:
            LOG(FATAL)<<"unsupported data type when converting tensor "
                <<data_type;
    }

}
