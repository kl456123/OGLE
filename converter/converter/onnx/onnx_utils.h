#ifndef CONVERTER_ONNX_ONNX_UTILS_H_
#define CONVERTER_ONNX_ONNX_UTILS_H_
/*******************************
 * The Utils is to help to parse onnx proto model format
 */
#include "onnx.pb.h"
#include "dlcl.pb.h"
#include <vector>
#include <string>

void ParseAttrValueToString(const onnx::AttributeProto&,
        std::string* pieces);

typedef google::protobuf::RepeatedPtrField<onnx::AttributeProto> AttributeProtoList;
void ParseAttrListToString(const AttributeProtoList& attr_list, std::string* pieces);

void MakeTensorFromProto(const onnx::TensorProto&, dlxnet::TensorProto*);


#endif
