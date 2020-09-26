#include "opengl/core/kernel.h"
#include "opengl/core/program.h"
#include "opengl/core/tensor.h"
#include "opengl/utils/macros.h"
#include "opengl/core/fbo_session.h"
#include "opengl/core/driver.h"


namespace opengl{
    namespace{
        struct Vertex {
            float x, y;
        };
    }
    Kernel::Kernel(Context* context)
        :context_(context){}

    Kernel::~Kernel(){}

    void Kernel::SetupProgram(Program* program){
        // not own
        program_=program;
    }

    void Kernel::SetFrameBuffer(const TensorList& outputs){
        program_->SetRetVal(outputs);
    }

    void Kernel::Setup(const dlxnet::NodeProto& node,
            FBOSession* session){
        set_kernel_name(node.name());
        set_kernel_type(node.type());
        set_session(session);

        // fill inputs and outputs
        for(int i=0; i<node.input_index_size(); ++i){
            input_tensor_indexes_.emplace_back(node.input_index(i));
        }
        for(int i=0; i<node.output_index_size(); ++i){
            output_tensor_indexes_.emplace_back(node.output_index(i));
        }

        SetupAttr(node.attr());


    }

    std::string Kernel::DebugString()const{
        // print input shape and output shape for debug
        std::stringstream ss;
        ss<<"In kernel \n type: "<<kernel_type()
            <<" name: "<<kernel_name()<<"\n";
        // fill input and output
        TensorShapeList input_shapes;
        TensorShapeList output_shapes;
        for(int index:input_tensor_indexes_){
            auto ptr = session_->FindTensorById(index);
            if(ptr){
                input_shapes.emplace_back(ptr->shape());
            }
        }
        for(int index: output_tensor_indexes_){
            auto ptr = session_->FindTensorById(index);
            if(ptr){
                output_shapes.emplace_back(ptr->shape());
            }
        }

        for(int j=0;j<input_tensor_indexes_.size();++j){
            // for each input shape
            ss<<"(";
            for(int k=0;k<input_shapes[j].size();++k){
                ss<<input_shapes[j][k]<<" ";
            }
            ss<<")"<<"["<<input_tensor_indexes_[j]<<"]"<< ", ";
        }
        ss<<"->";
        for(int j=0;j<output_tensor_indexes_.size();++j){
            // for each output shape
            ss<<"(";
            for(int k=0;k<output_shapes[j].size();++k){
                ss<<output_shapes[j][k]<<" ";
            }
            ss<<")"<<"["<<output_tensor_indexes_[j]<<"]";
        }
        return ss.str();
    }

    DataFormat Kernel::GetOutputDFormat(int i)const{
        CHECK_LT(i, output_tensor_dformats_.size());
        CHECK_GE(i, 0);
        return output_tensor_dformats_[i];
    }

    DataFormat Kernel::GetRequestInputDFormat(int i)const{
        CHECK_LT(i, request_tensor_dformats_.size());
        CHECK_GE(i, 0);
        return request_tensor_dformats_[i];
    }

    void Kernel::InferOutputShape(const TensorList& inputs,
            TensorShapeList& output_shapes){
        TensorShapeList input_shapes;
        for(auto input_tensor: inputs){
            input_shapes.emplace_back(input_tensor->shape());
        }
        InferOutputShape(input_shapes, output_shapes);
    }

    void Kernel::Compute(){
        Compute(input_tensors_, output_tensors_);
    }


}//namespace opengl
