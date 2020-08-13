#ifndef OPENGL_CORE_KERNEL_H_
#define OPENGL_CORE_KERNEL_H_
#include <string>

#include "opengl/core/opengl.h"
#include "opengl/core/types.h"
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class Program;
    class Tensor;
    class Context;
    class FBOSession;

    class Kernel{
        public:
            Kernel(Context* context);
            virtual ~Kernel();

            virtual void SetupProgram(Program* program);
            /*!
             * Run Kernel, do computation actually
             */
            virtual void Compute(TensorList& inputs, TensorList& outputs)=0;

            virtual void SetupAttr(const dlxnet::Attribute& attr)=0;

            virtual void Setup(const dlxnet::NodeProto& node_proto,
                    FBOSession* session);

            void Compute();

            DataFormat GetOutputDFormat(int i)const;
            DataFormat GetRequestInputDFormat(int i)const;

            std::string DebugString()const;

            /*!
             * Compute output shapes according to their input tensor shape
             */
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)=0;
            // we need input tensor value to get the output shape for some ops
            // like Reshape op
            virtual void InferOutputShape(const TensorList& inputs,
                    TensorShapeList& output_shapes);

            // some accessors
            void set_kernel_name(std::string name){
                kernel_name_ = name;
            }
            void set_kernel_type(std::string name){
                kernel_type_ = name;
            }
            std::string kernel_name()const{
                return kernel_name_;
            }
            std::string kernel_type()const{
                return kernel_type_;
            }
            void set_session(FBOSession* session){
                session_ = session;
            }
            FBOSession* session()const{
                return session_;
            }

            virtual bool ForceReady()const{return false;}
            const std::string kernel_fname()const{return kernel_fname_;}

            std::string build_options()const{return build_options_;}
            void set_build_options(const std::string& opt){build_options_=opt;}

            virtual void SelectKernel(const TensorList& inputs){};
        protected:
            // attach output tensor to the target(fbo)
            // used in compute function of subclass
            // TODO(breakpoint) remove it
            void SetFrameBuffer(const TensorList& outputs);

            // TODO(breakpoint) remove it
            void SetVertexShader(){}

            // kernel program(opencl) or shader(opengl)
            // not own
            Program* program_;

            // opengl driver, it wrapping all API about platform(opengl or opencl)
            // not owned
            Context* context_;

            // filename of kernel source file
            std::string kernel_fname_;

            // global works size and local work size
            unsigned long work_sizes_[3];

            // store input and output indexes
            // not owned
            std::vector<Tensor*> input_tensors_;
            std::vector<Tensor*> output_tensors_;

            std::vector<int> input_tensor_indexes_;
            std::vector<int> output_tensor_indexes_;

            // set internal dformats for each kernel, it cannot be changed due to
            // imutable attributes for allowed input dformats and possible output dformats
            // if input tensors' dformat is not consistent with requested dformats, dformats
            // conversion will happened
            std::vector<DataFormat> request_tensor_dformats_;
            std::vector<DataFormat> output_tensor_dformats_;

            // make it can fill input and output tensors
            friend class FBOSession;

            std::string kernel_name_;
            std::string kernel_type_;

            std::string build_options_;

            // reference of current activated session
            FBOSession* session_=nullptr;
    };
}//namespace opengl


#endif
