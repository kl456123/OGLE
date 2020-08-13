#include <random>
#include <cmath>
#include "opengl/core/tensor.h"
#include "opengl/core/tensor_format.h"
#include "opengl/core/tensor_description.pb.h"


namespace opengl{
    namespace{
        const int kMaxChannelSize = 4;
    }
    Tensor::~Tensor(){
        if(mem_type()==HOST_MEMORY){
            // host memory
            CHECK_EQ(dtype(), DT_FLOAT);
            if(host_){
                delete reinterpret_cast<float*>(host_);
            }
        }else{
            // in some case device_ is already freed
            if(device_){
                // device memory
                delete reinterpret_cast<Texture*>(device_);
            }
        }
    }

    template<>
        Tensor::Tensor(DataType dtype, IntList shape, float* data, DataFormat dformat)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY), dformat_(dformat){
            // AmendShape();
            CheckShapeAndDFormat();

            requested_size_ = CalcRequestSize();
            host_ = data;

            initialized_=true;
            allocated_size_ = CalcAllocatedSize1D(shape, dformat)*sizeof(float);
        }

    Tensor::Tensor(const dlxnet::TensorProto& tensor_proto)
        :dformat_(tensor_proto.data_format()), mem_type_(HOST_MEMORY){
            // get tensor shape
            for(auto& dim:tensor_proto.dims()){
                shape_.add_dim(dim);
            }
            const auto last_stride = GetLastStride();
            const uint64 size = CalcAllocatedSize1D(shape(), dformat());
            // AmendShape();
            // get tensor data
            switch(tensor_proto.data_type()){
                case dlxnet::TensorProto::FLOAT32:
                    {
                        dtype_ = DT_FLOAT;
                        requested_size_ = CalcRequestSize();
                        host_ = cpu_allocator()->AllocateRaw(32, size*sizeof(float));
                        allocated_size_= size*sizeof(float);
                        memset(host_, 0, allocated_size_);
                        // host_ = StrideAllocator::Allocate(cpu_allocator(),
                        // requested_size_, last_stride, AllocationAttributes());
                        float* target_data = static_cast<float*>(host_);
                        const int data_size = tensor_proto.float_data_size();
                        for(int i=0;i<data_size;++i){
                            target_data[i] = tensor_proto.float_data(i);
                        }
                        break;
                    }

                case dlxnet::TensorProto::INT32:
                    {
                        dtype_ = DT_INT;
                        requested_size_ = CalcRequestSize();
                        host_ = cpu_allocator()->AllocateRaw(32, size*sizeof(int));
                        allocated_size_= size*sizeof(int);
                        memset(host_, 0, allocated_size_);
                        float* target_data = static_cast<float*>(host_);
                        const int data_size = tensor_proto.int32_data_size();
                        for(int i=0;i<data_size;++i){
                            target_data[i] = tensor_proto.int32_data(i);
                        }
                        break;
                    }

                default:
                    LOG(FATAL)<<"unsupported const type: "<<tensor_proto.data_type();
            }
            initialized_=true;

            CheckShapeAndDFormat();
        }

    // here we just allocate memory in host memory in hard code
    // TODO(breakpoint) add customs allocator input to allow
    // allocate device memory
    /*static*/ Tensor* Tensor::Empty(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = new Tensor(dtype, shape, Tensor::HOST_MEMORY, dformat);
        // const int num_elements = tensor->num_elements();
        // float* image_data = new float[num_elements];
        // tensor->set_host(image_data);
        return tensor;
    }

    /*static*/ Tensor* Tensor::Random(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Zeros(dtype, shape, dformat);
        float* data = tensor->host<float>();
        const int num_elements = tensor->AllocatedElements();
        for(int i=0; i<num_elements; ++i){
            if(CheckIndexValid(i, shape, dformat)){
                data[i] = 1.0*random()/RAND_MAX;
            }
        }
        return tensor;
    }

    /*static*/ Tensor* Tensor::Zeros(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Empty(dtype, shape, dformat);
        memset(tensor->host(), 0, tensor->AllocatedSize());
        return tensor;
    }

    /*static*/ Tensor* Tensor::Ones(DataType dtype, IntList shape,
            DataFormat dformat){
        Tensor* tensor = Tensor::Zeros(dtype, shape, dformat);
        for(int i=0;i<tensor->AllocatedElements();++i){
            if(CheckIndexValid(i, shape, dformat)){
                tensor->host<float>()[i] = 1.0;
            }
        }
        return tensor;
    }

    std::string Tensor::DebugString()const{
        CHECK(is_host());
        std::stringstream ss;
        ss<<"\n";
        auto&output_shape = shape();
        return ss.str();
    }

    std::string Tensor::ShortDebugString()const{
        CHECK(is_host());
        std::stringstream ss;
        ss<<"\n";
        const int num = AllocatedElements();
        for(int i=0;i<num;++i){

            if(!CheckIndexValid(i, shape(), dformat())){continue;}
            ss<< host<float>()[i] <<", ";
            if(num-i>10&&i>10){
                ss<<"..., ";
                i = num-10;
            }
        }
        return ss.str();
    }

    void Tensor::AsProto(dlxnet::TensorProto* proto)const{
        CHECK(is_host())<<"AsProto Only used in CPU Tensoor";
        // set shape
        for(auto dim: shape()){
            proto->add_dims(dim);
        }

        // set type
        proto->set_data_type(dlxnet::TensorProto::FLOAT32);
        proto->set_target_data_format(FormatToStride4(dformat()));
        proto->set_data_format(dformat());

        // set value
        const int num_elements_value = num_elements();
        proto->clear_float_data();
        const float* host_data = host<float>();
        for(int j=0;j<num_elements_value;++j){
            proto->add_float_data(host_data[j]);
        }
    }

    void Tensor::FillDescription(TensorDescription* description)const{
        description->set_data_type(dlxnet::TensorProto::FLOAT32);
        description->set_data_format(dformat());
        for(auto dim: shape()){
            description->add_dims(dim);
        }
    }

    Tensor::Tensor(Allocator* a, DataType dtype, IntList shape,
            DataFormat dformat)
        :shape_(shape),dtype_(dtype),mem_type_(HOST_MEMORY), dformat_(dformat){
            // TODO(breakpoint) how to handle it
            size_t num_elements, bytes;
            num_elements = shape_.num_elements();

            requested_size_ = CalcRequestSize();
            allocated_size_ = CalcAllocatedSize1D(shape, dformat)*sizeof(float);

            host_ = cpu_allocator()->AllocateRaw(32, allocated_size_);

            CheckShapeAndDFormat();
            initialized_=true;
        }

    int Tensor::GetLastStride()const{
        int last_stride;
        if(IsStrideDFormat(this->dformat())){
            last_stride = this->last_stride();
        }else{
            last_stride = num_elements();
        }
        return std::min(GetMaxTextureSize()*kMaxChannelSize, last_stride);
    }

    Tensor::Tensor(DataType dtype, IntList shapes, MemoryType mem_type, DataFormat dformat)
        :shape_(shapes), dtype_(dtype),mem_type_(mem_type), dformat_(dformat){
            // AmendShape();
            CheckShapeAndDFormat();
            size_t num_elements, bytes;
            num_elements = shape_.num_elements();
            requested_size_ = CalcRequestSize();

            if(mem_type==HOST_MEMORY){
                allocated_size_ = CalcAllocatedSize1D(shapes, dformat)*sizeof(float);
                host_ = cpu_allocator()->AllocateRaw(32, allocated_size_);
            }else if(mem_type==DEVICE_BUFFER){
                LOG(FATAL);
            }else if(mem_type==DEVICE_TEXTURE){
                auto texture_shape = CalcAllocatedSize2D(shapes, dformat);
                allocated_size_ = texture_shape[1]*texture_shape[0]*4*sizeof(float);
                device_ = new Texture({texture_shape[1], texture_shape[0]}, GL_RGBA32F, GL_TEXTURE_2D, nullptr);
            }else{
                LOG(FATAL)<<"unsupported types!";
            }
            initialized_=true;
        }


    Tensor::Tensor(const Tensor* tensor)
        :dtype_(tensor->dtype()), mem_type_(HOST_MEMORY),dformat_(tensor->dformat()),
        shape_(tensor->shape()), requested_size_(tensor->RequestedSize()),
        allocated_size_(tensor->AllocatedSize()){
            host_ = StrideAllocator::Allocate(cpu_allocator(),
                    allocated_size_, allocated_size_/sizeof(float), AllocationAttributes());;
            initialized_ = true;
        }


    void Tensor::SwapData(Tensor* other_tensor){
        CHECK_EQ(mem_type(), other_tensor->mem_type());
        if(mem_type()==Tensor::HOST_MEMORY){
            void* tmp = host_;
            host_ = other_tensor->host_;
            other_tensor->host_ = tmp;
        }else{
            // swap texture
            void* tmp = device_;
            device_ = other_tensor->device_;
            other_tensor->device_ = tmp;
        }
    }



}//namespace opengl
