#ifndef TENSOR_H_
#define TENSOR_H_
/*      Core Data Class in All DeepLearning Framework
 * class Tensor includes tensor shape and its buffer storing real data
 * and contains some informations like dtype, dformat, mem_type and etc.
 *      As for user, all should to know is that data should only be initilized
 * in host memory, if you want to initialize data whatever you want in device,
 * you must copy data from host to memory by youself, Initialize data in device
 * is not allowed.
 */
#include <vector>

#include "opengl/core/types.h"
#include "opengl/core/texture.h"
#include "opengl/core/buffer.h"
#include "opengl/utils/macros.h"
#include "opengl/core/dlxnet.pb.h"
#include "opengl/core/driver.h"
#include "opengl/core/ogl_allocator.h"
#include <glog/logging.h>

namespace dlxnet{
    class TensorDescription;
}
namespace opengl{

    using dlxnet::TensorDescription;
    //TODO(breakpoint) put all tensor attributes in a struct
    // enum Tensor::DataType;
    // enum Tensor::MemoryType;

    // struct TensorAttributes{
    // Tensor::DataType dtype;
    // Tensor::MemoryType mem_type;
    // Tensor::DataFormat dformat;
    // };

    class TensorShape{
        public:
            TensorShape(const std::vector<int>& dims)
                :dims_(dims){}
            TensorShape(){}
            size_t num_elements()const{
                size_t size = 1;
                for(auto dim:dims_){
                    size*=dim;
                }
                return size;
            }

            const IntList& dims()const{return dims_;}
            void add_dim(int dim){
                dims_.emplace_back(dim);
            }
            void insert_dim(int i, int dim){
                // insanity check
                dims_.insert(dims_.begin()+i, dim);
            }

            const int dims_size()const{
                return dims_.size();
            }

            const int operator[](int i)const{
                return dims_[i];
            }
        private:
            std::vector<int> dims_;
    };

    // device and host sperate storage
    class Tensor{

        public:
            enum DataType{
                DT_INT=0,
                DT_FLOAT=1,
                DT_DOUBLE=2,
                DT_INVALID
            };

            enum MemoryType{
                HOST_MEMORY=0,
                DEVICE_TEXTURE,
                DEVICE_BUFFER
            };

            // initialize tensor with shape and type
            // other than content value
            Tensor(DataType dtype, IntList shape, MemoryType mem_type=HOST_MEMORY,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);

            // make tensor from cpu host memory, data is defined by user.
            template<typename T>
                Tensor(DataType dtype, IntList shape, T* data,
                        DataFormat dformat=dlxnet::TensorProto::NHWC);

            // make tensor from proto, common used when
            // loading graph proto model
            Tensor(const dlxnet::TensorProto& tensor_proto);

            Tensor(Allocator* a, DataType dtype, IntList shape,
                    DataFormat dformat);

            Tensor(const Tensor* tensor);

            // to help test or debug, no need to allocate data by user
            // some helper funcs zeros, ones, empty, random
            // note that callee dont own it.
            static Tensor* Random(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Empty(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Ones(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            static Tensor* Zeros(DataType dtype, IntList shape,
                    DataFormat dformat=dlxnet::TensorProto::NHWC);
            ~Tensor();

            // get data pointer from device or cpu host
            // note that we just use one of them instead of both
            void* device()const{return device_;}
            void* host()const{return host_;}

            // typed pointer
            template<typename T>
                T* device()const{
                    return reinterpret_cast<T*>(device_);
                }
            template<typename T>
                T* host()const{
                    return reinterpret_cast<T*>(host_);
                }
            template<typename T>
                T* host(){
                    return reinterpret_cast<T*>(host_);
                }



            // accessor
            template<typename T>
                GLuint device_id(){
                    return reinterpret_cast<T*>(device_)->id();
                }
            const IntList& shape()const{return shape_.dims();}
            size_t num_elements()const{return shape_.num_elements();}
            const DataType dtype()const{return dtype_;}
            MemoryType mem_type()const{
                return mem_type_;
            }
            const int dims_size()const{return shape_.dims_size();}
            const DataFormat dformat()const{return dformat_;}
            void set_host(void* data){
                host_ = data;
            }

            void set_dformat(DataFormat dformat){
                dformat_ = dformat;
            }

            // helper functions
            // TODO(breakpoint) change it member function name
            bool is_host()const{return host_==nullptr? false: true;}
            const bool Initialized()const{
                return initialized_;
            }

            const int channel()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    return shape_[dims_size-1];
                }
                if(dims_size>=3){
                    return shape_[dims_size-3];
                }
                return 1;
            }
            const int width()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    if(dims_size>=2){
                        return shape_[shape_.dims_size()-2];
                    }
                    return 1;
                }
                return shape_[dims_size-1];
            }

            const int height()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dformat_==dlxnet::TensorProto::NHWC
                        ||dformat_==dlxnet::TensorProto::NHWC4){
                    if(dims_size>=3){
                        return shape_[shape_.dims_size()-3];
                    }
                    return 1;
                }
                // nchw or hwn4c4
                // note that for hwn4c4, it is designed for
                // filter. due to filter is nchw(n_out,n_in, h, w)
                // and it is not changed during transformation so it fall
                // to nchw case
                if(dims_size>=2){
                    return shape_[dims_size-2];
                }
                return 1;
            }

            void CheckIsNotGeneralTensor()const{
                CHECK_NE(dformat(), dlxnet::TensorProto::ANY);
                CHECK_NE(dformat(), dlxnet::TensorProto::ANY4);
            }

            const int num()const{
                CheckIsNotGeneralTensor();
                const auto dims_size = shape_.dims_size();
                if(dims_size>=4){
                    return shape_[dims_size-4];
                }
                return 1;

            }

            const int last_stride()const{
                auto tmp = shape_.dims_size();
                return tmp==0?1:shape_[tmp-1];
            }

            std::string DebugString()const;
            std::string ShortDebugString()const;

            void AsProto(dlxnet::TensorProto* proto)const;

            void CheckShapeAndDFormat(){
                // self-check
                if(shape_.dims_size()!=4){
                    CHECK(dformat()==::dlxnet::TensorProto::ANY
                            ||dformat()==::dlxnet::TensorProto::ANY4);
                }
            }
            void FillDescription(TensorDescription* description)const;

            uint64 AllocatedSize()const{return allocated_size_;}
            uint64 AllocatedElements()const{return allocated_size_/sizeof(float);}
            uint64 RequestedSize() const{return requested_size_;}
            void SwapData(Tensor* other_tensor);

        private:
            // inner data pointer
            void* device_=nullptr;
            void* host_=nullptr;

            int GetLastStride()const;
            template<typename T>
                uint64 CalcRequestSize()const;

            uint64 CalcRequestSize()const;
            bool IsIntialized()const{return initialized_;}

            // help to tell if it is empty or not
            bool initialized_=false;

            uint64 allocated_size_;
            uint64 requested_size_;

            // common attributes for tensor, like data type, shape and mem type
            DataType dtype_;
            MemoryType mem_type_;
            TensorShape shape_;
            DataFormat dformat_;

            // disallow copy and assign
            DISALLOW_COPY_AND_ASSIGN(Tensor);
    };

    template<typename T>
        uint64 Tensor::CalcRequestSize()const{
            return num_elements()*sizeof(T);
        }

    inline uint64 Tensor::CalcRequestSize()const{
        if(this->dtype()==Tensor::DT_FLOAT){
            return CalcRequestSize<float>();
        }
        return CalcRequestSize<int>();
    }
}//namespace opengl
#endif
