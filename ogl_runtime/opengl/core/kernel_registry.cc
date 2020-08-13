#include "opengl/core/kernel_registry.h"

namespace opengl{

    /*static*/KernelRegistry* KernelRegistry::Global(){
        static auto registry = new KernelRegistry;
        return registry;
    }

    void KernelRegistry::RegisterKernel(std::string kernel_name,
            KernelFactory kernel_factory){
        kernel_map_.insert({kernel_name, kernel_factory});
    }

    KernelRegistry::KernelRegistry(){}

    void KernelRegistry::CreateKernel(std::string kernel_name, Kernel** kernel,
            Context* context){
        auto iter = kernel_map_.find(kernel_name);
        if(iter==kernel_map_.end()){
            // LOG(ERROR)<<"Create Kernel Failed, Kernel"
            // <<kernel_name <<" is not found";
            return ;
        }
        auto kernel_factory = iter->second;
        kernel_factory(kernel, context);
    }


}//namespace opengl



