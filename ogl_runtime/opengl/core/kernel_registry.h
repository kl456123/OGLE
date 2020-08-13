#ifndef OPENGL_CORE_KERNEL_REGISTRY_H_
#define OPENGL_CORE_KERNEL_REGISTRY_H_

#include "opengl/core/types.h"

namespace opengl{
    class KernelRegistry{
        public:
            void RegisterKernel(std::string kernel_name,
                    KernelFactory kernel_factory);
            void CreateKernel(std::string kernel_name, Kernel** kernel,
                    Context* context);

            static KernelRegistry* Global();
        private:
            KernelRegistry();
            mutable KernelMap kernel_map_;
    };

    namespace registry{
        template<typename T>
            class KernelRegisterHelper{
                public:
                    KernelRegisterHelper(const char* kernel_name){
                        KernelRegistry::Global()->RegisterKernel(std::string(kernel_name),
                                [](Kernel** kernel, Context* context){
                                *kernel = new T(context);
                                });
                    }
            };

    }//namespace registry
}//namespace opengl


#define REGISTER_KERNEL(Class)   REGISTER_KERNEL_UNIQ_HELPER(__COUNTER__, Class, #Class)
#define REGISTER_KERNEL_UNIQ_HELPER(ctr, Class, name) REGISTER_KERNEL_UNIQ(ctr, Class, name)
#define REGISTER_KERNEL_UNIQ(ctr, Class, class_name)                                \
    static auto __register##ctr = ::opengl::registry::KernelRegisterHelper<Class>(class_name)

#define REGISTER_KERNEL_WITH_NAME(Class, class_name) \
    REGISTER_KERNEL_UNIQ_HELPER(__COUNTER__, Class, class_name)

#endif

