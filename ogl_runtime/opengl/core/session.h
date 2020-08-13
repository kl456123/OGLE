#ifndef OPENGL_CORE_SESSION_H_
#define OPENGL_CORE_SESSION_H_
#include <vector>

#include "opengl/core/types.h"
#include "opengl/core/context.h"

namespace opengl{
    class Session{
        public:

            /*! Init Session with a collection of some kernels
            */
            Session(Context* context);

            /*!
             * Just A Empty Session
             */
            Session();

            /*!
             * Create all kernels with kernel names
             */
            void LoadGraph(StringList kernel_names);

            void LoadGraph(KernelList kernels);

            void Run();


            /*! set inputs tensor to the network, if input tensor is on cpu,
             * then just copy them to device if needed
             */
            void Setup(TensorList inputs);


            /*! just like as before, get output from network, make sure outputs
             * is in cpu host memory
             */
            void GetOutputs(TensorList outputs);
        private:
            // In internal of session, use texture tensor to do computation
            TensorList texture_inputs_;
            TensorList texture_outputs_;
            Context* context_;

            KernelList kernels_;
    };
}//namespace opengl


#endif
