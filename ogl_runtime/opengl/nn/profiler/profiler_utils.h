#ifndef OPENGL_NN_PROFILER_PROFILER_UTILS_H_
#define OPENGL_NN_PROFILER_PROFILER_UTILS_H_

namespace opengl{
    namespace profiler {

        bool AcquireProfilerLock();

        void ReleaseProfilerLock();

    }  // namespace profiler
} // namespace opengl


#endif
