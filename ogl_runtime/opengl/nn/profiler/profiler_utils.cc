#include "opengl/nn/profiler/profiler_utils.h"
#include <atomic>

namespace opengl{
    namespace profiler {

        // Track whether there's an active profiler session.
        // Prevents another profiler session from creating ProfilerInterface(s).
        std::atomic<bool> session_active(false);

        bool AcquireProfilerLock() { return !session_active.exchange(true); }

        void ReleaseProfilerLock() { session_active.store(false); }

    }  // namespace profiler
} // namespace opengl
