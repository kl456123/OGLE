#ifndef OPENGL_NN_PROFILER_PROFILER_SESSTION_H_
#define OPENGL_NN_PROFILER_PROFILER_SESSTION_H_
#include <memory>

#include "opengl/utils/macros.h"
#include "opengl/utils/status.h"
#include "opengl/nn/profiler/profiler_interface.h"

namespace opengl{
    class StepStats;
    // A profiler which will start profiling when creating the object and will stop
    // when either the object is destroyed or SerializedToString is called. It will
    // profile all operations run under the given EagerContext.
    // Multiple instances of it can be created, but at most one of them will profile
    // for each EagerContext. Status() will return OK only for the instance that is
    // profiling.
    // Thread-safety: ProfilerSession is thread-safe.
    class ProfilerSession {
        public:
            // Creates and ProfilerSession and starts profiling.
            static std::unique_ptr<ProfilerSession> Create(
                    const profiler::ProfilerOptions& options);
            static std::unique_ptr<ProfilerSession> Create();

            // Deletes an exsiting Profiler and enables starting a new one.
            ~ProfilerSession();

            opengl::Status Status();

            opengl::Status CollectData(StepStats* step_stats);
            opengl::Status SerializeToString(string* content);

        private:
            // Constructs an instance of the class and starts profiling
            explicit ProfilerSession(const profiler::ProfilerOptions& options);

            // ProfilerSession is neither copyable or movable.
            DISALLOW_COPY_AND_ASSIGN(ProfilerSession);

            std::vector<std::unique_ptr<profiler::ProfilerInterface>> profilers_;

            // True if the session is active.
            bool active_;

            opengl::Status status_;
            const uint64 start_time_micros_;
    };
} // namespace opengl


#endif

