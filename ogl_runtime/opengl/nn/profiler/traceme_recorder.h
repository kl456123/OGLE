#ifndef OPENGL_NN_PROFILER_TRACEME_RECORDER_H_
#define OPENGL_NN_PROFILER_TRACEME_RECORDER_H_
#include <atomic>
#include "opengl/core/types.h"
#include "opengl/utils/macros.h"

namespace opengl{
    namespace profiler{
        namespace internal {

            // Current trace level.
            // Static atomic so TraceMeRecorder::Active can be fast and non-blocking.
            // Modified by TraceMeRecorder singleton when tracing starts/stops.
            extern std::atomic<int> g_trace_level;

        } // namespace internal

        // TraceMeRecorder is a singleton repository of TraceMe events.
        // It can be safely and cheaply appended to by multiple threads.
        //
        // Start() and Stop() must be called in pairs, Stop() returns the events added
        // since the previous Start().
        //
        // This is the backend for TraceMe instrumentation.
        // The profiler starts the recorder, the TraceMe constructor records begin
        // events, and the destructor records end events.
        // The profiler then stops the recorder and finds start/end pairs. (Unpaired
        // start/end events are discarded at that point).
        class TraceMeRecorder {
            public:
                // An Event is either the start of a TraceMe, the end of a TraceMe, or both.
                // Times are in ns since the Unix epoch.
                struct Event {
                    uint64 activity_id;
                    string name;
                    uint64 start_time;  // 0 = missing
                    uint64 end_time;    // 0 = missing
                };

                struct ThreadInfo {
                    int32 tid;
                    string name;
                };
                struct ThreadEvents {
                    ThreadInfo thread;
                    std::vector<Event> events;
                };
                using Events = std::vector<ThreadEvents>;
                // Starts recording of TraceMe().
                // Only traces <= level will be recorded.
                // Level must be >= 0. If level is 0, no traces will be recorded.
                static bool Start(int level) { return Get()->StartRecording(level); }

                // Stops recording and returns events recorded since Start().
                // Events passed to Record after Stop has started will be dropped.
                static Events Stop() { return Get()->StopRecording(); }

                // Returns whether we're currently recording. Racy, but cheap!
                static inline bool Active(int level = 1) {
                    return internal::g_trace_level.load(std::memory_order_acquire) >= level;
                }

                // Default value for trace_level_ when tracing is disabled
                static constexpr int kTracingDisabled = -1;

                // Records an event. Non-blocking.
                static void Record(Event event);

                // Returns an activity_id for TraceMe::ActivityStart.
                static uint64 NewActivityId();

            private:
                class ThreadLocalRecorder;

                // Returns singleton.
                static TraceMeRecorder* Get();

                TraceMeRecorder() = default;

                DISALLOW_COPY_AND_ASSIGN(TraceMeRecorder);

                void RegisterThread(int32 tid, ThreadLocalRecorder* thread);
                void UnregisterThread(int32 tid);

                bool StartRecording(int level);
                Events StopRecording();

                // Gathers events from all active threads, and clears their buffers.
                Events Clear();
                // Map of the static container instances (thread_local storage) for each
                // thread. While active, a ThreadLocalRecorder stores trace events.
                std::unordered_map<int32, ThreadLocalRecorder*> threads_;

                // Events from threads that died during recording.
                TraceMeRecorder::Events orphaned_events_;
        };
    }//namespace profiler
}//namespace opengl

#endif
