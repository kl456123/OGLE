#ifndef OPENGL_NN_PROFILER_HOST_TRACER_UTILS_H_
#define OPENGL_NN_PROFILER_HOST_TRACER_UTILS_H_
#include "opengl/nn/profiler/traceme_recorder.h"

namespace opengl{
    namespace profiler {

        // Returns true if event was created by TraceMe::ActivityStart.
        inline bool IsStartEvent(const TraceMeRecorder::Event& event) {
            return (event.start_time != 0) && (event.end_time == 0);
        }

        // Returns true if event was created by TraceMe::ActivityEnd.
        inline bool IsEndEvent(const TraceMeRecorder::Event& event) {
            return (event.start_time == 0) && (event.end_time != 0);
        }

        // Returns true if event was created by TraceMe::Stop or MakeCompleteEvents
        // below.
        inline bool IsCompleteEvent(const TraceMeRecorder::Event& event) {
            return (event.start_time != 0) && (event.end_time != 0);
        }

        // Combine events created by TraceMe::ActivityStart and TraceMe::ActivityEnd,
        // which can be paired up by their activity_id.
        void MakeCompleteEvents(TraceMeRecorder::Events* events);
    } // namespace profiler
} // namespace opengl


#endif
