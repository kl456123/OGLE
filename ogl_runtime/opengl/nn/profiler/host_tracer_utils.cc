#include "opengl/nn/profiler/host_tracer_utils.h"
#include <unordered_map>
#include <vector>


namespace opengl{
    namespace profiler{
        void MakeCompleteEvents(TraceMeRecorder::Events* events) {
            // Track events created by ActivityStart and copy their data to events created
            // by ActivityEnd. TraceMe records events in its destructor, so this results
            // in complete events sorted by their end_time in the thread they ended.
            // Within the same thread, the record created by ActivityStart must appear
            // before the record created by ActivityEnd. Cross-thread events must be
            // processed in a separate pass. A single map can be used because the
            // activity_id is globally unique.
            std::unordered_map<uint64, TraceMeRecorder::Event*> start_events;
            std::vector<TraceMeRecorder::Event*> end_events;
            for (auto& thread : *events) {
                for (auto& event : thread.events) {
                    if (IsStartEvent(event)) {
                        start_events.emplace(event.activity_id, &event);
                    } else if (IsEndEvent(event)) {
                        auto iter = start_events.find(event.activity_id);
                        if (iter != start_events.end()) {  // same thread
                            auto* start_event = iter->second;
                            event.name = std::move(start_event->name);
                            event.start_time = start_event->start_time;
                            start_events.erase(iter);
                        } else {  // cross-thread
                            end_events.push_back(&event);
                        }
                    }
                }
            }

            for (auto* event : end_events) {  // cross-thread
                auto iter = start_events.find(event->activity_id);
                if (iter != start_events.end()) {
                    auto* start_event = iter->second;
                    event->name = std::move(start_event->name);
                    event->start_time = start_event->start_time;
                    start_events.erase(iter);
                }
            }
        }
    } // namespace profiler
} // namespace opengl
