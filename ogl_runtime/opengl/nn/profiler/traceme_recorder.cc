#include "opengl/nn/profiler/traceme_recorder.h"
#include "opengl/utils/env.h"

#include <queue>

namespace opengl{
    namespace profiler{
        namespace internal {
            std::atomic<int> g_trace_level(TraceMeRecorder::kTracingDisabled);
        }//namespace internal

        namespace{
            class EventQueue{
            };

            std::vector<TraceMeRecorder::Event> PopAll(
                    std::queue<TraceMeRecorder::Event>* queue) {
                // Read index before contents.
                std::vector<TraceMeRecorder::Event> result;
                while (!queue->empty ()) {
                    result.emplace_back(queue->front());
                    queue->pop();
                }
                return result;
            }
        }


        // To avoid unnecessary synchronization between threads, each thread has a
        // ThreadLocalRecorder that independently records its events.
        class TraceMeRecorder::ThreadLocalRecorder {
            public:
                // The recorder is created the first time TraceMeRecorder::Record() is called
                // on a thread.
                ThreadLocalRecorder() {
                    auto* env = Env::Default();
                    info_.tid = env->GetCurrentThreadId();
                    env->GetCurrentThreadName(&info_.name);
                    TraceMeRecorder::Get()->RegisterThread(info_.tid, this);
                }

                // The destructor is called when the thread shuts down early.
                ~ThreadLocalRecorder() {
                    // Unregister the thread. Clear() will be called from TraceMeRecorder.
                    TraceMeRecorder::Get()->UnregisterThread(info_.tid);
                }

                // Record is only called from the owner thread.
                void Record(TraceMeRecorder::Event&& event) { queue_.push(std::move(event)); }

                // Clear is called from the control thread when tracing starts/stops, or from
                // the owner thread when it shuts down (see destructor).
                TraceMeRecorder::ThreadEvents Clear() { return {info_, PopAll(&queue_)}; }

            private:
                TraceMeRecorder::ThreadInfo info_;
                std::queue<TraceMeRecorder::Event> queue_;
        };

        /*static*/ TraceMeRecorder* TraceMeRecorder::Get() {
            static TraceMeRecorder* singleton = new TraceMeRecorder;
            return singleton;
        }

        bool TraceMeRecorder::StartRecording(int level) {
            level = std::max(0, level);
            // Change trace_level_ while holding mutex_.
            int expected = kTracingDisabled;
            bool started = internal::g_trace_level.compare_exchange_strong(
                    expected, level, std::memory_order_acq_rel);
            if (started) {
                // We may have old events in buffers because Record() raced with Stop().
                Clear();
            }
            return started;
        }

        void TraceMeRecorder::RegisterThread(int32 tid, ThreadLocalRecorder* thread) {
            threads_.emplace(tid, thread);
        }

        void TraceMeRecorder::UnregisterThread(int32 tid) {
            auto it = threads_.find(tid);
            if (it != threads_.end()) {
                auto events = it->second->Clear();
                if (!events.events.empty()) {
                    orphaned_events_.push_back(std::move(events));
                }
                threads_.erase(it);
            }
        }

        TraceMeRecorder::Events TraceMeRecorder::StopRecording() {
            TraceMeRecorder::Events events;
            // Change trace_level_ while holding mutex_.
            if (internal::g_trace_level.exchange(
                        kTracingDisabled, std::memory_order_acq_rel) != kTracingDisabled) {
                events = Clear();
            }
            return events;
        }

        // This method is performance critical and should be kept fast. It is called
        // when tracing starts/stops. The mutex is held, so no threads can be
        // registered/unregistered. This prevents calling ThreadLocalRecorder::Clear
        // from two different threads.
        TraceMeRecorder::Events TraceMeRecorder::Clear() {
            TraceMeRecorder::Events result;
            std::swap(orphaned_events_, result);
            for (const auto& entry : threads_) {
                auto* recorder = entry.second;
                TraceMeRecorder::ThreadEvents events = recorder->Clear();
                if (!events.events.empty()) {
                    result.push_back(std::move(events));
                }
            }
            return result;
        }

        void TraceMeRecorder::Record(Event event) {
            static thread_local ThreadLocalRecorder thread_local_recorder;
            thread_local_recorder.Record(std::move(event));
        }

        /*static*/ uint64 TraceMeRecorder::NewActivityId() {
            // Activity IDs: To avoid contention over a counter, the top 32 bits identify
            // the originating thread, the bottom 32 bits name the event within a thread.
            // IDs may be reused after 4 billion events on one thread, or 4 billion
            // threads.
            static std::atomic<uint32> thread_counter(1);  // avoid kUntracedActivity
            const thread_local static uint32 thread_id =
                thread_counter.fetch_add(1, std::memory_order_relaxed);
            thread_local static uint32 per_thread_activity_id = 0;
            return static_cast<uint64>(thread_id) << 32 | per_thread_activity_id++;
        }
    }//namespace profiler
}//namespace opengl
