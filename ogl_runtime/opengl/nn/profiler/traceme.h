#include "opengl/nn/profiler/traceme_recorder.h"
#include "opengl/utils/env_time.h"
#include "opengl/utils/macros.h"
#include "opengl/core/opengl.h"


namespace opengl{
    namespace profiler{
        // Predefined levels:
        // - Level 1 (kCritical) is the default and used only for user instrumentation.
        // - Level 2 (kInfo) is used by profiler for instrumenting high level program
        //   execution details (expensive TF ops, XLA ops, etc).
        // - Level 3 (kVerbose) is also used by profiler to instrument more verbose
        //   (low-level) program execution details (cheap TF ops, etc).
        enum TraceMeLevel {
            kCritical = 1,
            kInfo = 2,
            kVerbose = 3,
        };

        // This class permits user-specified (CPU) tracing activities. A trace activity
        // is started when an object of this class is created and stopped when the
        // object is destroyed.
        //
        // CPU tracing can be useful when trying to understand what parts of GPU
        // computation (e.g., kernels and memcpy) correspond to higher level activities
        // in the overall program. For instance, a collection of kernels maybe
        // performing one "step" of a program that is better visualized together than
        // interspersed with kernels from other "steps". Therefore, a TraceMe object
        // can be created at each "step".
        //
        // Two APIs are provided:
        //   (1) Scoped object: a TraceMe object starts tracing on construction, and
        //       stops tracing when it goes out of scope.
        //          {
        //            TraceMe trace("step");
        //            ... do some work ...
        //          }
        //       TraceMe objects can be members of a class, or allocated on the heap.
        //   (2) Static methods: ActivityStart and ActivityEnd may be called in pairs.
        //          auto id = ActivityStart("step");
        //          ... do some work ...
        //          ActivityEnd(id);
        class TraceMe {
            public:
                // string&& constructor to prevent an unnecessary string copy, e.g. when a
                // TraceMe is constructed based on the result of a StrCat operation.
                // Note: We can't take the string by value because a) it would make the
                // overloads ambiguous, and b) we want lvalue strings to use the string_view
                // constructor so we avoid copying them when tracing is disabled.
                explicit TraceMe(string &&activity_name, int level = 1, bool sync=false)
                    :sync_(sync){
                        DCHECK_GE(level, 1);
                        if (TraceMeRecorder::Active(level)) {
                            new (&no_init_.name) string(std::move(activity_name));
                            SyncIfNeeded(sync_);
                            start_time_ = EnvTime::Default()->NowNanos();
                        }
                    }

                static void SyncIfNeeded(bool sync){
                    if(sync){
                        OPENGL_CALL(glFinish());
                    }
                }

                // Stop tracing the activity. Called by the destructor, but exposed to allow
                // stopping tracing before the object goes out of scope. Only has an effect
                // the first time it is called.
                void Stop() {
                    // We do not need to check the trace level again here.
                    // - If tracing wasn't active to start with, we have kUntracedActivity.
                    // - If tracing was active and was stopped, we have
                    //   TraceMeRecorder::Active().
                    // - If tracing was active and was restarted at a lower level, we may
                    //   spuriously record the event. This is extremely rare, and acceptable as
                    //   event will be discarded when its start timestamp fall outside of the
                    //   start/stop session timestamp.
                    if (PREDICT_FALSE(start_time_ != kUntracedActivity)) {
                        if (PREDICT_TRUE(TraceMeRecorder::Active())) {
                            SyncIfNeeded(sync_);
                            TraceMeRecorder::Record({kCompleteActivity, std::move(no_init_.name),
                                    start_time_, EnvTime::Default()->NowNanos()});
                        }
                        no_init_.name.~string();
                        start_time_ = kUntracedActivity;
                    }
                }

                ~TraceMe() { Stop(); }

                // Static API, for use when scoped objects are inconvenient.

                // Record the start time of an activity.
                // Returns the activity ID, which is used to stop the activity.
                static uint64 ActivityStart(const string& name, int level = 1, bool sync=false) {
                    if (PREDICT_FALSE(TraceMeRecorder::Active(level))) {
                        uint64 activity_id = TraceMeRecorder::NewActivityId();
                        SyncIfNeeded(sync);
                        TraceMeRecorder::Record({activity_id, string(name),
                                /*start_time=*/EnvTime::Default()->NowNanos(),
                                /*end_time=*/0});
                        return activity_id;
                    }
                    return kUntracedActivity;
                }

                // Record the end time of an activity started by ActivityStart().
                static void ActivityEnd(uint64 activity_id, bool sync=false) {
                    // We don't check the level again (see TraceMe::Stop()).
                    if (PREDICT_FALSE(activity_id != kUntracedActivity)) {
                        if (PREDICT_TRUE(TraceMeRecorder::Active())) {
                            SyncIfNeeded(sync);
                            TraceMeRecorder::Record({activity_id, /*name=*/"", /*start_time=*/0,
                                    /*end_time=*/EnvTime::Default()->NowNanos()});
                        }
                    }
                }

                static bool Active(int level = 1) {
                    return TraceMeRecorder::Active(level);
                }

            private:
                // Activity ID or start time used when tracing is disabled.
                constexpr static uint64 kUntracedActivity = 0;
                // Activity ID used as a placeholder when both start and end are present.
                constexpr static uint64 kCompleteActivity = 1;

                DISALLOW_COPY_AND_ASSIGN(TraceMe);

                // Wrap the name into a union so that we can avoid the cost of string
                // initialization when tracing is disabled.
                union NoInit {
                    NoInit() {}
                    ~NoInit() {}
                    string name;
                } no_init_;

                uint64 start_time_ = kUntracedActivity;
                const bool sync_;
        };
    }
}
