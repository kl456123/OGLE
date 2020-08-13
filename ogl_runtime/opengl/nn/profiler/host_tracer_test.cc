#include "opengl/nn/profiler/host_tracer_utils.h"
#include "opengl/nn/profiler/profiler_interface.h"
#include "opengl/nn/profiler/traceme.h"
#include "opengl/utils/env.h"
#include "opengl/utils/status_test_util.h"
#include "opengl/core/step_stats.pb.h"

namespace opengl{
    namespace profiler{
        std::unique_ptr<ProfilerInterface> CreateHostTracer(
                const ProfilerOptions& options);

        namespace {
            TEST(HostTracerTest, CollectsTraceMeEvents) {
                int32 thread_id = Env::Default()->GetCurrentThreadId();

                auto tracer = CreateHostTracer(ProfilerOptions());

                DLXNET_ASSERT_OK(tracer->Start());
                { TraceMe traceme("hello"); }
                { TraceMe traceme("world"); }
                { TraceMe traceme("contains#inside"); }
                { TraceMe traceme("good#key1=value1#"); }
                { TraceMe traceme("morning#key1=value1,key2=value2#"); }
                { TraceMe traceme("incomplete#key1=value1,key2#"); }
                DLXNET_ASSERT_OK(tracer->Stop());

                StepStats step_stats;
                DLXNET_ASSERT_OK(tracer->CollectData(&step_stats));

                EXPECT_EQ(step_stats.dev_stats_size(), 1);
                EXPECT_EQ(step_stats.dev_stats(0).node_stats_size(), 6);
            }
        }//namespace
    }//namespace profier
}//namespace opengl


