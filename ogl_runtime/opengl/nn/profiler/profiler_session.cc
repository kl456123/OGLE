#include <memory>

#include "opengl/nn/profiler/profiler_session.h"
#include "opengl/nn/profiler/profiler_utils.h"
#include "opengl/utils/env.h"


namespace opengl{

    /*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
            const profiler::ProfilerOptions& options) {
        return std::unique_ptr<ProfilerSession>(new ProfilerSession(options));
    }

    /*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create() {
        int64 host_tracer_level = 2;
        // opengl::Status s = ReadInt64FromEnvVar("TF_PROFILER_HOST_TRACER_LEVEL", 2,
        // &host_tracer_level);
        // if (!s.ok()) {
            // LOG(WARNING) << "ProfilerSession: " << s.error_message();
        // }
        profiler::ProfilerOptions options;
        options.host_tracer_level = host_tracer_level;
        return Create(options);
    }

    Status ProfilerSession::Status() {
        return status_;
    }
    Status ProfilerSession::CollectData(StepStats* step_stats) {
        if (!status_.ok()) return status_;
        for (auto& profiler : profilers_) {
            profiler->Stop().IgnoreError();
        }

        for (auto& profiler : profilers_) {
            profiler->CollectData(step_stats).IgnoreError();
        }

        if (active_) {
            // Allow another session to start.
            profiler::ReleaseProfilerLock();
            active_ = false;
        }

        return Status::OK();
    }

    ProfilerSession::ProfilerSession(const profiler::ProfilerOptions& options)
        : active_(profiler::AcquireProfilerLock()),
        start_time_micros_(Env::Default()->NowNanos() / EnvTime::kMicrosToNanos) {
            if (!active_) {
                status_ = opengl::Status(dlxnet::error::UNAVAILABLE,
                        "Another profiler session is active.");
                return;
            }

            LOG(INFO) << "Profiler session started.";

            CreateProfilers(options, &profilers_);
            status_ = Status::OK();

            for (auto& profiler : profilers_) {
                auto start_status = profiler->Start();
                if (!start_status.ok()) {
                    LOG(WARNING) << "Encountered error while starting profiler: "
                        << start_status.ToString();
                }
            }
        }

    ProfilerSession::~ProfilerSession() {
        for (auto& profiler : profilers_) {
            profiler->Stop().IgnoreError();
        }

        if (active_) {
            // Allow another session to start.
            profiler::ReleaseProfilerLock();
        }
    }
}//namespace opengl
