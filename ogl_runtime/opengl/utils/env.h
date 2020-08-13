#ifndef OPENGL_UTILS_ENV_H_
#define OPENGL_UTILS_ENV_H_
#include "opengl/core/types.h"
#include "opengl/utils/env_time.h"
#include "opengl/utils/macros.h"

namespace opengl{
    /// \brief An interface used by the tensorflow implementation to
    /// access operating system functionality like the filesystem etc.
    ///
    /// Callers may wish to provide a custom Env object to get fine grain
    /// control.
    ///
    /// All Env implementations are safe for concurrent access from
    /// multiple threads without any external synchronization.
    class Env {
        public:
            Env()=default;
            virtual ~Env() = default;

            /// \brief Returns a default environment suitable for the current operating
            /// system.
            ///
            /// Sophisticated users may wish to provide their own Env
            /// implementation instead of relying on this default environment.
            ///
            /// The result of Default() belongs to this library and must never be deleted.
            static Env* Default();

            // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
            // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
            // provide a routine to get the absolute time.

            /// \brief Returns the number of nano-seconds since the Unix epoch.
            virtual uint64 NowNanos() const { return env_time_->NowNanos(); }

            /// \brief Returns the number of micro-seconds since the Unix epoch.
            virtual uint64 NowMicros() const { return env_time_->NowMicros(); }

            /// \brief Returns the number of seconds since the Unix epoch.
            virtual uint64 NowSeconds() const { return env_time_->NowSeconds(); }


            // Returns the thread id of calling thread.
            // Posix: Returns pthread id which is only guaranteed to be unique within a
            //        process.
            // Windows: Returns thread id which is unique.
            virtual int32 GetCurrentThreadId() = 0;

            // Copies current thread name to "name". Returns true if success.
            virtual bool GetCurrentThreadName(string* name) = 0;
        private:
            DISALLOW_COPY_AND_ASSIGN(Env);
            EnvTime* env_time_ = EnvTime::Default();
    };
}//namespace opengl


#endif
