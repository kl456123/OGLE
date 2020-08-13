#ifndef OPENGL_UTILS_ENV_TIME_H_
#define OPENGL_UTILS_ENV_TIME_H_
/* Manage System IO and Time For Now
 *
 */
#include <cstdint>

namespace{
    using uint64=uint64_t;
}
/// \brief An interface used by the tensorflow implementation to
/// access timer related operations.
class EnvTime {
    public:
        static constexpr uint64 kMicrosToPicos = 1000ULL * 1000ULL;
        static constexpr uint64 kMicrosToNanos = 1000ULL;
        static constexpr uint64 kMillisToMicros = 1000ULL;
        static constexpr uint64 kMillisToNanos = 1000ULL * 1000ULL;
        static constexpr uint64 kSecondsToMillis = 1000ULL;
        static constexpr uint64 kSecondsToMicros = 1000ULL * 1000ULL;
        static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

        EnvTime() = default;
        virtual ~EnvTime() = default;

        /// \brief Returns a default impl suitable for the current operating
        /// system.
        ///
        /// The result of Default() belongs to this library and must never be deleted.
        static EnvTime* Default();

        /// \brief Returns the number of nano-seconds since the Unix epoch.
        virtual uint64 NowNanos() const = 0;

        /// \brief Returns the number of micro-seconds since the Unix epoch.
        virtual uint64 NowMicros() const { return NowNanos() / kMicrosToNanos; }

        /// \brief Returns the number of seconds since the Unix epoch.
        virtual uint64 NowSeconds() const { return NowNanos() / kSecondsToNanos; }
};

#endif
