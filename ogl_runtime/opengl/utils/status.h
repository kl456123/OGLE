#ifndef OPENGL_UTILS_STATUS_H_
#define OPENGL_UTILS_STATUS_H_
#include <memory>
#include "opengl/core/opengl.h"
#include "opengl/core/types.h"
#include "opengl/core/error_codes.pb.h"


// just for simplication
namespace opengl{
    typedef GLint OGLStatus;
    /// @ingroup core
    /// Denotes success or failure of a call in dlxnet.
    class Status {
        public:
            /// Create a success status.
            Status() {}

            /// \brief Create a status with the specified error code and msg as a
            /// human-readable string containing more detailed information.
            Status(dlxnet::error::Code code, const string& msg);

            /// Copy the specified status.
            Status(const Status& s);
            Status& operator=(const Status& s);
#ifndef SWIG
            Status(Status&& s) noexcept;
            Status& operator=(Status&& s) noexcept;
#endif  // SWIG

            static Status OK() { return Status(); }

            /// Returns true iff the status indicates success.
            bool ok() const { return (state_ == NULL); }

            dlxnet::error::Code code() const {
                return ok() ? dlxnet::error::OK : state_->code;
            }

            const string& error_message() const {
                return ok() ? empty_string() : state_->msg;
            }

            bool operator==(const Status& x) const;
            bool operator!=(const Status& x) const;

            /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
            /// preserves the current status, but may augment with additional
            /// information about `new_status`.
            ///
            /// Convenient way of keeping track of the first error encountered.
            /// Instead of:
            ///   `if (overall_status.ok()) overall_status = new_status`
            /// Use:
            ///   `overall_status.Update(new_status);`
            void Update(const Status& new_status);

            /// \brief Return a string representation of this status suitable for
            /// printing. Returns the string `"OK"` for success.
            string ToString() const;

            // Ignores any errors. This method does nothing except potentially suppress
            // complaints from any tools that are checking that errors are not dropped on
            // the floor.
            void IgnoreError() const;

        private:
            static const string& empty_string();
            struct State {
                dlxnet::error::Code code;
                string msg;
            };

            // OK status has a `NULL` state_.  Otherwise, `state_` points to
            // a `State` structure containing the error code and message(s)
            std::unique_ptr<State> state_;

            void SlowCopyFrom(const State* src);
    };

    inline Status::Status(const Status& s)
        : state_((s.state_ == nullptr) ? nullptr : new State(*s.state_)) {}

    inline Status& Status::operator=(const Status& s) {
        // The following condition catches both aliasing (when this == &s),
        // and the common case where both s and *this are ok.
        if (state_ != s.state_) {
            SlowCopyFrom(s.state_.get());
        }
        return *this;
    }

#ifndef SWIG
    inline Status::Status(Status&& s) noexcept : state_(std::move(s.state_)) {}

    inline Status& Status::operator=(Status&& s) noexcept {
        if (state_ != s.state_) {
            state_ = std::move(s.state_);
        }
        return *this;
    }
#endif  // SWIG

    inline bool Status::operator==(const Status& x) const {
        return (this->state_ == x.state_) || (ToString() == x.ToString());
    }

    inline bool Status::operator!=(const Status& x) const { return !(*this == x); }

    /// @ingroup core
    std::ostream& operator<<(std::ostream& os, const Status& x);

    typedef std::function<void(const Status&)> StatusCallback;

    extern string* TfCheckOpHelperOutOfLine(
            const Status& v, const char* msg);

    inline string* TfCheckOpHelper(Status v,
            const char* msg) {
        if (v.ok()) return nullptr;
        return TfCheckOpHelperOutOfLine(v, msg);
    }

#define TF_DO_CHECK_OK(val, level)                                \
    while (auto _result = ::opengl::TfCheckOpHelper(val, #val)) \
    LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

    // DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
    // mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
    while (false && (::opengl::Status::OK() == (val))) LOG(FATAL)
#endif
}//namespace opengl

#endif
