#include "opengl/utils/status.h"
#include "opengl/utils/logging.h"
#include "opengl/utils/stacktrace.h"
#include "opengl/core/error_codes.pb.h"


namespace opengl{
    Status::Status(dlxnet::error::Code code, const string& msg) {
        assert(code != dlxnet::error::OK);
        state_ = std::unique_ptr<State>(new State);
        state_->code = code;
        state_->msg = string(msg);
        VLOG(5) << "Generated non-OK status: \"" << *this << "\". "
            << CurrentStackTrace();
    }

    void Status::Update(const Status& new_status) {
        if (ok()) {
            *this = new_status;
        }
    }

    void Status::SlowCopyFrom(const State* src) {
        if (src == nullptr) {
            state_ = nullptr;
        } else {
            state_ = std::unique_ptr<State>(new State(*src));
        }
    }

    const string& Status::empty_string() {
        static string* empty = new string;
        return *empty;
    }

    string error_name(dlxnet::error::Code code) {
        switch (code) {
            case dlxnet::error::OK:
                return "OK";
                break;
            case dlxnet::error::CANCELLED:
                return "Cancelled";
                break;
            case dlxnet::error::UNKNOWN:
                return "Unknown";
                break;
            case dlxnet::error::INVALID_ARGUMENT:
                return "Invalid argument";
                break;
            case dlxnet::error::DEADLINE_EXCEEDED:
                return "Deadline exceeded";
                break;
            case dlxnet::error::NOT_FOUND:
                return "Not found";
                break;
            case dlxnet::error::ALREADY_EXISTS:
                return "Already exists";
                break;
            case dlxnet::error::PERMISSION_DENIED:
                return "Permission denied";
                break;
            case dlxnet::error::UNAUTHENTICATED:
                return "Unauthenticated";
                break;
            case dlxnet::error::RESOURCE_EXHAUSTED:
                return "Resource exhausted";
                break;
            case dlxnet::error::FAILED_PRECONDITION:
                return "Failed precondition";
                break;
            case dlxnet::error::ABORTED:
                return "Aborted";
                break;
            case dlxnet::error::OUT_OF_RANGE:
                return "Out of range";
                break;
            case dlxnet::error::UNIMPLEMENTED:
                return "Unimplemented";
                break;
            case dlxnet::error::INTERNAL:
                return "Internal";
                break;
            case dlxnet::error::UNAVAILABLE:
                return "Unavailable";
                break;
            case dlxnet::error::DATA_LOSS:
                return "Data loss";
                break;
            default:
                char tmp[30];
                snprintf(tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code));
                return tmp;
                break;
        }
    }

    string Status::ToString() const {
        if (state_ == nullptr) {
            return "OK";
        } else {
            string result(error_name(code()));
            result += ": ";
            result += state_->msg;
            return result;
        }
    }

    void Status::IgnoreError() const {
        // no-op
    }

    std::ostream& operator<<(std::ostream& os, const Status& x) {
        os << x.ToString();
        return os;
    }

    string* TfCheckOpHelperOutOfLine(const ::opengl::Status& v,
            const char* msg) {
        string r("Non-OK-status: ");
        r += msg;
        r += " status: ";
        r += v.ToString();
        // Leaks string but this is only to be used in a fatal error message
        return new string(r);
    }
}//namespace opengl
