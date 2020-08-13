#ifndef OPENGL_UTILS_ERRORS_H_
#define OPENGL_UTILS_ERRORS_H_
#include "opengl/core/error_codes.pb.h"
#include "opengl/utils/strings.h"
#include "opengl/utils/status.h"

namespace opengl{
    namespace errors {

        typedef ::dlxnet::error::Code Code;

        namespace internal {

            // The DECLARE_ERROR macro below only supports types that can be converted
            // into StrCat's AlphaNum. For the other types we rely on a slower path
            // through std::stringstream. To add support of a new type, it is enough to
            // make sure there is an operator<<() for it:
            //
            //   std::ostream& operator<<(std::ostream& os, const MyType& foo) {
            //     os << foo.ToString();
            //     return os;
            //   }
            // Eventually absl::strings will have native support for this and we will be
            // able to completely remove PrepareForStrCat().
            template <typename T>
            string PrepareForStrCat(const T& t) {
                std::stringstream ss;
                ss << t;
                return ss.str();
            }
            // inline const string& PrepareForStrCat(const T& a) {
                // return a;
            // }
        } // namespace internal


        // For propagating errors when calling a function.
        #define RETURN_IF_ERROR(...)                          \
        do {                                                   \
            ::opengl::Status _status = (__VA_ARGS__);        \
            if (PREDICT_FALSE(!_status.ok())) return _status; \
        } while (0)

  // Convenience functions for generating and using error status.
  // Example usage:
  //   status.Update(errors::InvalidArgument("The ", foo, " isn't right."));
  //   if (errors::IsInvalidArgument(status)) { ... }
  //   switch (status.code()) { case error::INVALID_ARGUMENT: ... }

  #define DECLARE_ERROR(FUNC, CONST)                                       \
    template <typename... Args>                                            \
    ::opengl::Status FUNC(Args... args) {                              \
      return ::opengl::Status(                                         \
          ::dlxnet::error::CONST,                                      \
          ::opengl::strings::StrCat(                                   \
              ::opengl::errors::internal::PrepareForStrCat(args)...)); \
    }                                                                      \
    inline bool Is##FUNC(const ::opengl::Status& status) {             \
      return status.code() == ::dlxnet::error::CONST;                  \
    }

  DECLARE_ERROR(Cancelled, CANCELLED)
  DECLARE_ERROR(InvalidArgument, INVALID_ARGUMENT)
  DECLARE_ERROR(NotFound, NOT_FOUND)
  DECLARE_ERROR(AlreadyExists, ALREADY_EXISTS)
  DECLARE_ERROR(ResourceExhausted, RESOURCE_EXHAUSTED)
  DECLARE_ERROR(Unavailable, UNAVAILABLE)
  DECLARE_ERROR(FailedPrecondition, FAILED_PRECONDITION)
  DECLARE_ERROR(OutOfRange, OUT_OF_RANGE)
  DECLARE_ERROR(Unimplemented, UNIMPLEMENTED)
  DECLARE_ERROR(Internal, INTERNAL)
  DECLARE_ERROR(Aborted, ABORTED)
  DECLARE_ERROR(DeadlineExceeded, DEADLINE_EXCEEDED)
  DECLARE_ERROR(DataLoss, DATA_LOSS)
  DECLARE_ERROR(Unknown, UNKNOWN)
  DECLARE_ERROR(PermissionDenied, PERMISSION_DENIED)
  DECLARE_ERROR(Unauthenticated, UNAUTHENTICATED)

  #undef DECLARE_ERROR
    }//namespace errors
} // namespace opengl



#endif
