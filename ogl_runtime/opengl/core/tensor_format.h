#ifndef OPENGL_CORE_TENSOR_FORMAT_H_
#define OPENGL_CORE_TENSOR_FORMAT_H_
/* Tensor Format give means for each dim in tensor.
 * for a tensor of shape like (1,3,4,4), if use nhwc to interpret it,
 * 4 means 4 channels, 3 means height and so on.
 *
 */

#include <string>
#include <glog/logging.h>
#include "opengl/core/types.h"

namespace opengl{
    // NHWC
    // NCHW
    // NHWC4
    // HWN4C4

    int GetChannel(const IntList& shape,
            DataFormat dformat);

    int GetBatch(const IntList& shape,
            DataFormat dformat);

    int GetWidth(const IntList& shape,
            DataFormat dformat);

    int GetHeight(const IntList& shape,
            DataFormat dformat);

    IntList MakeTensorShape(const int batch, const int height,
            const int width, const int channels, DataFormat dformat);

    IntList MakeTextureShape(const IntList shape, DataFormat dformat);

    IntList TensorShapeFromFormat(DataFormat dst_format,
            const IntList& src_shape, DataFormat src_format);

    // dformat strings and dformat enum
    std::string FormatToStr(DataFormat);
    DataFormat StrToFormat(std::string format_str);
    DataFormat FormatToStride4(DataFormat);
    DataFormat FormatFromStride4(DataFormat dformat);

    uint64 CalcAllocatedNumForDformat(const IntList, DataFormat dformat);
    bool IsHostDFormat(DataFormat dformat);
    bool IsDeviceDFormat(DataFormat dformat);
    bool IsStrideDFormat(DataFormat dformat);

    uint64 CalcAllocatedSize1D(const IntList& shape, DataFormat dformat);
    IntList CalcAllocatedSize2D(const IntList& shape, DataFormat dformat);

    bool CheckIndexValid(const uint64 index, const IntList& shape,
            DataFormat dformat);
}//namespace


#endif
