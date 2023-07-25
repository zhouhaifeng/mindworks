// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"

#if CPU_SIMD
  #include "ie_system_conf.h"

#ifdef HAVE_AVX512
  #include "cpu_x86_avx512/ie_preprocess_gapi_kernels_avx512.hpp"
#endif

#ifdef HAVE_AVX2
  #include "cpu_x86_avx2/ie_preprocess_gapi_kernels_avx2.hpp"
#endif

#ifdef HAVE_SSE
  #include "cpu_x86_sse42/ie_preprocess_gapi_kernels_sse42.hpp"
#endif

#endif

#ifdef HAVE_NEON
  #include "arm_neon/ie_preprocess_gapi_kernels_neon.hpp"
#endif

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>
#include <opencv2/gapi/gcompoundkernel.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>
#include <functional>

#if defined(__GNUC__) && (__GNUC__ <= 5)
#include <cmath>
#endif

namespace InferenceEngine {
namespace gapi {
namespace kernels {

using isas_set = typelist<
#ifdef HAVE_AVX512
        avx512_tag,
#endif
#ifdef HAVE_AVX2
        avx2_tag,
#endif
#ifdef HAVE_SSE
        sse42_tag,
#endif
#ifdef HAVE_NEON
        neon_tag,
#endif
    //scalar "ISA" have to be the last one in the list,
    //as the search for supported ISA is performed until first match
    scalar_tag>;
#ifdef HAVE_AVX512
    inline bool is_present(avx512_tag) { return with_cpu_x86_avx512f(); }
#endif  // HAVE_AVX512

#ifdef HAVE_AVX2
    inline bool is_present(avx2_tag) { return with_cpu_x86_avx2(); }
#endif  // HAVE_AVX2

#ifdef HAVE_SSE
    inline bool is_present(sse42_tag) { return with_cpu_x86_sse42(); }
#endif  // HAVE_SSE

#ifdef HAVE_NEON
    inline bool is_present(neon_tag) { return true; }
#endif  // HAVE_NEON

//scalar version of kernels is always available
inline bool is_present(scalar_tag) { return true; }

struct is_isa_present {
    template< typename isa_tag_t>
    bool operator()(type_to_type<isa_tag_t>) {
        return is_present(isa_tag_t{});
    }
};

namespace {

using merge_supported_types = typelist<uint8_t, int8_t, uint16_t, int16_t, int32_t, float, fp_16_t>;

template<typename T, int chs>
inline void mergeRowImpl(scalar_tag, const std::array<const T*, chs>& ins, T* out, const int length) {
    for (int x = 0; x < length; ++x) {
        for (int c = 0; c < chs; ++c) {
            out[chs * x + c] = ins[c][x];
        }
    }
}

template<typename isa_tag_t, int chs>
struct typed_merge_row {
    using p_f = void (*)(const std::array<const uint8_t*, chs>& ins, uint8_t* out, const int length);

    template <typename type>
    inline typename std::enable_if<std::is_same<isa_tag_t, scalar_tag>::value ||
                                  (!std::is_same<isa_tag_t, scalar_tag>::value &&
                                   !std::is_same<type, uint8_t>::value &&
                                   !std::is_same<type, float>::value), p_f>::type
    operator()(type_to_type<type> ) {
        return [](const std::array<const uint8_t*, chs>& ins, uint8_t* out, const int length) {
            const auto inT = reinterpret_cast<const std::array<const type*, chs>&>(ins);
            auto outT = reinterpret_cast<type*>(out);
            scalar_tag t;
            mergeRowImpl<type, chs>(t, inT, outT, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](const std::array<const uint8_t*, chs>& ins, uint8_t* out, const int length) {
            tag t;
            mergeRowImpl<tag, uint8_t, chs>(t, ins, out, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<float>) {
        return [](const std::array<const uint8_t*, chs>& ins, uint8_t* out, const int length) {
            const auto inT = reinterpret_cast<const std::array<const float*, chs>&>(ins);
            auto outT = reinterpret_cast<float*>(out);
            tag t;
            mergeRowImpl<tag, float, chs>(t, inT, outT, length);
        };
    }
};

}  // namespace

namespace {
using split_supported_types = typelist<uint8_t, int8_t, uint16_t, int16_t, int32_t, float, fp_16_t>;

template<typename T, int chs>
inline void splitRowImpl(scalar_tag, const T* in, std::array<T*, chs>& outs, const int length) {
    for (int x = 0; x < length; ++x) {
        for (int c = 0; c < chs; ++c) {
            outs[c][x] = in[chs * x + c];
        }
    }
}

template<typename isa_tag_t, int chs>
struct typed_split_row {
    using p_f = void (*)(const uint8_t* in, std::array<uint8_t*, chs>& outs, const int length);

    template <typename type>
    inline typename std::enable_if<std::is_same<isa_tag_t, scalar_tag>::value ||
                                   (!std::is_same<isa_tag_t, scalar_tag>::value &&
                                    !std::is_same<type, uint8_t>::value &&
                                    !std::is_same<type, float>::value), p_f>::type
    operator()(type_to_type<type> ) {
        return [](const uint8_t* in, std::array<uint8_t*, chs>& outs, const int length) {
            const auto inT = reinterpret_cast<const type*>(in);
            auto outT = reinterpret_cast<std::array<type*, chs>&>(outs);
            scalar_tag t;
            splitRowImpl<type, chs>(t, inT, outT, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](const uint8_t* in, std::array<uint8_t*, chs>& outs, const int length) {
            tag t;
            splitRowImpl<tag, uint8_t, chs>(t, in, outs, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<float>) {
        return [](const uint8_t* in, std::array<uint8_t*, chs>& outs, const int length) {
            const auto inT = reinterpret_cast<const float*>(in);
            auto outT = reinterpret_cast<std::array<float*, chs>&>(outs);
            tag t;
            splitRowImpl<tag, float, chs>(t, inT, outT, length);
        };
    }
};
}  // namespace

//    GAPI_OCV_KERNEL(OCVChanToPlane, ChanToPlane) {
//        static void run(const cv::Mat &in, int chan, cv::Mat &out) {
//            out.create(in.rows, in.cols, in.depth());
//            const auto rowFunc = (in.depth() == CV_8U) ? &chanToPlaneRow<uint8_t> : &chanToPlaneRow<float>;

//            for (int y = 0; y < out.rows; y++)
//            {
//                rowFunc(in.data + y*in.step, chan, in.channels(), out.data + y*out.step, in.cols);
//            }
//        }
//    };

//    GAPI_OCV_KERNEL(OCVScalePlane, ScalePlane) {
//        static void run(const cv::Mat &in, int /*type*/, const Size &sz, int interp, cv::Mat &out) {
//            cv::resize(in, out, sz, 0, 0, interp);
//        }
//    };

//    GAPI_OCV_KERNEL(OCVMerge2, Merge2) {
//        static void run(const cv::Mat &a, const cv::Mat &b, cv::Mat out) {
//            out.create(a.rows, a.cols, CV_MAKETYPE(a.depth(), 2));
//            const auto rowFunc = (a.depth() == CV_8U) ? &mergeRow<uint8_t, 2> : &mergeRow<float, 2>;

//            for (int y = 0; y < out.rows; y++)
//            {
//                rowFunc({a.data + y*a.step, b.data + y*b.step}, out.data + out.step, a.cols);
//            }
//        }
//    };

namespace {

using chan_to_plane_supported_types = typelist<uint8_t, float>;

template<typename T>
inline void chanToPlaneRowImpl(scalar_tag, const T* in, int chan, int chs, T* out, int length) {
    for (int x = 0; x < length; x++) {
        out[x] = in[x*chs + chan];
    }
}

template<typename isa_tag_t>
struct typed_chan_to_plane_row {
    using p_f = void (*)(const uint8_t* in, int chan, int chs, uint8_t* out, int length);

    template <typename type>
    p_f operator()(type_to_type<type> ) {
        return [](const uint8_t* in, int chan, int chs, uint8_t* out, int length){
            const auto inT  = reinterpret_cast<const type*>(in);
                  auto outT = reinterpret_cast<      type*>(out);

            chanToPlaneRowImpl(isa_tag_t{}, inT, chan, chs, outT, length);
        };
    }
};
} //namespace

namespace {

using nv12_to_rgb_supported_types = typelist<uint8_t>;

inline void nv12ToRgbRowImpl(scalar_tag, const uint8_t** y_rows, const uint8_t* uv_row,
                             uint8_t** out_rows, const int buf_width) {
    for (int i = 0; i < buf_width; i += 2) {
        uint8_t u = uv_row[i];
        uint8_t v = uv_row[i + 1];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uint8_t vy = y_rows[y][i + x];
                uint8_t r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                out_rows[y][3 * (i + x)] = r;
                out_rows[y][3 * (i + x) + 1] = g;
                out_rows[y][3 * (i + x) + 2] = b;
            }
        }
    }
}

template<typename isa_tag_t>
struct typed_nv12_to_rgb_row {
    using p_f = void (*)(const uint8_t** y_rows, const uint8_t* uv_row,
                         uint8_t** out_rows, const int buf_width);

    template <typename type>
    p_f operator()(type_to_type<type>) {
        return [](const uint8_t** y_rows, const uint8_t* uv_row,
                  uint8_t** out_rows, const int buf_width) {
            const auto inT1 = reinterpret_cast<const type**>(y_rows);
            const auto inT2 = reinterpret_cast<const type*>(uv_row);
            auto outT = reinterpret_cast<type**>(out_rows);

            nv12ToRgbRowImpl(isa_tag_t{}, inT1, inT2, outT, buf_width);
        };
    }
};
}  // namespace

namespace {

using i420_to_rgb_supported_types = typelist<uint8_t>;

inline void i420ToRgbRowImpl(scalar_tag, const  uint8_t** y_rows,
                             const  uint8_t* u_row,
                             const  uint8_t* v_row,
                             uint8_t** out_rows,
                             const int buf_width) {
    for (int i = 0; i < buf_width; i += 2) {
        uchar u = u_row[i / 2];
        uchar v = v_row[i / 2];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uchar vy = y_rows[y][i + x];
                uchar r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                out_rows[y][3 * (i + x)] = r;
                out_rows[y][3 * (i + x) + 1] = g;
                out_rows[y][3 * (i + x) + 2] = b;
            }
        }
    }
}

template<typename isa_tag_t>
struct typed_i420_to_rgb_row {
    using p_f = void (*)(const uint8_t** y_rows, const uint8_t* u_row, const uint8_t* v_row,
                         uint8_t** out_rows, const int buf_width);

    template <typename type>
    p_f operator()(type_to_type<type>) {
        return [](const uint8_t** y_rows, const uint8_t* u_row, const uint8_t* v_row,
                  uint8_t** out_rows, const int buf_width) {
                const auto inT1 = reinterpret_cast<const type**>(y_rows);
                const auto inT2 = reinterpret_cast<const type*>(u_row);
                const auto inT3 = reinterpret_cast<const type*>(v_row);
                auto outT = reinterpret_cast<type**>(out_rows);

                i420ToRgbRowImpl(isa_tag_t{}, inT1, inT2, inT3, outT, buf_width);
        };
    }
};
}  // namespace

namespace linear {
struct Mapper {
    typedef short alpha_type;
    typedef short index_type;
    constexpr static const int unity = ONE;

    typedef MapperUnit<short, short> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        float f = static_cast<float>((outCoord + 0.5) * ratio - 0.5);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>(ONE * (1.0f - f));
        u.alpha1 = saturate_cast<short>(ONE *         f);

        return u;
    }
};
}  // namespace linear

namespace linear32f {
struct Mapper {
    typedef float alpha_type;
    typedef int   index_type;
    constexpr static const float unity = 1;

    typedef MapperUnit<float, int> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        float f = static_cast<float>((outCoord + 0.5) * ratio - 0.5);
        int s = cvFloor(f);
        f -= s;

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = 1.f - f;
        u.alpha1 =       f;

        return u;
    }
};
}  // namespace linear32f

template<typename T, typename Mapper, int chanNum>
struct linearScratchDesc {
    using alpha_t = typename Mapper::alpha_type;
    using index_t = typename Mapper::index_type;

    alpha_t* alpha;
    alpha_t* clone;
    index_t* mapsx;
    alpha_t* beta;
    index_t* mapsy;
    T*       tmp;

    linearScratchDesc(int /*inW*/, int /*inH*/, int outW, int outH,  void* data) {
        alpha = reinterpret_cast<alpha_t*>(data);
        clone = reinterpret_cast<alpha_t*>(alpha + outW);
        mapsx = reinterpret_cast<index_t*>(clone + outW*4);
        beta  = reinterpret_cast<alpha_t*>(mapsx + outW);
        mapsy = reinterpret_cast<index_t*>(beta  + outH);
        tmp   = reinterpret_cast<T*>      (mapsy + outH*2);
    }

    static int bufSize(int inW, int /*inH*/, int outW, int outH, int lpi) {
        auto size = outW * sizeof(alpha_t)     +
                    outW * sizeof(alpha_t) * 4 +  // alpha clones // previous alpha is redundant?
                    outW * sizeof(index_t)     +
                    outH * sizeof(alpha_t)     +
                    outH * sizeof(index_t) * 2 +
                     inW * sizeof(T) * lpi * chanNum;

        return static_cast<int>(size);
    }
};

static inline double invRatio(int inSz, int outSz) {
    return static_cast<double>(outSz) / inSz;
}

static inline double ratio(int inSz, int outSz) {
    return 1 / invRatio(inSz, outSz);
}

template<typename T, typename Mapper, int chanNum = 1>
static inline void initScratchLinear(const cv::GMatDesc& in,
                                     const         Size& outSz,
                                     cv::gapi::fluid::Buffer& scratch,
                                     int  lpi) {
    using alpha_type = typename Mapper::alpha_type;
    static const auto unity = Mapper::unity;

    auto inSz = in.size;
    auto sbufsize = linearScratchDesc<T, Mapper, chanNum>::bufSize(inSz.width, inSz.height, outSz.width, outSz.height, lpi);

    Size scratch_size{sbufsize, 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    double hRatio = ratio(in.size.width, outSz.width);
    double vRatio = ratio(in.size.height, outSz.height);

    linearScratchDesc<T, Mapper, chanNum> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    auto *alpha = scr.alpha;
    auto *clone = scr.clone;
    auto *index = scr.mapsx;

    for (int x = 0; x < outSz.width; x++) {
        auto map = Mapper::map(hRatio, 0, in.size.width, x);
        auto alpha0 = map.alpha0;
        auto index0 = map.index0;

        // TRICK:
        // Algorithm takes pair of input pixels, sx0'th and sx1'th,
        // and compute result as alpha0*src[sx0] + alpha1*src[sx1].
        // By definition: sx1 == sx0 + 1 either sx1 == sx0, and
        // alpha0 + alpha1 == unity (scaled appropriately).
        // Here we modify formulas for alpha0 and sx1: by assuming
        // that sx1 == sx0 + 1 always, and patching alpha0 so that
        // result remains intact.
        // Note that we need in.size.width >= 2, for both sx0 and
        // sx0+1 were indexing pixels inside the input's width.
        if (map.index1 != map.index0 + 1) {
            GAPI_DbgAssert(map.index1 == map.index0);
            GAPI_DbgAssert(in.size.width >= 2);
            if (map.index0 < in.size.width-1) {
                // sx1=sx0+1 fits inside row,
                // make sure alpha0=unity and alpha1=0,
                // so that result equals src[sx0]*unity
                alpha0 = saturate_cast<alpha_type>(unity);
            } else {
                // shift sx0 to left by 1 pixel,
                // and make sure that alpha0=0 and alpha1==1,
                // so that result equals to src[sx0+1]*unity
                alpha0 = 0;
                index0--;
            }
        }

        alpha[x] = alpha0;
        index[x] = index0;

        for (int l = 0; l < 4; l++) {
            clone[4*x + l] = alpha0;
        }
    }

    auto *beta    = scr.beta;
    auto *index_y = scr.mapsy;

    for (int y = 0; y < outSz.height; y++) {
        auto mapY = Mapper::map(vRatio, 0, in.size.height, y);
        beta[y] = mapY.alpha0;
        index_y[y] = mapY.index0;
        index_y[outSz.height + y] = mapY.index1;
    }
}

template<typename T, typename IT, typename AT, class Mapper>
inline void calcRowLinearC1Impl(T*    dst[],
                                const T*    src0[],
                                const T*    src1[],
                                const AT    alpha[],
                                const IT    mapsx[],
                                const AT    beta[],
                                const Size& inSz,
                                const Size& outSz,
                                const int   lpi,
                                const int   length) {
    using alpha_type = typename Mapper::alpha_type;
    for (int l = 0; l < lpi; l++) {
        constexpr static const auto unity = Mapper::unity;

        auto beta0 = beta[l];
        auto beta1 = saturate_cast<alpha_type>(unity - beta[l]);

        for (int x = 0; x < length; x++) {
            auto alpha0 = alpha[x];
            auto alpha1 = saturate_cast<alpha_type>(unity - alpha[x]);
            auto sx0 = mapsx[x];
            auto sx1 = sx0 + 1;
            T tmp0 = calc(beta0, src0[l][sx0], beta1, src1[l][sx0]);
            T tmp1 = calc(beta0, src0[l][sx1], beta1, src1[l][sx1]);
            dst[l][x] = calc(alpha0, tmp0, alpha1, tmp1);
        }
    }
}

namespace {

using resizeLinearU8C1_suptypes = typelist<uint8_t>;

template<class Mapper>
inline void calcRowLinear8UC1Impl(scalar_tag, uint8_t* dst[],
                                  const uint8_t* src0[],
                                  const uint8_t* src1[],
                                  const short    alpha[],
                                  const short    clone[],  // 4 clones of alpha
                                  const short    mapsx[],
                                  const short    beta[],
                                  uint8_t        tmp[],
                                  const Size&    inSz,
                                  const Size&    outSz,
                                  const int      lpi,
                                  const int      length) {
    calcRowLinearC1Impl<uint8_t, short,
                        short, Mapper>(dst, src0, src1, alpha, mapsx,
                                       beta, inSz, outSz, lpi, length);
}

template<typename isa_tag_t, class Mapper>
struct typed_resizeLinearU8C1 {
    using p_f = void (*)(uint8_t* dst[], const uint8_t* src0[], const uint8_t* src1[],
                         const short alpha[], const short clone[], const short mapsx[],
                         const short beta[], uint8_t tmp[], const Size& inSz,
                         const Size& outSz, const int lpi, const int length);

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](uint8_t* dst[], const uint8_t* src0[], const uint8_t* src1[],
                  const short alpha[], const short clone[], const short mapsx[],
                  const short beta[], uint8_t tmp[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            if (!calcRowLinear8UC1Impl(isa_tag_t{}, dst, src0,
                                       src1, alpha, clone,
                                       mapsx, beta, tmp,
                                       inSz, outSz, lpi, length))
                calcRowLinear8UC1Impl<Mapper>(scalar_tag{}, dst, src0, src1, alpha, clone,
                                              mapsx, beta, tmp, inSz, outSz, lpi, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](uint8_t* dst[], const uint8_t* src0[], const uint8_t* src1[],
                  const short alpha[], const short clone[], const short mapsx[],
                  const short beta[], uint8_t tmp[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            calcRowLinear8UC1Impl<Mapper>(isa_tag_t{}, dst, src0, src1, alpha, clone,
                                          mapsx, beta, tmp, inSz, outSz, lpi, length);
        };
    }
};
}  // namespace

namespace {

using resizeLinearF32C1_suptypes = typelist<float>;

template<class Mapper>
inline void calcRowLinear32FC1Impl(scalar_tag, float* dst[],
                                   const float* src0[],
                                   const float* src1[],
                                   const float  alpha[],
                                   const int    mapsx[],
                                   const float  beta[],
                                   const Size& inSz,
                                   const Size& outSz,
                                   const int   lpi,
                                   const int  length) {
    calcRowLinearC1Impl<float, int,
                        float, Mapper>(dst, src0, src1, alpha, mapsx,
                                       beta, inSz, outSz, lpi, length);
}

template<typename isa_tag_t, class Mapper>
struct typed_resizeLinearF32C1 {
    using p_f = void (*)(float* dst[], const float* src0[], const float* src1[],
                         const float alpha[], const int mapsx[],
                         const float beta[], const Size& inSz,
                         const Size& outSz, const int lpi, const int length);

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
        operator()(type_to_type<float>) {
        return [](float* dst[], const float* src0[], const float* src1[],
                  const float alpha[], const int mapsx[],
                  const float beta[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            calcRowLinear32FC1Impl(isa_tag_t{}, dst, src0, src1, alpha,
                                   mapsx, beta, inSz, outSz, lpi, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<std::is_same<tag, scalar_tag>::value, p_f>::type
        operator()(type_to_type<float>) {
        return [](float* dst[], const float* src0[], const float* src1[],
                  const float alpha[], const int mapsx[],
                  const float beta[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            calcRowLinear32FC1Impl<Mapper>(isa_tag_t{}, dst, src0, src1, alpha,
                                           mapsx, beta, inSz, outSz, lpi, length);
        };
    }
};
}  // namespace

namespace {

using resizeLinearU8C3C4_suptypes = typelist<uint8_t>;

template<class Mapper, int chs>
inline void calcRowLinear8UC3C4Impl(scalar_tag,
                                    std::array<std::array<uint8_t*, 4>, chs>& dst,
                                    const uint8_t* src0[],
                                    const uint8_t* src1[],
                                    const short    alpha[],
                                    const short    clone[],  // 4 clones of alpha
                                    const short    mapsx[],
                                    const short    beta[],
                                    uint8_t        tmp[],
                                    const Size&    inSz,
                                    const Size&    outSz,
                                    const int      lpi,
                                    const int      length) {
    using alpha_type = typename Mapper::alpha_type;
    for (int l = 0; l < lpi; l++) {
        constexpr static const auto unity = Mapper::unity;

        auto beta0 = beta[l];
        auto beta1 = saturate_cast<alpha_type>(unity - beta[l]);

        for (int x = 0; x < length; x++) {
            auto alpha0 = alpha[x];
            auto alpha1 = saturate_cast<alpha_type>(unity - alpha[x]);
            auto sx0 = mapsx[x];
            auto sx1 = sx0 + 1;

            for (int c = 0; c < chs; c++) {
                auto idx0 = chs * sx0 + c;
                auto idx1 = chs * sx1 + c;
                uint8_t tmp0 = calc(beta0, src0[l][idx0], beta1, src1[l][idx0]);
                uint8_t tmp1 = calc(beta0, src0[l][idx1], beta1, src1[l][idx1]);
                dst[c][l][x] = calc(alpha0, tmp0, alpha1, tmp1);
            }
        }
    }
}

template<typename isa_tag_t, class Mapper, int chs>
struct typed_resizeLinearU8C3C4 {
    using p_f = void (*)(std::array<std::array<uint8_t*, 4>, chs>& dst, const uint8_t* src0[], const uint8_t* src1[],
                         const short alpha[], const short clone[], const short mapsx[],
                         const short beta[], uint8_t tmp[], const Size& inSz,
                         const Size& outSz, const int lpi, const int length);

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<!std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](std::array<std::array<uint8_t*, 4>, chs>& dst, const uint8_t* src0[], const uint8_t* src1[],
                  const short alpha[], const short clone[], const short mapsx[],
                  const short beta[], uint8_t tmp[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            if (!calcRowLinear8UC3C4Impl<isa_tag_t, chs>(isa_tag_t{}, dst, src0,
                                         src1, alpha, clone,
                                         mapsx, beta, tmp,
                                         inSz, outSz, lpi, length))
                calcRowLinear8UC3C4Impl<Mapper, chs>(scalar_tag{}, dst, src0, src1, alpha, clone,
                                                   mapsx, beta, tmp, inSz, outSz, lpi, length);
        };
    }

    template<typename tag = isa_tag_t>
    inline typename std::enable_if<std::is_same<tag, scalar_tag>::value, p_f>::type
    operator()(type_to_type<uint8_t>) {
        return [](std::array<std::array<uint8_t*, 4>, chs>& dst, const uint8_t* src0[],
                  const uint8_t* src1[], const short alpha[], const short clone[],
                  const short mapsx[], const short beta[], uint8_t tmp[], const Size& inSz,
                  const Size& outSz, const int lpi, const int length) {
            calcRowLinear8UC3C4Impl<Mapper, chs>(isa_tag_t{}, dst, src0, src1, alpha, clone,
                                                 mapsx, beta, tmp, inSz, outSz, lpi, length);
        };
    }
};
}  // namespace

template<typename A, typename I, typename W>
struct AreaDownMapper {
    typedef A alpha_type;
    typedef I index_type;
    typedef W  work_type;

    typedef MapperUnit<alpha_type, index_type> Unit;

    inline Unit map(int outCoord) {
        double inCoord0 =  outCoord      * ratio;
        double inCoord1 = (outCoord + 1) * ratio;

        double index0 = std::floor(inCoord0 + 0.001);
        double index1 =  std::ceil(inCoord1 - 0.001);

        double alpha0 =   (index0 + 1 - inCoord0) * inv_ratio;
        double alpha1 = - (index1 - 1 - inCoord1) * inv_ratio;

        GAPI_Assert((0 <= outCoord) && (outCoord <= outSz-1));
        GAPI_Assert((0 <= index0) && (index0 < index1) && (index1 <= inSz));

        Unit unit;

        unit.index0 = checked_cast<index_type>(index0);
        unit.index1 = checked_cast<index_type>(index1);

        unit.alpha0 = convert_cast<alpha_type>(alpha0);
        unit.alpha1 = convert_cast<alpha_type>(alpha1);

        return unit;
    }

    int    inSz, outSz;
    double ratio, inv_ratio;

    alpha_type  alpha;  // == inv_ratio, rounded

    AreaDownMapper(int _inSz, int _outSz) {
        inSz  = _inSz;
        outSz = _outSz;

        inv_ratio = invRatio(inSz, outSz);
        ratio     = 1.0 / inv_ratio;

        alpha = convert_cast<alpha_type>(inv_ratio);
    }
};

namespace areaDownscale32f {
struct Mapper: public AreaDownMapper<float, int, float> {
    Mapper(int _inSz, int _outSz):AreaDownMapper(_inSz, _outSz) {}
};
}

namespace areaDownscale8u {
struct Mapper: public AreaDownMapper<Q0_16, short, Q8_8> {
    Mapper(int _inSz, int _outSz):AreaDownMapper(_inSz, _outSz) {}
};
}

namespace areaUpscale {
struct Mapper {
    typedef short alpha_type;
    typedef short index_type;
    constexpr static const int unity = ONE;

    typedef MapperUnit<short, short> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        const int s = cvFloor(outCoord*ratio);
        float res = static_cast<float>((outCoord + 1) - (s + 1)/ratio);
        res = res <= 0 ? 0.f : res - cvFloor(res);

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((res == 0.0) || (s + 1 >= max)) ? s - start : s - start + 1;

        u.alpha0 = saturate_cast<short>(ONE * (1.0f - res));
        u.alpha1 = saturate_cast<short>(ONE * res);

        return u;
    }
};
}  // namespace areaUpscale

namespace areaUpscale32f {
struct Mapper {
    typedef float alpha_type;
    typedef int   index_type;
    constexpr static const float unity = 1;

    typedef MapperUnit<float, int> Unit;

    static inline Unit map(double ratio, int start, int max, int outCoord) {
        int s = cvFloor(outCoord*ratio);
        float f = static_cast<float>((outCoord+1) - (s+1)/ratio);
        f = f <= 0 ? 0.f : f - cvFloor(f);

        Unit u;

        u.index0 = std::max(s - start, 0);
        u.index1 = ((f == 0.0) || s + 1 >= max) ? s - start : s - start + 1;

        u.alpha0 = 1.0f - f;
        u.alpha1 =        f;

        return u;
    }
};
}  // namespace areaUpscale32f

template<typename Mapper>
static void initScratchArea(const cv::GMatDesc& in, const Size& outSz,
                            cv::gapi::fluid::Buffer &scratch) {
    using Unit = typename Mapper::Unit;
    using alpha_type = typename Mapper::alpha_type;
    using index_type = typename Mapper::index_type;

    // compute the chunk of input pixels for each output pixel,
    // along with the coefficients for taking the weigthed sum

    Size inSz = in.size;
    Mapper mapper(inSz.width, outSz.width);

    std::vector<Unit> xmaps(outSz.width);
    int  maxdif = 0;

    for (int w = 0; w < outSz.width; w++) {
        Unit map = mapper.map(w);
        xmaps[w] = map;

        maxdif = std::max(maxdif, map.index1 - map.index0);
    }

    // This assertion is critical for our trick with chunk sizes:
    // we would expand a chunk it is smaller than maximal size
    GAPI_Assert(inSz.width >= maxdif);

    // pack the input chunks positions and coefficients into scratch-buffer,
    // along with the maximal size of chunk (note that chunk size may vary)

    size_t scratch_bytes =               sizeof(int)
                         + outSz.width * sizeof(index_type)
                         + outSz.width * sizeof(alpha_type) * maxdif
                         +  inSz.width * sizeof(alpha_type);
    Size scratch_size{static_cast<int>(scratch_bytes), 1};

    cv::GMatDesc desc(CV_8UC1, 1, scratch_size);

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    auto *maxdf =  scratch.OutLine<int>();
    auto *index = reinterpret_cast<index_type*>(maxdf + 1);
    auto *alpha = reinterpret_cast<alpha_type*>(index + outSz.width);

    for (int w = 0; w < outSz.width; w++) {
        // adjust input indices so that:
        // - data chunk is exactly maxdif pixels
        // - data chunk fits inside input width
        int index0 = xmaps[w].index0;
        int index1 = xmaps[w].index1;
        int i0 = index0;
        int i1 = index1;
        i1 = std::min(i0 + maxdif, in.size.width);
        i0 =          i1 - maxdif;
        GAPI_DbgAssert(i0 >= 0);

        // fulfill coefficients for the data chunk,
        // extending with zeros if any extra pixels
        alpha_type *alphaw = &alpha[w * maxdif];
        for (int i = 0; i < maxdif; i++) {
            if (i + i0 == index0) {
                alphaw[i] = xmaps[w].alpha0;

            } else if (i + i0 == index1 - 1) {
                alphaw[i] = xmaps[w].alpha1;

            } else if (i + i0 > index0 && i + i0 < index1 - 1) {
                alphaw[i] = mapper.alpha;

            } else {
                alphaw[i] = 0;
            }
        }

        // start input chunk with adjusted position
        index[w] = i0;
    }

    *maxdf = maxdif;
}

#if USE_CVKL
static int getResizeAreaTabSize(int dst_go, int ssize, int dsize, float scale) {
    static const float threshold = 1e-3f;
    int max_count = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if ((sx1 - fsx1) > threshold) {
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            count++;
        }

        if ((fsx2 - sx2) > threshold) {
            count++;
        }
        max_count = std::max(max_count, count);
    }

    return max_count;
}

// taken from: ie_preprocess_data.cpp
static void computeResizeAreaTab(int src_go, int dst_go, int ssize, int dsize, float scale,
                                 uint16_t* si, uint16_t* alpha, int max_count) {
    static const float threshold = 1e-3f;
    int k = 0;

    for (int col = dst_go; col < dst_go + dsize; col++) {
        int count = 0;

        float fsx1 = col * scale;
        float fsx2 = fsx1 + scale;
        float cellWidth = (std::min)(scale, ssize - fsx1);

        int sx1 = static_cast<int>(ceil(fsx1));
        int sx2 = static_cast<int>(floor(fsx2));

        sx2 = (std::min)(sx2, ssize - 1);
        sx1 = (std::min)(sx1, sx2);

        si[col - dst_go] = (uint16_t)(sx1 - src_go);

        if (sx1 - fsx1 > threshold) {
            si[col - dst_go] = (uint16_t)(sx1 - src_go - 1);
            alpha[k++] = (uint16_t)((1 << 16) * ((sx1 - fsx1) / cellWidth));
            count++;
        }

        for (int sx = sx1; sx < sx2; sx++) {
            alpha[k++] = (uint16_t)((1 << 16) * (1.0f / cellWidth));
            count++;
        }

        if (fsx2 - sx2 > threshold) {
            alpha[k++] = (uint16_t)((1 << 16) * ((std::min)((std::min)(fsx2 - sx2, 1.f), cellWidth) / cellWidth));
            count++;
        }

        if (count != max_count) {
            alpha[k++] = 0;
        }
    }
}

// teken from: ie_preprocess_data.cpp
static void generate_alpha_and_id_arrays(int x_max_count, int dcols, const uint16_t* xalpha, uint16_t* xsi,
                                         uint16_t** alpha, uint16_t** sxid) {
    if (x_max_count <= 4) {
        for (int col = 0; col < dcols; col++) {
            for (int x = 0; x < x_max_count; x++) {
                alpha[x][col] = xalpha[col*x_max_count + x];
            }
        }
    }
    if (x_max_count <= 4) {
        for (int col = 0; col <= dcols - 8; col += 8) {
            for (int chunk_num_h = 0; chunk_num_h < x_max_count; chunk_num_h++) {
                for (int i = 0; i < 128 / 16; i++) {
                    int id_diff = xsi[col + i] - xsi[col];

                    for (int chunk_num_v = 0; chunk_num_v < x_max_count; chunk_num_v++) {
                        uint16_t* sxidp = sxid[chunk_num_v] + col * x_max_count + chunk_num_h * 8;

                        int id0 = (id_diff + chunk_num_v) * 2 + 0;
                        int id1 = (id_diff + chunk_num_v) * 2 + 1;

                        (reinterpret_cast<int8_t*>(sxidp + i))[0] = static_cast<int8_t>(id0 >= (chunk_num_h * 16) && id0 < (chunk_num_h + 1) * 16 ? id0 : -1);
                        (reinterpret_cast<int8_t*>(sxidp + i))[1] = static_cast<int8_t>(id1 >= (chunk_num_h * 16) && id1 < (chunk_num_h + 1) * 16 ? id1 : -1);
                    }
                }
            }
        }
    }
}

// taken from: ie_preprocess_data.cpp
// (and simplified for specifically downscale area 8u)
static size_t resize_get_buffer_size(const Size& inSz, const Size& outSz) {
    int dst_full_width  = outSz.width;
    int dst_full_height = outSz.height;
    int src_full_width  =  inSz.width;
    int src_full_height =  inSz.height;

    auto resize_area_u8_downscale_sse_buffer_size = [&]() {
        const int dwidth  = outSz.width;
        const int dheight = outSz.height;
        const int swidth  =  inSz.width;

        const int dst_go_x = 0;
        const int dst_go_y = 0;

        int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width,  dwidth,  static_cast<float>(src_full_width)  / dst_full_width)  + 1;
        int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, static_cast<float>(src_full_height) / dst_full_height) + 1;

        size_t si_buf_size = sizeof(uint16_t) * dwidth + sizeof(uint16_t) * dheight;
        size_t alpha_buf_size =
                sizeof(uint16_t) * (dwidth * x_max_count + 8 * 16) + sizeof(uint16_t) * dheight * y_max_count;
        size_t vert_sum_buf_size = sizeof(uint16_t) * (swidth * 2);
        size_t alpha_array_buf_size = sizeof(uint16_t) * 4 * dwidth;
        size_t sxid_array_buf_size = sizeof(uint16_t) * 4 * 4 * dwidth;

        size_t buffer_size = si_buf_size +
                             alpha_buf_size +
                             vert_sum_buf_size +
                             alpha_array_buf_size +
                             sxid_array_buf_size;

        return buffer_size;
    };

    return resize_area_u8_downscale_sse_buffer_size();
}

// buffer-fulfill is taken from: ie_preprocess_data_sse42.cpp
static void initScratchArea_CVKL_U8(const cv::GMatDesc & in,
                                    const       Size   & outSz,
                                    cv::gapi::fluid::Buffer & scratch) {
    const Size& inSz = in.size;

    // estimate buffer size
    size_t scratch_bytes = resize_get_buffer_size(inSz, outSz);

    // allocate buffer

    Size scratch_size{static_cast<int>(scratch_bytes), 1};

    cv::GMatDesc desc;
    desc.chan = 1;
    desc.depth = CV_8UC1;
    desc.size = scratch_size;

    cv::gapi::fluid::Buffer buffer(desc);
    scratch = std::move(buffer);

    // fulfil buffer
    {
        // this code is taken from: ie_preprocess_data_sse42.cpp
        // (and simplified for 1-channel cv::Mat instead of blob)

        auto dwidth  = outSz.width;
        auto dheight = outSz.height;
        auto swidth  =  inSz.width;
        auto sheight =  inSz.height;

        const int src_go_x = 0;
        const int src_go_y = 0;
        const int dst_go_x = 0;
        const int dst_go_y = 0;

        auto src_full_width  = swidth;
        auto src_full_height = sheight;
        auto dst_full_width  = dwidth;
        auto dst_full_height = dheight;

        float scale_x = static_cast<float>(src_full_width)  / dst_full_width;
        float scale_y = static_cast<float>(src_full_height) / dst_full_height;

        int x_max_count = getResizeAreaTabSize(dst_go_x, src_full_width,  dwidth,  scale_x);
        int y_max_count = getResizeAreaTabSize(dst_go_y, src_full_height, dheight, scale_y);

        auto* maxdif = scratch.OutLine<int>();
        auto* xsi = reinterpret_cast<uint16_t*>(maxdif + 2);
        auto* ysi = xsi + dwidth;
        auto* xalpha = ysi + dheight;
        auto* yalpha = xalpha + dwidth*x_max_count + 8*16;

        maxdif[0] = x_max_count;
        maxdif[1] = y_max_count;

        computeResizeAreaTab(src_go_x, dst_go_x, src_full_width, dwidth, scale_x, xsi, xalpha, x_max_count);
        computeResizeAreaTab(src_go_y, dst_go_y, src_full_height, dheight, scale_y, ysi, yalpha, y_max_count);

        int vest_sum_size = 2*swidth;
        uint16_t* vert_sum = yalpha + dheight*y_max_count;
        uint16_t* alpha0 = vert_sum + vest_sum_size;
        uint16_t* alpha1 = alpha0 + dwidth;
        uint16_t* alpha2 = alpha1 + dwidth;
        uint16_t* alpha3 = alpha2 + dwidth;
        uint16_t* sxid0 = alpha3 + dwidth;
        uint16_t* sxid1 = sxid0 + 4*dwidth;
        uint16_t* sxid2 = sxid1 + 4*dwidth;
        uint16_t* sxid3 = sxid2 + 4*dwidth;

        uint16_t* alpha[] = {alpha0, alpha1, alpha2, alpha3};
        uint16_t* sxid[] = {sxid0, sxid1, sxid2, sxid3};
        generate_alpha_and_id_arrays(x_max_count, dwidth, xalpha, xsi, alpha, sxid);
    }
}

static void calcAreaRow_CVKL_U8(const cv::gapi::fluid::View   & in,
                                      cv::gapi::fluid::Buffer & out,
                                      cv::gapi::fluid::Buffer & scratch) {
    Size inSz  =  in.meta().size;
    Size outSz = out.meta().size;

    // this method is valid only for down-scale
    GAPI_DbgAssert(inSz.width  >= outSz.width);
    GAPI_DbgAssert(inSz.height >= outSz.height);

    int dwidth  = outSz.width;
    int dheight = outSz.height;

    auto* maxdif = scratch.OutLine<int>();
    int x_max_count = maxdif[0];
    int y_max_count = maxdif[1];

    auto* xsi = reinterpret_cast<uint16_t*>(maxdif + 2);
    auto* ysi    = xsi + dwidth;
    auto* xalpha = ysi + dheight;
    auto* yalpha = xalpha + dwidth*x_max_count + 8*16;
    auto* vert_sum = yalpha + dheight*y_max_count;

    int iny =  in.y();
    int   y = out.y();

    int lpi = out.lpi();
    GAPI_DbgAssert(y + lpi <= outSz.height);

    for (int l = 0; l < lpi; l++) {
        int yin0 = ysi[y + l];
        int yin1 = yin0 + y_max_count;

        GAPI_Assert(yin1 - yin0 <= 32);
        const uint8_t *src[32] = {};

        for (int yin = yin0; yin < yin1 && yin < inSz.height; yin++) {
            if (yalpha[(y+l)*y_max_count + yin - yin0] == 0) {
                src[yin - yin0] = in.InLine<const uint8_t>(yin - iny - 1);
            } else {
                src[yin - yin0] = in.InLine<const uint8_t>(yin - iny);
            }
        }

        uint8_t *dst = out.OutLine<uint8_t>(l);

        calcRowArea_CVKL_U8_SSE42(src, dst, inSz, outSz, y + l, xsi, ysi,
                      xalpha, yalpha, x_max_count, y_max_count, vert_sum);
    }
}

#endif  // USE_CVKL

namespace {

using resizeArea_suptypes = typelist<uint8_t, float>;

template<typename T, typename A, typename I, typename W>
inline void calcRowAreaImpl(scalar_tag,
                            T dst[], const T* src[], const Size& inSz,
                            const Size& outSz, A yalpha,
                            const MapperUnit<A, I>& ymap, int xmaxdf,
                            const I xindex[], const A xalpha[],
                            W vbuf[]) {
    // vertical pass
    int y_1st = ymap.index0;
    int ylast = ymap.index1 - 1;
    if (y_1st < ylast) {
        for (int w = 0; w < inSz.width; w++) {
            vbuf[w] = mulas(ymap.alpha0, src[0][w])        // Q8_8 = Q0_16 * U8
                    + mulas(ymap.alpha1, src[ylast - y_1st][w]);
        }

        for (int i = 1; i < ylast - y_1st; i++) {
            for (int w = 0; w < inSz.width; w++) {
                vbuf[w] += mulas(yalpha, src[i][w]);
            }
        }
    } else {
        for (int w = 0; w < inSz.width; w++) {
            vbuf[w] = convert_cast<W>(src[0][w]);  // Q8_8 = U8
        }
    }

    // horizontal pass
    for (int x = 0; x < outSz.width; x++) {
        W sum = 0;

        auto        index =  xindex[x];
        const auto *alpha = &xalpha[x * xmaxdf];

        for (int i = 0; i < xmaxdf; i++) {
            sum +=  mulaw(alpha[i], vbuf[index + i]);      // Q8_8 = Q0_16 * Q8_8
        }

        dst[x] = convert_cast<T>(sum);                     // U8 = Q8_8
    }
}

template<typename isa_tag_t, typename T, typename A, typename I, typename W>
struct typed_resizeArea {
    using p_f = void (*)(T dst[], const T* src[], const Size& inSz, const Size& outSz,
                         A yalpha, const MapperUnit<A, I>& ymap, int xmaxdf,
                         const I xindex[], const A xalpha[], W vbuf[]);

template <typename type>
inline p_f operator()(type_to_type<type>) {
    return [](T dst[], const T* src[], const Size& inSz, const Size& outSz,
              A yalpha, const MapperUnit<A, I>& ymap, int xmaxdf,
              const I xindex[], const A xalpha[], W vbuf[]) {
        calcRowAreaImpl(isa_tag_t{}, dst, src, inSz, outSz, yalpha,
                        ymap, xmaxdf, xindex, xalpha, vbuf);
    };
}
};
}  // namespace

template <typename isa_tag_t>
struct choose_impl {
GAPI_FLUID_KERNEL(FChanToPlane, ChanToPlane, false) {
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View& in, int chan,
                    cv::gapi::fluid::Buffer& out) {
        GAPI_DbgAssert(is_cv_type_in_list<chan_to_plane_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<chan_to_plane_supported_types>(out.meta().depth, cv_type_id{}, typed_chan_to_plane_row<isa_tag_t>{}, nullptr);

        GAPI_DbgAssert(rowFunc);

        rowFunc(in.InLineB(0), chan, in.meta().chan, out.OutLineB(), in.length());
    }
};

GAPI_FLUID_KERNEL(FNV12toRGB, NV12toRGB, false) {
    static const int Window = 1;
    static const int LPI = 2;
    static const auto Kind = cv::GFluidKernel::Kind::YUV420toRGB;

    static void run(const cv::gapi::fluid::View & in_y,
                    const cv::gapi::fluid::View & in_uv,
                    cv::gapi::fluid::Buffer & out) {
        GAPI_DbgAssert(is_cv_type_in_list<nv12_to_rgb_supported_types>(out.meta().depth));

        const uchar* uv_row = in_uv.InLineB(0);
        const uchar* y_rows[2] = { in_y.InLineB(0), in_y.InLineB(1) };
        uchar* out_rows[2] = { out.OutLineB(0), out.OutLineB(1) };

        int buf_width = out.length();

        const auto rowFunc = type_dispatch<nv12_to_rgb_supported_types>(out.meta().depth, cv_type_id{}, typed_nv12_to_rgb_row<isa_tag_t>{}, nullptr);

        GAPI_DbgAssert(rowFunc);

        rowFunc(y_rows, uv_row, out_rows, buf_width);
    }
};

GAPI_FLUID_KERNEL(FI420toRGB, I420toRGB, false) {
    static const int Window = 1;
    static const int LPI = 2;
    static const auto Kind = cv::GFluidKernel::Kind::YUV420toRGB;

    static void run(const cv::gapi::fluid::View & in_y,
                    const cv::gapi::fluid::View & in_u,
                    const cv::gapi::fluid::View & in_v,
                    cv::gapi::fluid::Buffer & out) {
        GAPI_DbgAssert(is_cv_type_in_list<i420_to_rgb_supported_types>(out.meta().depth));

        const uchar* u_row = in_u.InLineB(0);
        const uchar* v_row = in_v.InLineB(0);
        const uchar* y_rows[2] = { in_y.InLineB(0), in_y.InLineB(1) };
        uchar* out_rows[2] = { out.OutLineB(0), out.OutLineB(1) };

        int buf_width = out.length();
        GAPI_DbgAssert(in_u.length() == in_v.length());

        const auto rowFunc = type_dispatch<i420_to_rgb_supported_types>(out.meta().depth, cv_type_id{},
                                                                        typed_i420_to_rgb_row<isa_tag_t>{},
                                                                        nullptr);

        GAPI_DbgAssert(rowFunc);

        rowFunc(y_rows, u_row, v_row, out_rows, buf_width);
    }
};

GAPI_FLUID_KERNEL(FSplit2, Split2, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & in,
                    cv::gapi::fluid::Buffer & out1,
                    cv::gapi::fluid::Buffer & out2) {
        GAPI_DbgAssert(2 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{},
                                                                  typed_split_row<isa_tag_t, 2>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 2> outs = { out1.OutLineB(i), out2.OutLineB(i) };
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

GAPI_FLUID_KERNEL(FSplit3, Split3, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & in,
                    cv::gapi::fluid::Buffer & out1,
                    cv::gapi::fluid::Buffer & out2,
                    cv::gapi::fluid::Buffer & out3) {
        GAPI_DbgAssert(3 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(1 == out3.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out3.meta().depth);

        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{},
                                                                  typed_split_row<isa_tag_t, 3>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 3> outs = { out1.OutLineB(i), out2.OutLineB(i),
                                             out3.OutLineB(i) };
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

GAPI_FLUID_KERNEL(FSplit4, Split4, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & in,
                    cv::gapi::fluid::Buffer & out1,
                    cv::gapi::fluid::Buffer & out2,
                    cv::gapi::fluid::Buffer & out3,
                    cv::gapi::fluid::Buffer & out4) {
        GAPI_DbgAssert(4 == in.meta().chan);
        GAPI_DbgAssert(1 == out1.meta().chan);
        GAPI_DbgAssert(1 == out2.meta().chan);
        GAPI_DbgAssert(1 == out3.meta().chan);
        GAPI_DbgAssert(1 == out4.meta().chan);
        GAPI_DbgAssert(in.meta().depth == out1.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out2.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out3.meta().depth);
        GAPI_DbgAssert(in.meta().depth == out4.meta().depth);
        GAPI_DbgAssert(is_cv_type_in_list<split_supported_types>(in.meta().depth));

        const auto rowFunc = type_dispatch<split_supported_types>(in.meta().depth, cv_type_id{},
                                                                  typed_split_row<isa_tag_t, 4>{}, nullptr);
        for (int i = 0, lpi = out1.lpi(); i < lpi; i++) {
            std::array<uint8_t*, 4> outs = { out1.OutLineB(i), out2.OutLineB(i),
                                             out3.OutLineB(i), out4.OutLineB(i) };
            rowFunc(in.InLineB(i), outs, in.length());
        }
    }
};

GAPI_FLUID_KERNEL(FMerge2, Merge2, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & a,
                    const cv::gapi::fluid::View & b,
                    cv::gapi::fluid::Buffer & out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{},
                                                                  typed_merge_row<isa_tag_t, 2>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({ a.InLineB(l), b.InLineB(l) }, out.OutLineB(l), a.length());
        }
    }
};

GAPI_FLUID_KERNEL(FMerge3, Merge3, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & a,
                    const cv::gapi::fluid::View & b,
                    const cv::gapi::fluid::View & c,
                    cv::gapi::fluid::Buffer & out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{},
                                                                  typed_merge_row<isa_tag_t, 3>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({ a.InLineB(l), b.InLineB(l), c.InLineB(l) }, out.OutLineB(l), a.length());
        }
    }
};

GAPI_FLUID_KERNEL(FMerge4, Merge4, false) {
    static const int LPI = 4;
    static const int Window = 1;
    static void run(const cv::gapi::fluid::View & a,
                    const cv::gapi::fluid::View & b,
                    const cv::gapi::fluid::View & c,
                    const cv::gapi::fluid::View & d,
                    cv::gapi::fluid::Buffer & out) {
        GAPI_DbgAssert(is_cv_type_in_list<merge_supported_types>(out.meta().depth));

        const auto rowFunc = type_dispatch<merge_supported_types>(out.meta().depth, cv_type_id{},
                                                                  typed_merge_row<isa_tag_t, 4>{}, nullptr);
        for (int l = 0; l < out.lpi(); l++) {
            rowFunc({ a.InLineB(l), b.InLineB(l), c.InLineB(l), d.InLineB(l) }, out.OutLineB(l), a.length());
        }
    }
};

template<typename KT, typename TL>
static inline void callRowFunc(uint8_t* dst[], const uint8_t* src0[],
                               const uint8_t* src1[], const short alpha[],
                               const short clone[], const short mapsx[],
                               const short beta[], uint8_t tmp[], const Size& inSz,
                               const Size& outSz, const int lpi, const int length, const int depth) {
    const auto rowFunc = type_dispatch<TL>(depth, cv_type_id{}, KT{}, nullptr);

    GAPI_DbgAssert(rowFunc);

    rowFunc(dst, src0, src1, alpha, clone, mapsx, beta, tmp, inSz, outSz, lpi, length);
}

template<typename KT, typename TL>
static inline void callRowFunc(float* dst[], const float* src0[], const float* src1[],
                               const float alpha[], const float clone[], const int mapsx[],
                               const float beta[], float tmp[], const Size& inSz,
                               const Size& outSz, const int lpi, const int length, const int depth) {
    const auto rowFunc = type_dispatch<TL>(depth, cv_type_id{}, KT{}, nullptr);

    GAPI_DbgAssert(rowFunc);

    rowFunc(dst, src0, src1, alpha, mapsx, beta, inSz, outSz, lpi, length);
}

template<typename T, class Mapper, typename KT, typename LT>
static inline void calcRowLinear(const cv::gapi::fluid::View& in,
                                 cv::gapi::fluid::Buffer& out,
                                 cv::gapi::fluid::Buffer& scratch) {
    GAPI_DbgAssert(is_cv_type_in_list<LT>(out.meta().depth));

    auto  inSz = in.meta().size;
    auto outSz = out.meta().size;

    auto inY = in.y();
    int length = out.length();
    int outY = out.y();
    int lpi = out.lpi();
    GAPI_DbgAssert(outY + lpi <= outSz.height);

    GAPI_DbgAssert(lpi <= 4);

    linearScratchDesc<T, Mapper, 1> scr(inSz.width, inSz.height, outSz.width,
                                        outSz.height, scratch.OutLineB());

    const auto* alpha = scr.alpha;
    const auto* clone = scr.clone;
    const auto* mapsx = scr.mapsx;
    const auto* beta0 = scr.beta;
    const auto* mapsy = scr.mapsy;
    auto* tmp = scr.tmp;

    const auto* beta = beta0 + outY;
    const T* src0[4];
    const T* src1[4];
    T* dst[4];

    for (int l = 0; l < lpi; l++) {
        auto index0 = mapsy[outY + l] - inY;
        auto index1 = mapsy[outSz.height + outY + l] - inY;
        src0[l] = in.InLine<const T>(index0);
        src1[l] = in.InLine<const T>(index1);
        dst[l] = out.OutLine<T>(l);
    }

    callRowFunc<KT, LT>(dst, src0, src1, alpha, clone, mapsx, beta, tmp, inSz, outSz,
                        lpi, length, out.meta().depth);
}

GAPI_FLUID_KERNEL(FScalePlane8u, ScalePlane8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc & in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer & scratch) {
        initScratchLinear<uchar, linear::Mapper>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View & in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer & out, cv::gapi::fluid::Buffer & scratch) {
        calcRowLinear<uint8_t, linear::Mapper,
                      typed_resizeLinearU8C1<isa_tag_t, linear::Mapper>,
                      resizeLinearU8C1_suptypes>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlane32f, ScalePlane32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc & in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer & scratch) {
        GAPI_DbgAssert(in.depth == CV_32F && in.chan == 1);

        initScratchLinear<float, linear32f::Mapper>(in, outSz, scratch, 0);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View & in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer & out, cv::gapi::fluid::Buffer & scratch) {
        calcRowLinear<float, linear32f::Mapper,
                      typed_resizeLinearF32C1<isa_tag_t, linear32f::Mapper>,
                      resizeLinearF32C1_suptypes>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FUpscalePlaneArea8u, UpscalePlaneArea8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc & in,
        Size outSz, int /*interp*/,
        cv::gapi::fluid::Buffer & scratch) {
        initScratchLinear<uchar, areaUpscale::Mapper>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View & in, Size /*sz*/, int /*interp*/,
        cv::gapi::fluid::Buffer & out, cv::gapi::fluid::Buffer & scratch) {
        calcRowLinear<uint8_t, areaUpscale::Mapper, typed_resizeLinearU8C1<isa_tag_t, areaUpscale::Mapper>,
                      resizeLinearU8C1_suptypes>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FUpscalePlaneArea32f, UpscalePlaneArea32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc & in,
        Size outSz, int /*interp*/,
        cv::gapi::fluid::Buffer & scratch) {
        initScratchLinear<float, areaUpscale32f::Mapper>(in, outSz, scratch, 0);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View & in, Size /*sz*/, int /*interp*/,
        cv::gapi::fluid::Buffer & out, cv::gapi::fluid::Buffer & scratch) {
        calcRowLinear<float, areaUpscale32f::Mapper,
                      typed_resizeLinearF32C1<isa_tag_t, areaUpscale32f::Mapper>,
                      resizeLinearF32C1_suptypes>(in, out, scratch);
    }
};

template<typename T, class Mapper, int chs>
static inline void calcRowLinearC(const cv::gapi::fluid::View& in,
                                  std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, chs>& out,
                                  cv::gapi::fluid::Buffer& scratch) {
    GAPI_DbgAssert(is_cv_type_in_list<resizeLinearU8C3C4_suptypes>(in.meta().depth));

    auto  inSz =  in.meta().size;
    auto outSz = out[0].get().meta().size;

    auto inY  = in.y();
    auto outY = out[0].get().y();
    auto lpi  = out[0].get().lpi();

    GAPI_DbgAssert(outY + lpi <= outSz.height);
    GAPI_DbgAssert(lpi <= 4);

    linearScratchDesc<T, Mapper, chs> scr(inSz.width, inSz.height, outSz.width, outSz.height, scratch.OutLineB());

    const auto *alpha = scr.alpha;
    const auto *clone = scr.clone;
    const auto *mapsx = scr.mapsx;
    const auto *beta0 = scr.beta;
    const auto *mapsy = scr.mapsy;
    auto *tmp         = scr.tmp;

    const auto *beta = beta0 + outY;
    const T *src0[4];
    const T *src1[4];
    std::array<std::array<T*, 4>, chs> dst;

    for (int l = 0; l < lpi; l++) {
        auto index0 = mapsy[outY + l] - inY;
        auto index1 = mapsy[outSz.height + outY + l] - inY;
        src0[l] = in.InLine<const T>(index0);
        src1[l] = in.InLine<const T>(index1);
        for (int c=0; c < chs; c++) {
            dst[c][l] = out[c].get().template OutLine<T>(l);
        }
    }
    auto length = out[0].get().length();

    const auto rowFunc = type_dispatch<resizeLinearU8C3C4_suptypes>(in.meta().depth,
                                                                    cv_type_id{},
                                                                    typed_resizeLinearU8C3C4<isa_tag_t, Mapper, chs>{},
                                                                    nullptr);
    GAPI_DbgAssert(rowFunc);

    rowFunc(dst, src0, src1, alpha, clone, mapsx, beta, tmp, inSz, outSz, lpi, length);
}

GAPI_FLUID_KERNEL(FScalePlanes, ScalePlanes, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in, int, Size,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, linear::Mapper, 3>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, int, Size, Size/*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out1,
                    cv::gapi::fluid::Buffer& out2,
                    cv::gapi::fluid::Buffer& out3,
                    cv::gapi::fluid::Buffer& scratch) {
        constexpr int numChan = 3;
        std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, numChan> out = {out1, out2, out3};
        calcRowLinearC<uint8_t, linear::Mapper, numChan>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlanes4, ScalePlanes4, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in, int, Size,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer &scratch) {
        initScratchLinear<uchar, linear::Mapper, 4>(in, outSz, scratch, LPI);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, int, Size, Size/*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out1,
                    cv::gapi::fluid::Buffer& out2,
                    cv::gapi::fluid::Buffer& out3,
                    cv::gapi::fluid::Buffer& out4,
                    cv::gapi::fluid::Buffer& scratch) {
        constexpr int numChan = 4;
        std::array<std::reference_wrapper<cv::gapi::fluid::Buffer>, numChan> out = {out1, out2, out3, out4};
        calcRowLinearC<uint8_t, linear::Mapper, numChan>(in, out, scratch);
    }
};

#if defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

template<typename T, typename Mapper>
static inline void calcAreaRow(const cv::gapi::fluid::View& in, cv::gapi::fluid::Buffer& out,
                               cv::gapi::fluid::Buffer& scratch) {
    using Unit = typename Mapper::Unit;
    using alpha_type = typename Mapper::alpha_type;
    using index_type = typename Mapper::index_type;
    using  work_type = typename Mapper::work_type;

    Size inSz  =  in.meta().size;
    Size outSz = out.meta().size;

    // this method is valid only for down-scale
    GAPI_DbgAssert(inSz.width  >= outSz.width);
    GAPI_DbgAssert(inSz.height >= outSz.height);

    Mapper ymapper(inSz.height, outSz.height);

    auto *xmaxdf = scratch.OutLine<const int>();
    auto  maxdif = xmaxdf[0];

    auto *xindex = reinterpret_cast<const index_type*>(xmaxdf + 1);
    auto *xalpha = reinterpret_cast<const alpha_type*>(xindex + outSz.width);
    auto *vbuf_c = reinterpret_cast<const  work_type*>(xalpha + outSz.width * maxdif);

    auto *vbuf = const_cast<work_type*>(vbuf_c);

    int iny = in.y();
    int y = out.y();

    int lpi = out.lpi();
    GAPI_DbgAssert(y + lpi <= outSz.height);

    const auto rowFunc = type_dispatch<resizeArea_suptypes>(in.meta().depth,
                                                            cv_type_id{},
                                                            typed_resizeArea<isa_tag_t, T, alpha_type, index_type, work_type>{},
                                                            nullptr);
    GAPI_DbgAssert(rowFunc);
    constexpr int max_num = 32;
    for (int l = 0; l < lpi; l++) {
        Unit ymap = ymapper.map(y + l);

        GAPI_Assert(ymap.index1 - ymap.index0 <= max_num);
        GAPI_Assert(ymap.index1 - ymap.index0 > 0);
        const T *src[max_num] = {};

        for (int yin = ymap.index0; yin < ymap.index1; yin++) {
            src[yin - ymap.index0] = in.InLine<const T>(yin - iny);
        }

        auto dst = out.OutLine<T>(l);

        rowFunc(dst, src, inSz, outSz, ymapper.alpha, ymap, xmaxdf[0], xindex, xalpha, vbuf);
    }
}

#if defined __GNUC__
# pragma GCC diagnostic pop
#endif

GAPI_FLUID_KERNEL(FScalePlaneArea32f, ScalePlaneArea32f, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer& scratch) {
        initScratchArea<areaDownscale32f::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer& scratch) {
        calcAreaRow<float, areaDownscale32f::Mapper>(in, out, scratch);
    }
};

GAPI_FLUID_KERNEL(FScalePlaneArea8u, ScalePlaneArea8u, true) {
    static const int Window = 1;
    static const int LPI = 4;
    static const auto Kind = cv::GFluidKernel::Kind::Resize;

    static void initScratch(const cv::GMatDesc& in,
                            Size outSz, int /*interp*/,
                            cv::gapi::fluid::Buffer& scratch) {
#if USE_CVKL
        if (with_cpu_x86_sse42()) {
            const Size& inSz = in.size;
            if (inSz.width > outSz.width && inSz.height > outSz.height) {
                // CVKL code we use supports only downscale
                initScratchArea_CVKL_U8(in, outSz, scratch);
                return;
            }
        }
#endif

        initScratchArea<areaDownscale8u::Mapper>(in, outSz, scratch);
    }

    static void resetScratch(cv::gapi::fluid::Buffer& /*scratch*/) {
    }

    static void run(const cv::gapi::fluid::View& in, Size /*sz*/, int /*interp*/,
                    cv::gapi::fluid::Buffer& out, cv::gapi::fluid::Buffer& scratch) {
#if USE_CVKL
        if (with_cpu_x86_sse42()) {
            auto  inSz = in.meta().size;
            auto outSz = out.meta().size;
            if (inSz.width > outSz.width && inSz.height > outSz.height) {
                // CVKL's code supports only downscale
                calcAreaRow_CVKL_U8(in, out, scratch);
                return;
            }
        }
#endif

        calcAreaRow<uint8_t, areaDownscale8u::Mapper>(in, out, scratch);
    }
};
};

namespace {
struct CC_and_MergeISA {
    cv::gapi::GKernelPackage& pckg;

    CC_and_MergeISA(cv::gapi::GKernelPackage& _pckg) : pckg(_pckg) {}

    template<typename isa_tag_t>
    inline bool operator()(type_to_type<isa_tag_t>) {
        pckg.include<typename choose_impl<isa_tag_t>::FI420toRGB>();
        pckg.include<typename choose_impl<isa_tag_t>::FNV12toRGB>();
        pckg.include<typename choose_impl<isa_tag_t>::FChanToPlane>();
        pckg.include<typename choose_impl<isa_tag_t>::FMerge2>();
        pckg.include<typename choose_impl<isa_tag_t>::FMerge3>();
        pckg.include<typename choose_impl<isa_tag_t>::FMerge4>();
        //at the moment type_dispatch requires something to be returned by the lambda
        return true;
    }
};

struct Split_ResizeISA {
    cv::gapi::GKernelPackage& pckg;

    Split_ResizeISA(cv::gapi::GKernelPackage& _pckg) : pckg(_pckg) {}

    template<typename isa_tag_t>
    inline bool operator()(type_to_type<isa_tag_t>) {
        pckg.include<typename choose_impl<isa_tag_t>::FSplit2>();
        pckg.include<typename choose_impl<isa_tag_t>::FSplit3>();
        pckg.include<typename choose_impl<isa_tag_t>::FSplit4>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlane8u>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlane32f>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlanes>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlanes4>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlaneArea8u>();
        pckg.include<typename choose_impl<isa_tag_t>::FScalePlaneArea32f>();
        pckg.include<typename choose_impl<isa_tag_t>::FUpscalePlaneArea8u>();
        pckg.include<typename choose_impl<isa_tag_t>::FUpscalePlaneArea32f>();
        //at the moment type_dispatch requires something to be returned by the lambda
        return true;
    }
};
}  //namespace

inline cv::gapi::GKernelPackage FKernelsChooseISA() {
    // At the moment AVX512 implementation of wide universal intrinsics is slower than AVX2.
    // So, disable it for now.
    using isas = remove_t<isas_set, avx512_tag>;

    cv::gapi::GKernelPackage pckg1, pckg2;
    CC_and_MergeISA ccISA{ pckg1 };
    Split_ResizeISA sISA{ pckg2 };

    type_dispatch<isas>(is_isa_present{}, ccISA, false);
    type_dispatch<isas_set>(is_isa_present{}, sISA, false);

    return combine(pckg1, pckg2);
}

//----------------------------------------------------------------------

GAPI_COMPOUND_KERNEL(FScalePlane, ScalePlane) {
    static cv::GMat expand(cv::GMat in, int type, const Size& szIn, const Size& szOut, int interp) {
        GAPI_DbgAssert(CV_8UC1 == type || CV_32FC1 == type);
        GAPI_DbgAssert(cv::INTER_AREA == interp || cv::INTER_LINEAR == interp);

        if (cv::INTER_AREA == interp) {
            bool upscale = szIn.width < szOut.width || szIn.height < szOut.height;
            if (CV_8UC1 == type) {
                if (upscale)
                    return UpscalePlaneArea8u::on(in, szOut, interp);
                else
                    return   ScalePlaneArea8u::on(in, szOut, interp);
            }
            if (CV_32FC1 == type) {
                if (upscale)
                    return UpscalePlaneArea32f::on(in, szOut, interp);
                else
                    return   ScalePlaneArea32f::on(in, szOut, interp);
            }
        }

        if (cv::INTER_LINEAR == interp) {
            if (CV_8UC1 == type) {
                return ScalePlane8u::on(in, szOut, interp);
            }
            if (CV_32FC1 == type) {
                return ScalePlane32f::on(in, szOut, interp);
            }
        }

        GAPI_Assert(!"unsupported parameters");
        return {};
    }
};

//------------------------------------------------------------------------------

namespace {

template <typename src_t, typename dst_t>
void convert_precision(const uint8_t* src, uint8_t* dst, const int width) {
    const auto *in  = reinterpret_cast<const src_t *>(src);
          auto *out = reinterpret_cast<dst_t *>(dst);

    for (int i = 0; i < width; i++) {
        out[i] = saturate_cast<dst_t>(in[i]);
    }
}

}  // namespace

GAPI_FLUID_KERNEL(FConvertDepth, ConvertDepth, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View& src, int depth, cv::gapi::fluid::Buffer& dst) {
        GAPI_Assert(src.meta().depth == CV_8U || src.meta().depth == CV_32F || src.meta().depth == CV_16U);
        GAPI_Assert(dst.meta().depth == CV_8U || dst.meta().depth == CV_32F || dst.meta().depth == CV_16U);
        GAPI_Assert(src.meta().chan == 1);
        GAPI_Assert(dst.meta().chan == 1);
        GAPI_Assert(src.length() == dst.length());

        constexpr unsigned supported_types_n = 3;
        using p_f = void (*)( const uint8_t* src,  uint8_t* dst, const int width);
        using table_string_t = std::array<p_f, supported_types_n>;

        constexpr std::array<table_string_t, supported_types_n> func_table = {
                table_string_t{convert_precision<uint16_t, uint16_t>, convert_precision<uint16_t, float>, convert_precision<uint16_t, uint8_t>},
                table_string_t{convert_precision<float,    uint16_t>, convert_precision<float,    float>, convert_precision<float,    uint8_t>},
                table_string_t{convert_precision<uint8_t,  uint16_t>, convert_precision<uint8_t,  float>, convert_precision<uint8_t,  uint8_t>}
        };

        auto depth_to_index = [](int depth){
            switch (depth) {
                case  CV_16U: return 0;
                case  CV_32F: return 1;
                case  CV_8U:  return 2;
                default: GAPI_Assert(!"not supported depth"); return -1;
            }
        };
        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();
        auto const src_index = depth_to_index(src.meta().depth);
        auto const dst_index = depth_to_index(dst.meta().depth);

        (func_table[src_index][dst_index])(in, out, width);
    }
};

namespace {
    template <typename src_t, typename dst_t>
    void sub(const uint8_t* src, uint8_t* dst, const int width, double c) {
        const auto *in  = reinterpret_cast<const src_t *>(src);
              auto *out = reinterpret_cast<dst_t *>(dst);

        for (int i = 0; i < width; i++) {
            out[i] = saturate_cast<dst_t>(in[i] - c);
        }
    }

    template <typename src_t, typename dst_t>
    void div(const uint8_t* src, uint8_t* dst, const int width, double c) {
        const auto *in  = reinterpret_cast<const src_t *>(src);
              auto *out = reinterpret_cast<dst_t *>(dst);

        for (int i = 0; i < width; i++) {
            out[i] = saturate_cast<dst_t>(in[i] / c);
        }
    }
}  // namespace

GAPI_FLUID_KERNEL(FSubC, GSubC, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View& src, const cv::Scalar &scalar, int depth, cv::gapi::fluid::Buffer& dst) {
        GAPI_Assert(src.meta().depth == CV_32F && src.meta().chan == 1);

        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();

        sub<float, float>(in, out, width, scalar[0]);
    }
};

GAPI_FLUID_KERNEL(FDivC, GDivC, false) {
    static const int Window = 1;

    static void run(const cv::gapi::fluid::View &src, const cv::Scalar &scalar, double _scale, int /*dtype*/,
            cv::gapi::fluid::Buffer &dst) {
        GAPI_Assert(src.meta().depth == CV_32F && src.meta().chan == 1);

        const auto *in  = src.InLineB(0);
              auto *out = dst.OutLineB();

        auto const width = dst.length();

        div<float, float>(in, out, width, scalar[0]);
    }
};
}  // namespace kernels

//----------------------------------------------------------------------

using namespace kernels;

cv::gapi::GKernelPackage preprocKernels() {
    return combine(
        FKernelsChooseISA(),
        cv::gapi::kernels
        < FScalePlane
        , FConvertDepth
        , FSubC
        , FDivC
        >());
}

}  // namespace gapi
}  // namespace InferenceEngine
