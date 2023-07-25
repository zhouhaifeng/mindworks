// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic functions to convert from FP16 to FP32 and vice versa
 * @file precision_utils.h
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "ie_api.h"

/**
 * @brief Inference Engine Plugin API namespace
 */
namespace InferenceEngine {

/**
 * @brief A type difinition for FP16 data type. Defined as a singed short
 * @ingroup ie_dev_api_precision
 */
using ie_fp16 = short;

/**
 * @brief Namespace for precision utilities
 * @ingroup ie_dev_api_precision
 */
namespace PrecisionUtils {

/**
 * @brief      Converts a single-precision floating point value to a half-precision floating poit value
 * @ingroup    ie_dev_api_precision
 *
 * @param[in]  x     A single-precision floating point value
 * @return     A half-precision floating point value
 */
INFERENCE_ENGINE_API_CPP(ie_fp16) f32tof16(float x);

/**
 * @brief      Convers a half-precision floating point value to a single-precision floating point value
 * @ingroup    ie_dev_api_precision
 *
 * @param[in]  x     A half-precision floating point value
 * @return     A single-precision floating point value
 */
INFERENCE_ENGINE_API_CPP(float) f16tof32(ie_fp16 x);

/**
 * @brief      Converts a half-precision floating point array to single-precision floating point array
 * 	           and applies `scale` and `bias` is needed
 * @ingroup    ie_dev_api_precision
 *
 * @param      dst    A destination array of single-precision floating point values
 * @param[in]  src    A source array of half-precision floating point values
 * @param[in]  nelem  A number of elements in arrays
 * @param[in]  scale  An optional scale parameter
 * @param[in]  bias   An optional bias parameter
 */
INFERENCE_ENGINE_API_CPP(void)
f16tof32Arrays(float* dst, const ie_fp16* src, size_t nelem, float scale = 1.f, float bias = 0.f);

/**
 * @brief      Converts a single-precision floating point array to a half-precision floating point array
 *             and applies `scale` and `bias` if needed
 * @ingroup    ie_dev_api_precision
 *
 * @param      dst    A destination array of half-precision floating point values
 * @param[in]  src    A sources array of single-precision floating point values
 * @param[in]  nelem  A number of elements in arrays
 * @param[in]  scale  An optional scale parameter
 * @param[in]  bias   An optional bias parameter
 */
INFERENCE_ENGINE_API_CPP(void)
f32tof16Arrays(ie_fp16* dst, const float* src, size_t nelem, float scale = 1.f, float bias = 0.f);

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4018)
#endif

namespace details {

// To overcame syntax parse error, when `>` comparison operator is threated as template closing bracket
constexpr inline bool Greater(size_t v1, size_t v2) {
    return v1 > v2;
}

}  // namespace details

/**
 * @brief      Converts one integral type to another saturating the result if the source value doesn't fit
 *             into destination type range
 * @ingroup    ie_dev_api_precision
 *
 * @param      value   Value to be converted
 * @return     A saturated value
 */
template <class OutT,
          class InT,
          typename std::enable_if<std::is_integral<OutT>::value && std::is_integral<InT>::value &&
                                  std::is_signed<InT>::value && !std::is_same<OutT, InT>::value>::type* = nullptr>
inline OutT saturate_cast(const InT& value) {
    using MaxT = typename std::conditional<details::Greater(sizeof(OutT), sizeof(InT)),
                                           typename std::make_unsigned<OutT>::type,
                                           typename std::make_unsigned<InT>::type>::type;
    using MinT = typename std::conditional<details::Greater(sizeof(OutT), sizeof(InT)),
                                           typename std::make_signed<OutT>::type,
                                           typename std::make_signed<InT>::type>::type;

    static const MaxT OUT_MAX = static_cast<MaxT>(std::numeric_limits<OutT>::max());
    static const MaxT IN_MAX = static_cast<MaxT>(std::numeric_limits<InT>::max());

    static const MinT OUT_MIN = static_cast<MinT>(std::numeric_limits<OutT>::min());
    static const MinT IN_MIN = static_cast<MinT>(std::numeric_limits<InT>::min());

    if (OUT_MAX > IN_MAX && OUT_MIN < IN_MIN) {
        return static_cast<OutT>(value);
    }

    const InT max = static_cast<InT>(OUT_MAX < IN_MAX ? OUT_MAX : IN_MAX);
    const InT min = static_cast<InT>(OUT_MIN > IN_MIN ? OUT_MIN : IN_MIN);

    return static_cast<OutT>(std::min(std::max(value, min), max));
}

/**
 * @brief      Converts one integral type to another saturating the result if the source value doesn't fit
 *             into destination type range
 * @ingroup    ie_dev_api_precision
 *
 * @param      value   Value to be converted
 * @return     A saturated value
 */
template <class OutT,
          class InT,
          typename std::enable_if<std::is_integral<OutT>::value && std::is_integral<InT>::value &&
                                  std::is_unsigned<InT>::value && !std::is_same<OutT, InT>::value>::type* = nullptr>
inline OutT saturate_cast(const InT& value) {
    using MaxT = typename std::conditional<details::Greater(sizeof(OutT), sizeof(InT)),
                                           typename std::make_unsigned<OutT>::type,
                                           typename std::make_unsigned<InT>::type>::type;

    static const MaxT OUT_MAX = static_cast<MaxT>(std::numeric_limits<OutT>::max());
    static const MaxT IN_MAX = static_cast<MaxT>(std::numeric_limits<InT>::max());

    if (OUT_MAX > IN_MAX) {
        return static_cast<OutT>(value);
    }

    const InT max = static_cast<InT>(OUT_MAX < IN_MAX ? OUT_MAX : IN_MAX);

    return static_cast<OutT>(std::min(value, max));
}

#if defined(_MSC_VER)
#    pragma warning(pop)
#endif

/**
 * @brief      Converts one integral type to another saturating the result if the source value doesn't fit
 *             into destination type range
 * @ingroup    ie_dev_api_precision
 *
 * @param      value   Value to be converted
 * @return     A saturated value
 */
template <class InT>
inline InT saturate_cast(const InT& value) {
    return value;
}

}  // namespace PrecisionUtils

}  // namespace InferenceEngine
