// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/runtime/half.hpp"

struct FLOAT16 {
    struct representation {
        uint16_t sign : 1;
        uint16_t exponent : 5;
        uint16_t significand : 10;
    };

    union {
        uint16_t v;
        representation format;  // added this struct for the .natvis file (for debug)
    };

    static constexpr FLOAT16 min_val() { return FLOAT16((uint16_t)(0x0400)); }

    static constexpr FLOAT16 lowest_val() { return FLOAT16((uint16_t)(0xfbff)); }

    operator double() const {
        double d = (double)cldnn::half_to_float(v);
        return d;
    }
    operator float() const {
        float f = cldnn::half_to_float(v);
        return f;
    }
    operator int16_t() const { return *(int16_t *)(&v); }
    operator long long int() const { return v; }
    operator uint32_t() const { return v; }
    FLOAT16(float f) { v = cldnn::float_to_half(f); }
    FLOAT16(size_t s) { v = cldnn::float_to_half(float(s)); }
    FLOAT16(int i) { v = cldnn::float_to_half(float(i)); }
    // TODO Below should have constructor tag to avoid ambigious behaviour, ex FLOAT16(16.f) != FLOAT16((uint16_t)16)
    explicit constexpr FLOAT16(int16_t d) : v(d) {}
    explicit constexpr FLOAT16(uint16_t d) : v(d) {}
    friend FLOAT16 operator+(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator-(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator*(const FLOAT16 &v1, const FLOAT16 &v2);
    friend FLOAT16 operator/(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator>(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator>=(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator<(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator>(const FLOAT16 &v1, const float &v2);
    friend bool operator<(const FLOAT16 &v1, const float &v2);
    friend bool operator==(const FLOAT16 &v1, const FLOAT16 &v2);
    friend bool operator!=(const FLOAT16 &v1, const FLOAT16 &v2);

    FLOAT16() { v = 0; }

    FLOAT16 &operator+=(const FLOAT16 &v1) {
        *this = (float)*this + (float)v1;
        return *this;
    }

    FLOAT16 &operator/=(const FLOAT16 &v1) {
        *this = (float)*this / (float)v1;
        return *this;
    }

    FLOAT16 &operator*=(const FLOAT16 &v1) {
        *this = (float)*this * (float)v1;
        return *this;
    }
};

inline FLOAT16 operator+(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 + (float)v2; }

inline FLOAT16 operator-(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 - (float)v2; }

inline FLOAT16 operator*(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 * (float)v2; }

inline FLOAT16 operator/(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 / (float)v2; }

inline bool operator>(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 > (float)v2; }

inline bool operator>=(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 >= (float)v2; }

inline bool operator<(const FLOAT16 &v1, const FLOAT16 &v2) { return (float)v1 < (float)v2; }

inline bool operator>(const FLOAT16 &v1, const float &v2) { return (float)v1 > v2; }

inline bool operator<(const FLOAT16 &v1, const float &v2) { return (float)v1 < v2; }

inline bool operator==(const FLOAT16 &v1, const FLOAT16 &v2) { return v1.v == v2.v; }

inline bool operator!=(const FLOAT16 &v1, const FLOAT16 &v2) { return v1.v != v2.v; }

namespace std {

template <>
struct numeric_limits<FLOAT16> {
    static constexpr FLOAT16 lowest() { return FLOAT16::lowest_val(); }
};

}  // namespace std
