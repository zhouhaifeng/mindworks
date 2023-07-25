// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//todo move to another folder

#pragma once

#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/tensor.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/execution_config.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <intel_gpu/graph/program.hpp>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/primitives/primitive.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/lrn.hpp>
#include <intel_gpu/primitives/roi_pooling.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/normalize.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/graph/serialization/utils.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "float16.h"
#include "random_gen.h"
#include "uniform_quantized_real_distribution.hpp"
#include "to_string_utils.h"
#include "program_node.h"

#include <iostream>
#include <limits>
#include <random>
#include <algorithm>
#include <memory>
#include <chrono>

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))

namespace cldnn {
template <>
struct type_to_data_type<FLOAT16> {
    static constexpr data_types value = data_types::f16;
};
}  // namespace cldnn

namespace tests {

std::shared_ptr<cldnn::engine> create_test_engine();
cldnn::engine& get_test_engine();
cldnn::stream_ptr get_test_stream_ptr(cldnn::ExecutionConfig cfg);
cldnn::stream_ptr get_test_stream_ptr();
cldnn::stream& get_test_stream();

// Set default configuration for test-cases
cldnn::ExecutionConfig get_test_default_config(const cldnn::engine&);
cldnn::ExecutionConfig get_test_default_config(const cldnn::engine&, ov::AnyMap::value_type values);
cldnn::ExecutionConfig get_test_default_config(const cldnn::engine&,
                                                std::initializer_list<ov::AnyMap::value_type> values);


template<typename T>
bool has_node_with_type(cldnn::program& prog) {
    for (auto node : prog.get_processing_order()) {
        if (node->is_type<T>())
            return true;
    }

    return false;
}

inline bool has_node(cldnn::program& prog, cldnn::primitive_id id) {
    for (auto node : prog.get_processing_order()) {
        if (node->id() == id)
            return true;
    }

    return false;
}

#define USE_RANDOM_SEED 0
#if USE_RANDOM_SEED
    std::random_device rnd_device;
    unsigned int const random_seed = rnd_device();
#else
    unsigned int const random_seed = 1337;
#endif

// rounds floating point number, fraction precision should be in the range [0,23]
// masks the bits:
// 1 11111111 11111111111111100000000
// |      |            |
// sign  exp        fraction
inline float float_round(float x, size_t fraction_precision = 15) {
    uint32_t mask = ~((1 << (23 - fraction_precision)) - 1);
    reinterpret_cast<uint32_t&>(x) &= mask;
    return x;
}

template<typename T>
using VF = std::vector<T>;        // float vector
template<typename T>
using VVF = std::vector<VF<T>>;        // feature map
template<typename T>
using VVVF = std::vector<VVF<T>>;        // 3d feature map
template<typename T>
using VVVVF = std::vector<VVVF<T>>;    // batch of 3d feature maps
template<typename T>
using VVVVVF = std::vector<VVVVF<T>>;    // split of bfyx filters
template<typename T>
using VVVVVVF = std::vector<VVVVVF<T>>;    // split of bfyx filters

template<typename T>
inline VF<T> flatten_4d(cldnn::format input_format, VVVVF<T> &data) {
    size_t a = data.size();
    size_t b = data[0].size();
    size_t c = data[0][0].size();
    size_t d = data[0][0][0].size();
    VF<T> vec(a * b * c * d, (T)(0.0f));
    size_t idx = 0;

    switch (input_format.value) {
        case cldnn::format::yxfb:
            for (size_t yi = 0; yi < c; ++yi)
                for (size_t xi = 0; xi < d; ++xi)
                    for (size_t fi = 0; fi < b; ++fi)
                        for (size_t bi = 0; bi < a; ++bi)
                            vec[idx++] = data[bi][fi][yi][xi];
            break;

        case cldnn::format::fyxb:
            for (size_t fi = 0; fi < b; ++fi)
                for (size_t yi = 0; yi < c; ++yi)
                    for (size_t xi = 0; xi < d; ++xi)
                        for (size_t bi = 0; bi < a; ++bi)
                            vec[idx++] = data[bi][fi][yi][xi];
            break;

        case cldnn::format::bfyx:
            for (size_t bi = 0; bi < a; ++bi)
                for (size_t fi = 0; fi < b; ++fi)
                    for (size_t yi = 0; yi < c; ++yi)
                        for (size_t xi = 0; xi < d; ++xi)
                            vec[idx++] = data[bi][fi][yi][xi];
            break;

        case cldnn::format::byxf:
            for (size_t bi = 0; bi < a; ++bi)
                for (size_t yi = 0; yi < c; ++yi)
                    for (size_t xi = 0; xi < d; ++xi)
                        for (size_t fi = 0; fi < b; ++fi)
                            vec[idx++] = data[bi][fi][yi][xi];
            break;

        default:
            assert(0);
    }
    return vec;
}

template<typename T>
inline VF<T> flatten_5d(cldnn::format input_format, VVVVVF<T> &data) {
    size_t a = data.size();
    size_t b = data[0].size();
    size_t c = data[0][0].size();
    size_t d = data[0][0][0].size();
    size_t e = data[0][0][0][0].size();
    VF<T> vec(a * b * c * d * e, (T)(0.0f));
    size_t idx = 0;

    switch (input_format.value) {
        case cldnn::format::bfzyx:
            for (size_t bi = 0; bi < a; ++bi)
                for (size_t fi = 0; fi < b; ++fi)
                    for (size_t zi = 0; zi < c; ++zi)
                        for (size_t yi = 0; yi < d; ++yi)
                            for (size_t xi = 0; xi < e; ++xi)
                                vec[idx++] = data[bi][fi][zi][yi][xi];
            break;

        default:
            assert(0);
    }
    return vec;
}

template<typename T>
inline VF<T> flatten_6d(cldnn::format input_format, VVVVVVF<T> &data) {
    size_t a = data.size();
    size_t b = data[0].size();
    size_t c = data[0][0].size();
    size_t d = data[0][0][0].size();
    size_t e = data[0][0][0][0].size();
    size_t f = data[0][0][0][0][0].size();
    VF<T> vec(a * b * c * d * e * f, (T)(0.0f));
    size_t idx = 0;

    switch (input_format.value) {
        case cldnn::format::bfwzyx:
            for (size_t bi = 0; bi < a; ++bi)
                for (size_t fi = 0; fi < b; ++fi)
                    for (size_t wi = 0; wi < c; ++wi)
                        for (size_t zi = 0; zi < d; ++zi)
                            for (size_t yi = 0; yi < e; ++yi)
                                for (size_t xi = 0; xi < f; ++xi)
                                    vec[idx++] = data[bi][fi][wi][zi][yi][xi];
            break;
        default:
            assert(0);
    }
    return vec;
}

template <class T> void set_value(T* ptr, uint32_t index, T value) { ptr[index] = value; }
template <class T> T    get_value(T* ptr, uint32_t index) { return ptr[index]; }

template<typename T>
void set_values(cldnn::memory::ptr mem, std::initializer_list<T> args) {
    cldnn::mem_lock<T> ptr(mem, get_test_stream());

    auto it = ptr.begin();
    for(auto x : args)
        *it++ = x;
}

template<typename T>
void set_values(cldnn::memory::ptr mem, std::vector<T> args) {
    cldnn::mem_lock<T> ptr(mem, get_test_stream());

    auto it = ptr.begin();
    for (auto x : args)
        *it++ = x;
}

template<typename T>
void set_values_per_batch_and_feature(cldnn::memory::ptr mem, std::vector<T> args) {
    cldnn::mem_lock<T> mem_ptr(mem, get_test_stream());
    auto&& pitches = mem->get_layout().get_pitches();
    auto&& l = mem->get_layout();
    for (cldnn::tensor::value_type b = 0; b < l.batch(); ++b) {
        for (cldnn::tensor::value_type f = 0; f < l.feature(); ++f) {
            for (cldnn::tensor::value_type y = 0; y < l.spatial(1); ++y) {
                for (cldnn::tensor::value_type x = 0; x < l.spatial(0); ++x) {
                    unsigned int input_it = b*pitches.batch[0] + f*pitches.feature[0] + y*pitches.spatial[1] + x*pitches.spatial[0];
                    mem_ptr[input_it] = args[b*l.feature() + f];
                }
            }
        }
    }

}

template<typename T, typename std::enable_if<std::is_floating_point<T>::value ||
                                             std::is_same<T, FLOAT16>::value>::type* = nullptr>
void set_random_values(cldnn::memory::ptr mem, bool sign = false, unsigned significand_bit = 8, unsigned scale = 1)
{
    cldnn::mem_lock<T> ptr(mem, get_test_stream());

    std::mt19937 gen;
    for (auto it = ptr.begin(); it != ptr.end(); ++it) {
        *it = rnd_generators::gen_number<T>(gen, significand_bit, sign, false, scale);
    }
}

template<class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void set_random_values(cldnn::memory::ptr mem)
{
    using T1 = typename std::conditional<std::is_same<int8_t, T>::value, int, T>::type;
    using T2 = typename std::conditional<std::is_same<uint8_t, T1>::value, unsigned int, T1>::type;

    cldnn::mem_lock<T> ptr(mem, get_test_stream());

    std::mt19937 gen;
    static std::uniform_int_distribution<T2> uid(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (auto it = ptr.begin(); it != ptr.end(); ++it) {
        *it = static_cast<T>(uid(gen));
    }
}

// Tries to construct a network, checking if an expected error appears
inline void check_exception_massage(cldnn::engine& engine, cldnn::topology& topology, std::string msg_to_find) {
    try {
        cldnn::network(engine, topology);
    } catch (std::exception & exc) {
        std::string msg(exc.what());
        if (msg.find(msg_to_find) != std::string::npos) {
            throw;
        } else {
            printf("%s\n", exc.what());
        }
    }
}

// Checks equality of floats.
// For values less than absoulte_error_limit, absolute error will be counted
// for others, the relatve error will be counted.
// Function returns false if error will exceed the threshold.
// Default values:
// relative_error_threshold = 1e-3
// absolute_error_threshold = 1e-6
// absolute_error_limit = 1e-4
inline bool are_equal(
    const float ref_item,
    const float item,
    const float relative_error_threshold = 1e-3,
    const float absolute_error_threshold = 1e-6,
    const float absolute_error_limit     = 1e-4) {

        if( fabs(item) < absolute_error_limit) {
            if(fabs( item - ref_item ) > absolute_error_threshold) {
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl;
                return false;
            }
        } else
            if(fabs(item - ref_item) / fabs(ref_item) > relative_error_threshold){
                std::cout << "Ref val: " << ref_item << "\tSecond val: " << item << std::endl;
                return false;
        }

        return true;
}

inline bool floating_point_equal(FLOAT16 x, FLOAT16 y, int max_ulps_diff = 4) {
    int16_t sign_bit_mask = 1;
    sign_bit_mask <<= 15;
    int16_t a = x.v, b = y.v;
    if ((a & sign_bit_mask) != (b & sign_bit_mask)) {
        a &= ~sign_bit_mask;
        b &= ~sign_bit_mask;
        return a == 0 && b == 0;
    }
    else {
        return std::abs(a - b) < (1 << (max_ulps_diff));
    }
}

inline bool floating_point_equal(float x, float y, int max_ulps_diff = 4) {
    int32_t sign_bit_mask = 1;
    sign_bit_mask <<= 31;
    int32_t a = reinterpret_cast<int32_t&>(x), b = reinterpret_cast<int32_t&>(y);
    if ((a & sign_bit_mask) != (b & sign_bit_mask)) {
        a &= ~sign_bit_mask;
        b &= ~sign_bit_mask;
        return a == 0 && b == 0;
    }
    else {
        return std::abs(a - b) < (1 << (max_ulps_diff));
    }
}

class test_params {
public:

    test_params() : fmt(cldnn::format::bfyx) { }

    test_params(cldnn::data_types dt, cldnn::format input_format, int32_t batch_size, int32_t feature_size, cldnn::tensor input_size, cldnn::ExecutionConfig config = {}) :
        data_type(dt),
        fmt(input_format),
        network_config(config) {
        cldnn::tensor t = cldnn::tensor(batch_size, feature_size, input_size.spatial[0],  input_size.spatial[1] );
        input_layouts.push_back( cldnn::layout(dt, fmt, t) );
    }

    cldnn::data_types data_type;
    cldnn::format fmt;
    std::vector<cldnn::layout> input_layouts;

    void * opaque_custom_param = nullptr;

    cldnn::ExecutionConfig network_config;

    std::string print();
    static std::string print_tensor(cldnn::tensor tensor);
};

struct pitches {
    size_t b, f, y, x, z;
};

struct memory_desc {
    pitches pitch;
    size_t offset;
};

struct test_dump {
    const std::string name() const;
    const std::string test_case_name() const;

private:
    const std::string test_case_name_str = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name();
    const std::string name_str = ::testing::UnitTest::GetInstance()->current_test_info()->name();
};

class generic_test : public ::testing::TestWithParam<std::tuple<std::shared_ptr<test_params>, std::shared_ptr<cldnn::primitive>>> {
public:
    generic_test();

    void run_single_test(bool is_caching_test = false);

    template<typename Type>
    void compare_buffers(const cldnn::memory::ptr out, const cldnn::memory::ptr ref);

    static size_t get_linear_index(const cldnn::layout & layout, size_t b, size_t f, size_t y, size_t x, const memory_desc& desc);
    static size_t get_linear_index(const cldnn::layout & layout, size_t b, size_t f, size_t z, size_t y, size_t x, const memory_desc& desc);
    static size_t get_linear_index_with_broadcast(const cldnn::layout& in_layout, size_t b, size_t f, size_t y, size_t x, const memory_desc& desc);

    static memory_desc get_linear_memory_desc(const cldnn::layout & layout);

    static std::vector<std::shared_ptr<test_params>> generate_generic_test_params(std::vector<std::shared_ptr<test_params>>& all_generic_params);

    virtual bool is_format_supported(cldnn::format format) = 0;

    virtual cldnn::tensor get_expected_output_tensor();

    struct custom_param_name_functor {
            std::string operator()(const ::testing::TestParamInfo<std::tuple<std::shared_ptr<test_params>, std::shared_ptr<cldnn::primitive>>>& info) {
                    return std::to_string(info.index);
            }
    };

    static cldnn::format get_plain_format_for(const cldnn::format);

protected:
    cldnn::engine& engine = get_test_engine();
    std::shared_ptr<test_params> generic_params;
    test_dump test_info;
    std::shared_ptr<cldnn::primitive> layer_params;
    int max_ulps_diff_allowed; //Max number of ulps allowed between 2 values when comparing the output buffer and the reference buffer.
    bool random_values; // if set memory buffers will be filled with random values
    bool dump_memory; // if set memory buffers will be dumped to file
    virtual cldnn::memory::ptr generate_reference(const std::vector<cldnn::memory::ptr>& inputs) = 0;
    // Allows the test to override the random input data that the framework generates

    virtual void prepare_input_for_test(std::vector<cldnn::memory::ptr>& /*inputs*/) { }

    static std::vector<cldnn::data_types> test_data_types();
    static std::vector<cldnn::format> test_input_formats;
    static std::vector<cldnn::format> test_weight_formats;
    static std::vector<int32_t> test_batch_sizes;
    static std::vector<int32_t> test_feature_sizes;
    static std::vector<cldnn::tensor> test_input_sizes;
};

// When a test assertion such as EXPECT_EQ fails, Google-Test prints the argument values to help with debugging.
// It does this using a user - extensible value printer.
// This function will be used to print the test params in case of an error.
inline void PrintTupleTo(const std::tuple<std::shared_ptr<test_params>, std::shared_ptr<cldnn::primitive>>& t, ::std::ostream* os) {
    std::stringstream str;

    auto test_param = std::get<0>(t);
    auto primitive = std::get<1>(t);

    str << std::endl << "Test params: " << test_param->print();

    str << "Layer params:\n"
        << "Output padding lower size: " << test_param->print_tensor(primitive->output_paddings[0].lower_size())
        << " upper size: " << test_param->print_tensor(primitive->output_paddings[0].upper_size()) << '\n';

    //TODO: do layers not have param dumping? we could consider adding it

    if (primitive->type == cldnn::concatenation::type_id()) {
        auto dc = std::static_pointer_cast<cldnn::concatenation>(primitive);
        (void)dc;
    } else if(primitive->type == cldnn::lrn::type_id()) {
        auto lrn = std::static_pointer_cast<cldnn::lrn >(primitive);
        std::string norm_region = (lrn->norm_region == cldnn::lrn_norm_region_across_channel) ? "across channel" : "within channel";
        str << "Norm region: " << norm_region
            << " Size: " << lrn->size
            << " Alpha: " << lrn->alpha
            << " Beta: " << lrn->beta
            << " K: " << lrn->k;
    } else if(primitive->type == cldnn::roi_pooling::type_id()) {
        auto p = std::static_pointer_cast<cldnn::roi_pooling >(primitive);
        str << "Pooling mode: " << (p->mode == cldnn::pooling_mode::max ? "MAX" : "AVG")
            << " Pooled width: " << p->pooled_width
            << " Pooled height: " << p->pooled_height
            << " Spatial scale: " << p->spatial_scale
            << " Spatial bins x: " << p->spatial_bins_x
            << " Spatial bins y: " << p->spatial_bins_y
            << " Output dim: " << p->output_dim;
    } else if(primitive->type == cldnn::softmax::type_id()) {
        auto sm = std::static_pointer_cast<cldnn::softmax>(primitive);
        (void)sm;
    } else if (primitive->type == cldnn::reorder::type_id()) {
        auto reorder = std::static_pointer_cast<cldnn::reorder>(primitive);
        str << "Output data type: " << cldnn::data_type_traits::name(*reorder->output_data_types[0]) << " Mean: " << reorder->mean << "Subtract per feature: " << "TODO" /*std::vector<float> subtract_per_feature*/;
    } else if (primitive->type == cldnn::normalize::type_id()) {
        auto normalize = std::static_pointer_cast<cldnn::normalize>(primitive);
        std::string norm_region = normalize->across_spatial ? "across_spatial" : "within_spatial";
        str << "Norm region: " << norm_region << " Epsilon: " << normalize->epsilon << " Scale input id: " << normalize->scale_input;
    } else if (primitive->type == cldnn::convolution::type_id()) {
        auto convolution = std::static_pointer_cast<cldnn::convolution>(primitive);
        str << "Stride x: " << convolution->stride[1] << " Stride y: " << convolution->stride[0]
            << " Dilation x: " << convolution->dilation[1] << " Dilation y: " << convolution->dilation[0]
            << " Pad x: " << convolution->padding_begin[1] << " Pad y: " << convolution->padding_begin[0];
    } else if (primitive->type == cldnn::activation::type_id()) {
        auto activation = std::static_pointer_cast<cldnn::activation>(primitive);
        str << "Negative slope: " << activation->additional_params.a << " Negative slope input id: " << activation->additional_params_input;
    } else if (primitive->type == cldnn::pooling::type_id()) {
        auto pooling = std::static_pointer_cast<cldnn::pooling>(primitive);
        std::string pooling_mode = (pooling->mode == cldnn::pooling_mode::max) ? "max" : "average";
        str << "Pooling mode: " << pooling_mode
            << " Pads_begin x: " << pooling->pads_begin[1] << " Pads_begin y: " << pooling->pads_begin[0]
            << " Pads_end x: " << pooling->pads_end[1] << " Pads_end y: " << pooling->pads_end[0]
            << " Stride x: " << pooling->stride[1] << " Stride y: " << pooling->stride[0]
            << " Size x: " << pooling->size[1] << " Size y: " << pooling->size[0];
    } else {
        throw std::runtime_error("Not implemented yet for this primitive.");
    }

    *os << str.str();
}

template <typename T, typename U>
T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <class T>
std::vector<float> get_output_values_to_float(cldnn::network& net, const cldnn::network_output& output, size_t max_cnt = std::numeric_limits<size_t>::max()) {
    std::vector<float> ret;
    auto ptr = output.get_memory();
    cldnn::mem_lock<T, cldnn::mem_lock_type::read> mem(ptr, net.get_stream());
    if (ptr->get_layout().data_type != cldnn::type_to_data_type<T>::value)
        OPENVINO_THROW("target type ", cldnn::data_type_traits::name(cldnn::type_to_data_type<T>::value),
                       " mismatched with actual type ", cldnn::data_type_traits::name(ptr->get_layout().data_type));
    for (size_t i = 0; i < std::min(max_cnt, ptr->get_layout().count()); i++)
        ret.push_back(mem[i]);
    return ret;
}

inline std::vector<float> get_output_values_to_float(cldnn::network& net, const cldnn::network_output& output, size_t max_cnt = std::numeric_limits<size_t>::max()) {
    switch(output.get_layout().data_type){
        case cldnn::data_types::f16:
            return get_output_values_to_float<FLOAT16>(net, output, max_cnt);
        case cldnn::data_types::f32:
            return get_output_values_to_float<float>(net, output, max_cnt);
        case cldnn::data_types::i8:
            return get_output_values_to_float<int8_t>(net, output, max_cnt);
        case cldnn::data_types::u8:
            return get_output_values_to_float<uint8_t>(net, output, max_cnt);
        case cldnn::data_types::i32:
            return get_output_values_to_float<int32_t>(net, output, max_cnt);
        case cldnn::data_types::i64:
            return get_output_values_to_float<int64_t>(net, output, max_cnt);
        default:
            OPENVINO_THROW( "Unknown output data_type");
    }
}

double default_tolerance(cldnn::data_types dt);
// inline void print_bin_blob(cldnn::memory& mem, std::string name)
// {
//     auto&& size = mem.get_layout().get_tensor();

//     std::cerr << name;
//     std::cerr << " shape: ";
//     std::cerr << size.batch[0] << " ";
//     std::cerr << size.feature[0] << " ";
//     std::cerr << size.spatial[1] << " ";
//     std::cerr << size.spatial[0] << " ";
//     std::cerr << "(" << size.batch[0] * size.feature[0] * size.spatial[1] * size.spatial[0] << ")" << std::endl;

//     auto mem_ptr = mem.pointer<uint32_t>();

//     bool packed_ic = mem.get_layout().format == cldnn::format::b_fs_yx_32fp ? 1 : 0;
//     int B = size.batch[0];
//     int C = size.feature[0];
//     int H = size.spatial[1];
//     int W = size.spatial[0];

//     for (cldnn::tensor::value_type b = 0; b < B; ++b)
//     {
//         for (cldnn::tensor::value_type f = 0; f < C; ++f)
//         {
//             for (cldnn::tensor::value_type y = 0; y < H; ++y)
//             {
//                 for (cldnn::tensor::value_type x = 0; x < W; ++x)
//                 {
//                     if (!packed_ic)
//                     {
//                         size_t input_it = b * C*H*W + f * W*H + y * W + x;
//                         size_t elem = input_it / 32;
//                         size_t bit = input_it % 32;
//                         std::cerr << ((mem_ptr[elem] & (1 << bit)) >> bit) << " ";
//                     }
//                     else
//                     {
//                         size_t input_it = b * (C / 32)*W*H + (f / 32)*W*H + y * W + x;
//                         size_t bit = f % 32;
//                         std::cerr << ((mem_ptr[input_it] & (1 << bit)) >> bit) << " ";
//                     }
//                 }
//                 std::cerr << std::endl;
//             }
//             std::cerr << std::endl;
//         }
//         std::cerr << "==============" << std::endl;
//     }
// }

// inline void print_bin_blob_packed(cldnn::memory& mem, std::string name)
// {
//     auto&& size = mem.get_layout().get_tensor();

//     std::cerr << name;
//     std::cerr << " shape: ";
//     std::cerr << size.batch[0] << " ";
//     std::cerr << size.feature[0] << " ";
//     std::cerr << size.spatial[1] << " ";
//     std::cerr << size.spatial[0] << " ";
//     std::cerr << "(" << size.batch[0] * size.feature[0] * size.spatial[1] * size.spatial[0] << ")" << std::endl;

//     auto mem_ptr = mem.pointer<uint32_t>();

//     int B = size.batch[0];
//     int C = size.feature[0];
//     int H = size.spatial[1];
//     int W = size.spatial[0];

//     for (cldnn::tensor::value_type b = 0; b < B; ++b)
//     {
//         for (cldnn::tensor::value_type f = 0; f < div_up(C, 32); ++f)
//         {
//             for (cldnn::tensor::value_type y = 0; y < H; ++y)
//             {
//                 for (cldnn::tensor::value_type x = 0; x < W; ++x)
//                 {
//                     size_t input_it = b * div_up(C, 32)*W*H + f * W*H + y * W + x;
//                     std::cerr << mem_ptr[input_it] << " ";
//                 }
//                 std::cerr << std::endl;
//             }
//             std::cerr << std::endl;
//         }
//         std::cerr << "==============" << std::endl;
//     }
// }

// inline void print_blob(cldnn::memory& mem, std::string name)
// {
//     auto&& size = mem.get_layout().get_tensor();

//     std::cerr << name;
//     std::cerr << " shape: ";
//     std::cerr << size.batch[0] << " ";
//     std::cerr << size.feature[0] << " ";
//     std::cerr << size.spatial[1] << " ";
//     std::cerr << size.spatial[0] << " ";
//     std::cerr << "(" << size.batch[0] * size.feature[0] * size.spatial[1] * size.spatial[0] << ")" << std::endl;

//     auto mem_ptr = mem.pointer<float>();

//     int B = size.batch[0];
//     int C = size.feature[0];
//     int H = size.spatial[1];
//     int W = size.spatial[0];

//     for (cldnn::tensor::value_type b = 0; b < B; ++b)
//     {
//         for (cldnn::tensor::value_type f = 0; f < C; ++f)
//         {
//             for (cldnn::tensor::value_type y = 0; y < H; ++y)
//             {
//                 for (cldnn::tensor::value_type x = 0; x < W; ++x)
//                 {
//                     size_t input_it = b * C*W*H + f * W*H + y * W + x;
//                     std::cerr << std::setw(4) << mem_ptr[input_it] << " ";
//                 }
//                 std::cerr << std::endl;
//             }
//             std::cerr << std::endl;
//         }
//         std::cerr << "==============" << std::endl;
//     }
// }

inline cldnn::network::ptr get_network(cldnn::engine& engine,
                                cldnn::topology& topology,
                                const ov::intel_gpu::ExecutionConfig& config,
                                cldnn::stream::ptr stream,
                                const bool is_caching_test) {
    cldnn::network::ptr network;
    if (is_caching_test) {
        cldnn::membuf mem_buf;
        {
            cldnn::network _network(engine, topology, config);
            std::ostream out_mem(&mem_buf);
            cldnn::BinaryOutputBuffer ob = cldnn::BinaryOutputBuffer(out_mem);
            _network.save(ob);
        }
        {
            std::istream in_mem(&mem_buf);
            cldnn::BinaryInputBuffer ib = cldnn::BinaryInputBuffer(in_mem, engine);
            network = std::make_shared<cldnn::network>(ib, config, stream, engine, true, 0);
        }
    } else {
        network = std::make_shared<cldnn::network>(engine, topology, config);
    }

    return network;
}

} // namespace tests
