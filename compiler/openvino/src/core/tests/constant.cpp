// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/type_prop.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace ngraph;
using namespace std;

//
// boolean
//

TEST(constant, boolean_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::boolean, shape, input);
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, boolean_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<string>{"1"});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, boolean_vector) {
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1, 0, 1, 0});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// float
//

TEST(constant, float_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::f32, shape, input);
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, float_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<string>{"1"});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, float_vector) {
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1, 0, 1, 0});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, float_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// double
//

TEST(constant, double_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::f64, shape, input);
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, double_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<string>{"1"});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, double_vector) {
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1, 0, 1, 0});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, double_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int4
//

TEST(constant, int4_string) {
    Shape shape{3};
    std::vector<std::string> input{"1", "0", "-1"};
    op::Constant c(element::i4, shape, input);
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], -1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x10, p[0]);
    EXPECT_EQ(0xF0, p[1] & 0xF0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int4_string_broadcast_negative_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<string>{"-1"});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], -1);
    EXPECT_EQ(v[1], -1);
    EXPECT_EQ(v[2], -1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xFF, p[0]);
    EXPECT_EQ(0xF0, p[1] & 0xF0);

    EXPECT_EQ(std::vector<std::string>(3, "-1"), c.get_value_strings());
}

TEST(constant, int4_string_broadcast_positive_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<string>{"1"});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x11, p[0]);
    EXPECT_EQ(0x10, p[1] & 0xF0);

    EXPECT_EQ(std::vector<std::string>(3, "1"), c.get_value_strings());
}

TEST(constant, int4_vector_negative_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<int8_t>{-1, -2, -1});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(-1));
    EXPECT_EQ(v[1], int8_t(-2));
    EXPECT_EQ(v[2], int8_t(-1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xFE, p[0]);
    EXPECT_EQ(0xF0, p[1] & 0xF0);
}

TEST(constant, int4_vector_positive_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<int8_t>{1, 2, 5});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(2));
    EXPECT_EQ(v[2], int8_t(5));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x12, p[0]);
    EXPECT_EQ(0x50, p[1] & 0xF0);
}

TEST(constant, int4_vector_broadcast_negative_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<int8_t>{-1});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(-1));
    EXPECT_EQ(v[1], int8_t(-1));
    EXPECT_EQ(v[2], int8_t(-1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xFF, p[0]);
    EXPECT_EQ(0xF0, p[1] & 0xF0);
}

TEST(constant, int4_vector_broadcast_positive_number) {
    Shape shape{3};
    op::Constant c(element::i4, shape, vector<int8_t>{3});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(3));
    EXPECT_EQ(v[1], int8_t(3));
    EXPECT_EQ(v[2], int8_t(3));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x33, p[0]);
    EXPECT_EQ(0x30, p[1] & 0xF0);
}

TEST(constant, int4_input_value_validation) {
    Shape shape{2};
    EXPECT_THROW(op::Constant c(element::i4, shape, 8), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::i4, shape, -9), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<int>{-9}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<int>{8}), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<int>{-9, 1}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<int>{8, 2}), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<std::string>{"-9", "1"}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::i4, shape, std::vector<std::string>{"8", "1"}), ::ngraph::CheckFailure);
}

//
// int8
//

TEST(constant, int8_string) {
    Shape shape{4};
    std::vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::i8, shape, input);
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int8_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<string>{"1"});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);

    EXPECT_EQ(std::vector<std::string>(4, "1"), c.get_value_strings());
}

TEST(constant, int8_vector) {
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1, 0, 1, 0});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int8_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int16
//

TEST(constant, int16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::i16, shape, input);
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int16_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<string>{"1"});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int16_vector) {
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1, 0, 1, 0});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int16_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int32
//

TEST(constant, int32_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::i32, shape, input);
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int32_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<string>{"1"});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int32_vector) {
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1, 0, 1, 0});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int32_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int64
//

TEST(constant, int64_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::i64, shape, input);
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int64_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<string>{"1"});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int64_vector) {
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1, 0, 1, 0});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int64_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint1
//

TEST(constant, uint1_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b10100000);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint1_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u1, shape, vector<string>{"1"});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b11110000);
}

TEST(constant, uint1_vector_less_than_single_byte) {
    Shape shape{4};
    vector<uint8_t> input{1, 0, 1, 0};
    op::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(v[i], input[i]) << "Error on index: " << i;
    }

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b10100000);
}

TEST(constant, uint1_vector_bigger_than_single_byte) {
    Shape shape{12};
    vector<uint8_t> input{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    op::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(v[i], input[i]) << "Error on index: " << i;
    }

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b10100000);
}

TEST(constant, uint1_vector_broadcast) {
    Shape shape{3};
    op::Constant c(element::u1, shape, vector<int8_t>{1});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(1));
    EXPECT_EQ(v[2], int8_t(1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xE0, p[0] & 0xE0);
}

//
// uint4
//

TEST(constant, uint4_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u4, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x10);
    EXPECT_EQ(p[1], 0x10);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint4_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u4, shape, vector<string>{"1"});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x11);
    EXPECT_EQ(p[1], 0x11);
}

TEST(constant, uint4_vector) {
    Shape shape{4};
    op::Constant c(element::u4, shape, vector<uint8_t>{1, 0, 1, 0});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x10);
    EXPECT_EQ(p[1], 0x10);
}

TEST(constant, uint4_vector_broadcast) {
    Shape shape{3};
    op::Constant c(element::u4, shape, vector<uint8_t>{1});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(1));
    EXPECT_EQ(v[2], int8_t(1));

    const auto p = c.get_data_ptr<uint8_t>();
    const auto first_byte = p[0];
    const auto second_byte = p[1] & 0xF0;
    EXPECT_EQ(0x11, first_byte);
    EXPECT_EQ(0x10, second_byte);
}

TEST(constant, uint4_input_value_validation) {
    Shape shape{2};
    EXPECT_THROW(op::Constant c(element::u4, shape, 16), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::u4, shape, -1), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<int>{-1}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<int>{16}), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<int>{-1, 1}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<int>{16, 2}), ::ngraph::CheckFailure);

    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<std::string>{"-1", "1"}), ::ngraph::CheckFailure);
    EXPECT_THROW(op::Constant c(element::u4, shape, std::vector<std::string>{"16", "1"}), ::ngraph::CheckFailure);
}

//
// uint8
//

TEST(constant, uint8_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u8, shape, input);
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint8_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<string>{"1"});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint8_vector) {
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint8_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint16
//

TEST(constant, uint16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u16, shape, input);
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint16_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<string>{"1"});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint16_vector) {
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint16_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint32
//

TEST(constant, uint32_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u32, shape, input);
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint32_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<string>{"1"});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint32_vector) {
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint32_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint64
//

TEST(constant, uint64_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::u64, shape, input);
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, uint64_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<string>{"1"});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint64_vector) {
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint64_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// bfloat16
//

TEST(constant, bfloat16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::bf16, shape, input);
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, bfloat16_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<string>{"1"});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

TEST(constant, bfloat16_vector) {
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1, 0, 1, 0});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));
}

TEST(constant, bfloat16_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

//
// float16
//

TEST(constant, float16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    op::Constant c(element::f16, shape, input);
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, float16_string_broadcast) {
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<string>{"1"});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, float16_vector) {
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1, 0, 1, 0});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));
}

TEST(constant, float16_vector_broadcast) {
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, shared_data) {
    Shape shape{100, 200};
    auto c1 = make_shared<op::Constant>(element::f16, shape, vector<float16>{123});
    auto c2 = static_pointer_cast<op::Constant>(c1->clone_with_new_inputs({}));
    const int16_t* p1 = c1->get_data_ptr<int16_t>();
    const int16_t* p2 = c2->get_data_ptr<int16_t>();
    EXPECT_EQ(p1, p2);
}

template <typename T1, typename T2>
::testing::AssertionResult test_convert() {
    Shape shape{5};
    vector<T1> expected{1, 2, 3, 4, 5};
    auto c1 = make_shared<op::Constant>(ov::element::from<T2>(), shape, expected);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Conversion failed";
    return rc;
}

TEST(constant, convert_input) {
    EXPECT_TRUE((test_convert<float, float>()));
    EXPECT_TRUE((test_convert<float, double>()));
    EXPECT_TRUE((test_convert<float, float16>()));
    EXPECT_TRUE((test_convert<float, bfloat16>()));
    EXPECT_TRUE((test_convert<float, int8_t>()));
    EXPECT_TRUE((test_convert<float, int16_t>()));
    EXPECT_TRUE((test_convert<float, int32_t>()));
    EXPECT_TRUE((test_convert<float, int64_t>()));
    EXPECT_TRUE((test_convert<float, uint8_t>()));
    EXPECT_TRUE((test_convert<float, uint16_t>()));
    EXPECT_TRUE((test_convert<float, uint32_t>()));
    EXPECT_TRUE((test_convert<float, uint64_t>()));

    EXPECT_TRUE((test_convert<double, float>()));
    EXPECT_TRUE((test_convert<double, double>()));
    EXPECT_TRUE((test_convert<double, float16>()));
    EXPECT_TRUE((test_convert<double, bfloat16>()));
    EXPECT_TRUE((test_convert<double, int8_t>()));
    EXPECT_TRUE((test_convert<double, int16_t>()));
    EXPECT_TRUE((test_convert<double, int32_t>()));
    EXPECT_TRUE((test_convert<double, int64_t>()));
    EXPECT_TRUE((test_convert<double, uint8_t>()));
    EXPECT_TRUE((test_convert<double, uint16_t>()));
    EXPECT_TRUE((test_convert<double, uint32_t>()));
    EXPECT_TRUE((test_convert<double, uint64_t>()));

    EXPECT_TRUE((test_convert<float16, float>()));
    EXPECT_TRUE((test_convert<float16, double>()));
    EXPECT_TRUE((test_convert<float16, float16>()));
    EXPECT_TRUE((test_convert<float16, bfloat16>()));
    EXPECT_TRUE((test_convert<float16, int8_t>()));
    EXPECT_TRUE((test_convert<float16, int16_t>()));
    EXPECT_TRUE((test_convert<float16, int32_t>()));
    EXPECT_TRUE((test_convert<float16, int64_t>()));
    EXPECT_TRUE((test_convert<float16, uint8_t>()));
    EXPECT_TRUE((test_convert<float16, uint16_t>()));
    EXPECT_TRUE((test_convert<float16, uint32_t>()));
    EXPECT_TRUE((test_convert<float16, uint64_t>()));

    EXPECT_TRUE((test_convert<bfloat16, float>()));
    EXPECT_TRUE((test_convert<bfloat16, double>()));
    EXPECT_TRUE((test_convert<bfloat16, float16>()));
    EXPECT_TRUE((test_convert<bfloat16, bfloat16>()));
    EXPECT_TRUE((test_convert<bfloat16, int8_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int16_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int32_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int64_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint8_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint16_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint32_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint64_t>()));

    EXPECT_TRUE((test_convert<int8_t, float>()));
    EXPECT_TRUE((test_convert<int8_t, double>()));
    EXPECT_TRUE((test_convert<int8_t, float16>()));
    EXPECT_TRUE((test_convert<int8_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int8_t, int8_t>()));
    EXPECT_TRUE((test_convert<int8_t, int16_t>()));
    EXPECT_TRUE((test_convert<int8_t, int32_t>()));
    EXPECT_TRUE((test_convert<int8_t, int64_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int16_t, float>()));
    EXPECT_TRUE((test_convert<int16_t, double>()));
    EXPECT_TRUE((test_convert<int16_t, float16>()));
    EXPECT_TRUE((test_convert<int16_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int16_t, int8_t>()));
    EXPECT_TRUE((test_convert<int16_t, int16_t>()));
    EXPECT_TRUE((test_convert<int16_t, int32_t>()));
    EXPECT_TRUE((test_convert<int16_t, int64_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int32_t, float>()));
    EXPECT_TRUE((test_convert<int32_t, double>()));
    EXPECT_TRUE((test_convert<int32_t, float16>()));
    EXPECT_TRUE((test_convert<int32_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int32_t, int8_t>()));
    EXPECT_TRUE((test_convert<int32_t, int16_t>()));
    EXPECT_TRUE((test_convert<int32_t, int32_t>()));
    EXPECT_TRUE((test_convert<int32_t, int64_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int64_t, float>()));
    EXPECT_TRUE((test_convert<int64_t, double>()));
    EXPECT_TRUE((test_convert<int64_t, float16>()));
    EXPECT_TRUE((test_convert<int64_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int64_t, int8_t>()));
    EXPECT_TRUE((test_convert<int64_t, int16_t>()));
    EXPECT_TRUE((test_convert<int64_t, int32_t>()));
    EXPECT_TRUE((test_convert<int64_t, int64_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint8_t, float>()));
    EXPECT_TRUE((test_convert<uint8_t, double>()));
    EXPECT_TRUE((test_convert<uint8_t, float16>()));
    EXPECT_TRUE((test_convert<uint8_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint8_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint16_t, float>()));
    EXPECT_TRUE((test_convert<uint16_t, double>()));
    EXPECT_TRUE((test_convert<uint16_t, float16>()));
    EXPECT_TRUE((test_convert<uint16_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint16_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint32_t, float>()));
    EXPECT_TRUE((test_convert<uint32_t, double>()));
    EXPECT_TRUE((test_convert<uint32_t, float16>()));
    EXPECT_TRUE((test_convert<uint32_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint32_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint64_t, float>()));
    EXPECT_TRUE((test_convert<uint64_t, double>()));
    EXPECT_TRUE((test_convert<uint64_t, float16>()));
    EXPECT_TRUE((test_convert<uint64_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint64_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint64_t>()));
}

template <typename T1, typename T2>
::testing::AssertionResult test_uniform_ctor() {
    Shape shape{5};
    vector<T1> expected{3, 3, 3, 3, 3};
    auto c1 = make_shared<op::Constant>(ov::element::from<T2>(), shape, 3);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Construction of uniform Constant failed";
    return rc;
}

TEST(constant, construct_uniform) {
    EXPECT_TRUE((test_uniform_ctor<float, float>()));
    EXPECT_TRUE((test_uniform_ctor<float, double>()));
    EXPECT_TRUE((test_uniform_ctor<float, float16>()));
    EXPECT_TRUE((test_uniform_ctor<float, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<float, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<double, float>()));
    EXPECT_TRUE((test_uniform_ctor<double, double>()));
    EXPECT_TRUE((test_uniform_ctor<double, float16>()));
    EXPECT_TRUE((test_uniform_ctor<double, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<double, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<float16, float>()));
    EXPECT_TRUE((test_uniform_ctor<float16, double>()));
    EXPECT_TRUE((test_uniform_ctor<float16, float16>()));
    EXPECT_TRUE((test_uniform_ctor<float16, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<bfloat16, float>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, double>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, float16>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int8_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int16_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int32_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int64_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint8_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint16_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint32_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint64_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint64_t>()));
}

TEST(constant, bad_get_data_ptr) {
    op::Constant c(element::f32, Shape{}, vector<float>{1.0});
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::f32>(), 1.0);
    try {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    } catch (const CheckFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    } catch (const CheckFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}

TEST(constant, hold_host_tensor) {
    Shape shape{4};
    void* hostDataPtr = nullptr;
    std::shared_ptr<op::Constant> constOp;
    {
        auto tensor = std::make_shared<runtime::HostTensor>(element::f32, Shape{1, 2, 3, 3});
        hostDataPtr = tensor->get_data_ptr();
        constOp = std::make_shared<op::Constant>(tensor);
    }
    const void* constDataPtr = constOp->get_data_ptr();
    ASSERT_EQ(constDataPtr, hostDataPtr);
}

// Test verifies 2 things:
// a) Checks that bitwise comparison happens on first call of 'get_all_data_elements_bitwise_identical'
// b) Next call of 'get_all_data_elements_bitwise_identical' takes already calculated value
TEST(constant, lazy_bitwise_identical) {
    auto shape = Shape{10, 1000, 1000};
    auto type = element::i32;
    auto byte_size = shape_size(shape) * sizeof(int32_t);
    auto aligned_weights_buffer = std::make_shared<ngraph::runtime::AlignedBuffer>(byte_size);
    std::memset(aligned_weights_buffer->get_ptr<char>(), 1, byte_size);
    auto weights = std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
        aligned_weights_buffer->get_ptr<char>(),
        aligned_weights_buffer->size(),
        aligned_weights_buffer);

    using namespace std::chrono;
    auto create_constant = [&]() {
        auto constant1 = std::make_shared<op::v0::Constant>(type, shape, weights);
        return constant1;
    };
    const int TIMEOUT_MS = 300;
    size_t created_count = 0;
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            create_constant();  // shall be O(1)
            created_count++;
        }
    }
    size_t bitwise_check_count = 0;
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            auto constant1 = create_constant();
            EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // can be O(N)
            bitwise_check_count++;
        }
    }

    size_t bitwise_check_count_only = 0;
    auto constant1 = create_constant();
    EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // first time calculation can be O(N)
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // next calls shall be O(1)
            bitwise_check_count_only++;
        }
    }
    std::cout << "Created: " << created_count << ", Created+Checked=" << bitwise_check_count
              << ", Checked_cached_value=" << bitwise_check_count_only << "\n";
    // Comparing creation from pre-allocated buffer with creation + checking identical
    // '10' times is guaranteed to be faster here (typical value is ~10'000)
    EXPECT_GT(created_count, bitwise_check_count * 10);

    // Comparing getting comparison value from cache with first-time calculation
    // '10' times is guaranteed to be faster here (typical value is ~200'000)
    EXPECT_GT(bitwise_check_count_only, bitwise_check_count * 10);
}

// Disabled just because of long execution time. Enable for nightly builds in future
TEST(constant, DISABLED_nightly_huge_size_4GB) {
    uint64_t start = 1llu << 32;
    uint64_t s = start + 5;
    std::vector<uint8_t> data(s);
    for (uint64_t i = start; i < s; i++) {
        data[i] = static_cast<uint8_t>(i - start + 42);
    }
    Shape shape{static_cast<Shape::size_type>(s)};
    op::Constant c(element::u8, shape, data.data());
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    for (uint64_t i = start; i < s; i++) {
        EXPECT_EQ(v[i], i - start + 42) << i << " failed";
    }
}
