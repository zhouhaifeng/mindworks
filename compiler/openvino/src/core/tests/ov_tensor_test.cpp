// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-spec-builders.h>
#include <gmock/gmock.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <openvino/core/shape.hpp>
#include <openvino/core/strides.hpp>
#include <openvino/core/type/element_type.hpp>

#include "ngraph/coordinate_transform.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

using OVTensorTest = ::testing::Test;

inline ov::Strides byteStrides(const ov::Strides& strides, const ov::element::Type& type) {
    ov::Strides byte_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i)
        byte_strides[i] = strides[i] * type.size();
    return byte_strides;
}

TEST_F(OVTensorTest, canCreateTensor) {
    ov::Shape shape = {4, 3, 2};
    ov::Tensor t{ov::element::f32, shape};
    const std::size_t totalSize = ov::shape_size(shape);
    ASSERT_EQ(totalSize, t.get_size());
    ASSERT_NE(nullptr, t.data());
    ASSERT_EQ(ov::element::f32, t.get_element_type());
    ASSERT_EQ(shape, t.get_shape());
    ASSERT_NE(shape, t.get_strides());
    ASSERT_EQ(byteStrides(ov::Strides({6, 2, 1}), t.get_element_type()), t.get_strides());
    ASSERT_EQ(ov::element::f32.size() * totalSize, t.get_byte_size());
    ASSERT_THROW(t.data(ov::element::i64), ov::Exception);
    ASSERT_THROW(t.data<std::int32_t>(), ov::Exception);
}

TEST_F(OVTensorTest, createTensorFromPort) {
    auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{1, 3, 2, 2});
    auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    auto parameter3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    float data[] = {5.f, 6.f, 7.f};
    ov::Tensor t1{parameter1->output(0)};
    ov::Tensor t2{parameter2->output(0), data};
    ov::Tensor t3{parameter3->output(0)};
    ov::Tensor t4{parameter3->output(0), data};

    EXPECT_EQ(t1.get_shape(), parameter1->get_shape());
    EXPECT_EQ(t1.get_element_type(), parameter1->get_element_type());
    EXPECT_EQ(t2.get_shape(), parameter2->get_shape());
    EXPECT_EQ(t2.get_element_type(), parameter2->get_element_type());
    EXPECT_EQ(t3.get_shape(), ov::Shape{0});
    EXPECT_EQ(t3.get_element_type(), parameter3->get_element_type());
    EXPECT_EQ(t4.get_shape(), ov::Shape{0});
    EXPECT_EQ(t4.get_element_type(), parameter3->get_element_type());
}

TEST_F(OVTensorTest, canAccessF16Tensor) {
    ov::Shape shape = {4, 3, 2};
    ov::Tensor t{ov::element::f16, shape};
    EXPECT_NE(nullptr, t.data());
    EXPECT_NO_THROW(t.data(ov::element::f16));
    EXPECT_NO_THROW(t.data<ov::float16>());
    EXPECT_THROW(t.data<ov::bfloat16>(), ov::Exception);
    EXPECT_THROW(t.data<std::uint16_t>(), ov::Exception);
    EXPECT_THROW(t.data<std::int16_t>(), ov::Exception);
}

TEST_F(OVTensorTest, emptySize) {
    ov::Tensor t(ov::element::f32, {0});
    ASSERT_NE(nullptr, t.data());
}

TEST_F(OVTensorTest, operators) {
    ov::Tensor t;
    ASSERT_FALSE(t);
    ASSERT_TRUE(!t);
}

OPENVINO_SUPPRESS_DEPRECATED_START
class OVMockAllocatorImpl : public ov::AllocatorImpl {
public:
    MOCK_METHOD(void*, allocate, (size_t, size_t), ());
    MOCK_METHOD(void, deallocate, (void*, size_t, size_t), ());                  // NOLINT(readability/casting)
    MOCK_METHOD(bool, is_equal, (const ov::AllocatorImpl&), (const, noexcept));  // NOLINT(readability/casting)
};

OPENVINO_SUPPRESS_DEPRECATED_START
TEST_F(OVTensorTest, canCreateTensorUsingMockAllocatorImpl) {
    ov::Shape shape = {1, 2, 3};
    auto allocator = std::make_shared<OVMockAllocatorImpl>();

    EXPECT_CALL(*allocator, allocate(::testing::_, ::testing::_))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator, deallocate(::testing::_, ::testing::_, ::testing::_)).Times(1);

    { ov::Tensor t{ov::element::f32, shape, ov::Allocator{allocator}}; }
}
OPENVINO_SUPPRESS_DEPRECATED_END

struct OVMockAllocator {
    struct Impl {
        MOCK_METHOD(void*, allocate, (size_t, size_t), ());
        MOCK_METHOD(void, deallocate, (void*, size_t, size_t), ());
        MOCK_METHOD(bool, is_equal, (const Impl&), (const, noexcept));
    };
    OVMockAllocator() : impl{std::make_shared<Impl>()} {}

    void* allocate(size_t b, size_t a) {
        return impl->allocate(b, a);
    }

    void deallocate(void* ptr, size_t b, size_t a) {
        impl->deallocate(ptr, b, a);
    }
    bool is_equal(const OVMockAllocator& other) const {
        return impl->is_equal(*other.impl);
    }

    std::shared_ptr<Impl> impl;
};

TEST_F(OVTensorTest, canCreateTensorUsingMockAllocator) {
    ov::Shape shape = {1, 2, 3};
    OVMockAllocator allocator;

    EXPECT_CALL(*allocator.impl, allocate(::testing::_, ::testing::_))
        .WillRepeatedly(testing::Return(reinterpret_cast<void*>(1)));
    EXPECT_CALL(*allocator.impl, deallocate(::testing::_, ::testing::_, ::testing::_)).Times(1);

    { ov::Tensor t{ov::element::f32, shape, allocator}; }
}

TEST_F(OVTensorTest, canAccessExternalData) {
    ov::Shape shape = {1, 1, 3};
    float data[] = {5.f, 6.f, 7.f};
    ov::Tensor t{ov::element::f32, shape, data};
    {
        float* ptr = t.data<float>();
        ASSERT_EQ(ptr[2], 7);
        ASSERT_EQ(data, t.data(ov::element::f32));
        ASSERT_EQ(data, ptr);
        ASSERT_THROW(t.data<std::int16_t>(), ov::Exception);
        ASSERT_EQ(byteStrides(ov::row_major_strides(shape), t.get_element_type()), t.get_strides());
        ASSERT_EQ(ov::shape_size(shape), t.get_size());
        ASSERT_EQ(ov::shape_size(shape) * ov::element::f32.size(), t.get_byte_size());
    }
}

TEST_F(OVTensorTest, canAccessExternalDataWithStrides) {
    ov::Shape shape = {2, 3};
    float data[] = {5.f, 6.f, 7.f, 0.f, 1.f, 42.f, 3.f, 0.f};
    ov::Tensor t{ov::element::f32, shape, data, {16, 4}};
    ASSERT_EQ(ov::Strides({16, 4}), t.get_strides());
    {
        ASSERT_EQ((ov::Shape{2, 3}), t.get_shape());
        const float* ptr = t.data<const float>();
        ASSERT_EQ(ptr[5], 42);
    }
}

TEST_F(OVTensorTest, cannotCreateTensorWithExternalNullptr) {
    ov::Shape shape = {2, 3};
    ASSERT_THROW(ov::Tensor(ov::element::f32, shape, nullptr), ov::Exception);
}

TEST_F(OVTensorTest, cannotCreateTensorWithWrongStrides) {
    ov::Shape shape = {2, 3};
    float data[] = {5.f, 6.f, 7.f, 0.f, 1.f, 42.f, 3.f, 0.f};
    const auto el = ov::element::f32;
    {
        // strides.size() != shape.size()
        EXPECT_THROW(ov::Tensor(el, shape, data, byteStrides({6, 3, 1}, el)), ov::Exception);
    }
    {
        // strides values are element-wise >= ov::row_major_strides(shape) values
        EXPECT_THROW(ov::Tensor(el, shape, data, byteStrides({2, 1}, el)), ov::Exception);
        EXPECT_THROW(ov::Tensor(el, shape, data, byteStrides({3, 0}, el)), ov::Exception);
        EXPECT_THROW(ov::Tensor(el, shape, data, byteStrides({3, 2}, el)), ov::Exception);
        EXPECT_NO_THROW(ov::Tensor(el, shape, data, byteStrides({6, 2}, el)));
    }
    {
        // strides are not divisible by elem_size
        EXPECT_THROW(ov::Tensor(el, shape, data, {7, el.size()}), ov::Exception);
        EXPECT_THROW(ov::Tensor(el, shape, data, {3, 0}), ov::Exception);
        EXPECT_THROW(ov::Tensor(el, shape, data, {el.size(), 3}), ov::Exception);
    }
}

TEST_F(OVTensorTest, saveDimsAndSizeAfterMove) {
    ov::Shape shape = {1, 2, 3};
    ov::Tensor t{ov::element::f32, shape};

    ov::Tensor new_tensor(std::move(t));

    ASSERT_EQ(shape, new_tensor.get_shape());
    ASSERT_EQ(ov::element::f32, new_tensor.get_element_type());
    ASSERT_EQ(byteStrides(ov::row_major_strides(shape), new_tensor.get_element_type()), new_tensor.get_strides());

    ASSERT_THROW(t.get_size(), ov::Exception);
    ASSERT_THROW(t.get_element_type(), ov::Exception);
    ASSERT_THROW(t.get_byte_size(), ov::Exception);
    ASSERT_THROW(t.get_strides(), ov::Exception);
    ASSERT_THROW(t.get_shape(), ov::Exception);
    ASSERT_THROW(t.set_shape({}), ov::Exception);
    ASSERT_THROW(t.data(), ov::Exception);
    ASSERT_THROW(t.data<float>(), ov::Exception);
}

// SetShape
TEST_F(OVTensorTest, canSetShape) {
    const ov::Shape origShape({1, 2, 3});
    ov::Tensor t{ov::element::f32, {1, 2, 3}};
    const ov::Shape newShape({4, 5, 6});

    const void* orig_data = t.data();
    ASSERT_EQ(t.get_shape(), origShape);
    ASSERT_NO_THROW(t.set_shape({4, 5, 6}));
    ASSERT_EQ(newShape, t.get_shape());
    ASSERT_EQ(byteStrides(ov::row_major_strides(newShape), t.get_element_type()), t.get_strides());
    ASSERT_NE(orig_data, t.data());

    // check that setShape for copy changes original Tensor
    {
        ov::Tensor t2 = t;
        ASSERT_NO_THROW(t2.set_shape(newShape));
        ASSERT_EQ(newShape, t.get_shape());
        ASSERT_EQ(t2.get_shape(), t.get_shape());
        orig_data = t.data();
    }

    // set_shape for smaller memory - does not perform reallocation
    {
        t.set_shape(origShape);
        ASSERT_EQ(origShape, t.get_shape());
        ASSERT_EQ(orig_data, t.data());
    }
}

TEST_F(OVTensorTest, cannotSetShapeOfBiggerSizeOnPreallocatedMemory) {
    float data[4 * 5 * 6 * 2];
    ov::Tensor t{ov::element::f32, {1, 2, 3}, data};
    const ov::Shape newShape({4, 5, 6});

    ASSERT_THROW(t.set_shape(newShape), ov::Exception);
}

TEST_F(OVTensorTest, canSetShapeOfSmallerSizeOnPreallocatedMemory) {
    float data[4 * 5 * 6 * 2];
    ov::Tensor t{ov::element::f32, {4, 5, 6}, data};
    const ov::Shape newShape({1, 2, 3});

    ASSERT_NO_THROW(t.set_shape(newShape));
}

TEST_F(OVTensorTest, canSetShapeOfSameSizeOnPreallocatedMemory) {
    float data[4 * 5 * 6 * 2];
    ov::Tensor t{ov::element::f32, {4, 5, 6}, data};
    const ov::Shape newShape({4, 5, 6});

    ASSERT_NO_THROW(t.set_shape(newShape));
}

TEST_F(OVTensorTest, canSetShapeOfOriginalSizeAfterDecreasingOnPreallocatedMemory) {
    float data[4 * 5 * 6 * 2];
    ov::Tensor t{ov::element::f32, {4, 5, 6}, data};
    const ov::Shape smallerShape({1, 2, 3});
    const ov::Shape originalShape({4, 5, 6});

    ASSERT_NO_THROW(t.set_shape(smallerShape));
    ASSERT_NO_THROW(t.set_shape(originalShape));
}

TEST_F(OVTensorTest, canChangeShapeOnStridedTensor) {
    float data[64 * 4];
    ov::Tensor t{ov::element::f32, {4, 2, 2}, data, {64, 16, 4}};
    const ov::Shape incorrect_shape({2, 4, 2});
    const ov::Shape correct_shape({1, 1, 2});

    ASSERT_THROW(t.set_shape(incorrect_shape), ov::Exception);
    ASSERT_NO_THROW(t.set_shape(correct_shape));
}

TEST_F(OVTensorTest, makeRangeRoiTensor) {
    ov::Tensor t{ov::element::i32, {1, 3, 6, 5}};  // RGBp picture of size (WxH) = 5x6
    ov::Tensor roi_tensor{t, {0, 0, 1, 2}, {1, 3, 5, 4}};
    ov::Shape ref_shape = {1, 3, 4, 2};
    ptrdiff_t ref_offset_elems = 7;
    ptrdiff_t ref_offset_bytes = ref_offset_elems * ov::element::i32.size();
    ov::Strides ref_strides = {90, 30, 5, 1};
    ASSERT_EQ(roi_tensor.get_shape(), ref_shape);
    ASSERT_EQ(roi_tensor.data<int32_t>() - t.data<int32_t>(), ref_offset_elems);
    ASSERT_EQ(reinterpret_cast<uint8_t*>(roi_tensor.data()) - reinterpret_cast<uint8_t*>(t.data()), ref_offset_bytes);
    ASSERT_EQ(roi_tensor.get_strides(), t.get_strides());
    ASSERT_EQ(byteStrides(ref_strides, roi_tensor.get_element_type()), roi_tensor.get_strides());
    ASSERT_EQ(roi_tensor.get_element_type(), t.get_element_type());
}

TEST_F(OVTensorTest, cannotSetShapeOnRoiTensor) {
    ov::Tensor t{ov::element::i32, {1, 3, 6, 5}};  // RGBp picture of size (WxH) = 5x6
    ov::Tensor roi_tensor{t, {0, 0, 1, 2}, {1, 3, 5, 4}};
    const ov::Shape newShape({4, 5, 6});

    ASSERT_THROW(roi_tensor.set_shape(newShape), ov::Exception);
}

TEST_F(OVTensorTest, tensorInt4DataAccess) {
    ov::Tensor t{ov::element::i4, {1, 6, 5, 3}};  // RGB picture of size (WxH) = 5x6
    ASSERT_THROW((ov::Tensor{t, {0, 1, 2, 0}, {1, 5, 4, 3}}), ov::Exception);
    ASSERT_THROW(t.get_strides(), ov::Exception);
    ASSERT_THROW(t.data<int8_t>(), ov::Exception);
    ASSERT_NO_THROW(t.data());
}

TEST_F(OVTensorTest, makeRangeRoiBlobWrongSize) {
    ov::Tensor t{ov::element::f32, {1, 3, 4, 4}};
    ASSERT_THROW((ov::Tensor{t, {0, 0, 1, 1}, {1, 3, 5, 5}}), ov::Exception);
    ASSERT_THROW((ov::Tensor{t, {0, 0, 1, 1, 3}, {1, 3, 4, 4}}), ov::Exception);
}

TEST_F(OVTensorTest, readRangeRoiBlob) {
    ov::Tensor t{ov::element::i32, {1, 3, 4, 8}};
    {
        const auto origPtr = t.data<int32_t>();
        ASSERT_NE(nullptr, origPtr);
        for (size_t i = 0; i < t.get_size(); ++i) {
            origPtr[i] = static_cast<int32_t>(i);
        }
    }
    ov::Tensor roi_tensor{t, {0, 0, 2, 4}, {1, 3, 4, 8}};
    ASSERT_NE(false, static_cast<bool>(roi_tensor));
    {
        const std::uint8_t* roi = reinterpret_cast<const std::uint8_t*>(roi_tensor.data());
        ASSERT_NE(nullptr, roi);
        auto strides = roi_tensor.get_strides();
        for (auto&& c : ngraph::CoordinateTransformBasic{roi_tensor.get_shape()}) {
            auto actual_addr = roi + c[3] * strides[3] + c[2] * strides[2] + c[1] * strides[1] + c[0] * strides[0];
            auto expected_addr = t.data<int32_t>() + ((c[3] + 4) * strides[3] + (c[2] + 2) * strides[2] +
                                                      (c[1] + 0) * strides[1] + (c[0] + 0) * strides[0]) /
                                                         t.get_element_type().size();
            ASSERT_EQ(actual_addr, reinterpret_cast<const std::uint8_t*>(expected_addr));
        }
    }
}

struct TestParams {
    ov::Shape src_shape;
    ov::Strides src_strides;
    ov::Shape dst_shape;
    ov::Strides dst_strides;
};

struct OVTensorTestCopy : ::testing::TestWithParam<std::tuple<ov::element::Type, TestParams>> {};

namespace {
template <class T>
std::vector<T> fill_data(const ov::Tensor& tensor) {
    std::vector<T> actual;
    const T* data = tensor.data<T>();
    auto strides = tensor.get_strides();
    for (auto&& c : ngraph::CoordinateTransformBasic{tensor.get_shape()}) {
        size_t offset = 0;
        for (size_t i = 0; i < strides.size(); i++)
            offset += c[i] * strides[i];
        actual.emplace_back(*(data + offset / tensor.get_element_type().size()));
    }
    return actual;
};
template <class T>
void compare_data(const ov::Tensor& src, const ov::Tensor& dst) {
    auto source_vec = fill_data<T>(src);
    auto dest_vec = fill_data<T>(dst);

    ASSERT_EQ(source_vec.size(), dest_vec.size());

    for (size_t i = 0; i < source_vec.size(); i++) {
        EXPECT_EQ(source_vec[i], dest_vec[i]);
    }
};

template <class T>
void init_tensor(const ov::Tensor& tensor, bool input) {
    const auto origPtr = tensor.data<T>();
    ASSERT_NE(nullptr, origPtr);
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        origPtr[i] = static_cast<T>(input ? i : -1);
    }
}

void init_tensor(const ov::Tensor& tensor, bool input) {
    switch (tensor.get_element_type()) {
    case ov::element::bf16:
        init_tensor<ov::element_type_traits<ov::element::bf16>::value_type>(tensor, input);
        break;
    case ov::element::f16:
        init_tensor<ov::element_type_traits<ov::element::f16>::value_type>(tensor, input);
        break;
    case ov::element::f32:
        init_tensor<ov::element_type_traits<ov::element::f32>::value_type>(tensor, input);
        break;
    case ov::element::f64:
        init_tensor<ov::element_type_traits<ov::element::f64>::value_type>(tensor, input);
        break;
    case ov::element::i8:
        init_tensor<ov::element_type_traits<ov::element::i8>::value_type>(tensor, input);
        break;
    case ov::element::i16:
        init_tensor<ov::element_type_traits<ov::element::i16>::value_type>(tensor, input);
        break;
    case ov::element::i32:
        init_tensor<ov::element_type_traits<ov::element::i32>::value_type>(tensor, input);
        break;
    case ov::element::i64:
        init_tensor<ov::element_type_traits<ov::element::i64>::value_type>(tensor, input);
        break;
    case ov::element::u8:
        init_tensor<ov::element_type_traits<ov::element::u8>::value_type>(tensor, input);
        break;
    case ov::element::u16:
        init_tensor<ov::element_type_traits<ov::element::u16>::value_type>(tensor, input);
        break;
    case ov::element::u32:
        init_tensor<ov::element_type_traits<ov::element::u32>::value_type>(tensor, input);
        break;
    case ov::element::u64:
        init_tensor<ov::element_type_traits<ov::element::u64>::value_type>(tensor, input);
        break;
    default:
        OPENVINO_THROW("Unsupported data type");
    }
}

void compare_tensors(const ov::Tensor& src, const ov::Tensor& dst) {
    ASSERT_EQ(src.get_byte_size(), dst.get_byte_size());
    ASSERT_EQ(src.get_size(), dst.get_size());
    ASSERT_EQ(src.get_element_type(), dst.get_element_type());
    switch (src.get_element_type()) {
    case ov::element::bf16:
        compare_data<ov::element_type_traits<ov::element::bf16>::value_type>(src, dst);
        break;
    case ov::element::f16:
        compare_data<ov::element_type_traits<ov::element::f16>::value_type>(src, dst);
        break;
    case ov::element::f32:
        compare_data<ov::element_type_traits<ov::element::f32>::value_type>(src, dst);
        break;
    case ov::element::f64:
        compare_data<ov::element_type_traits<ov::element::f64>::value_type>(src, dst);
        break;
    case ov::element::i8:
        compare_data<ov::element_type_traits<ov::element::i8>::value_type>(src, dst);
        break;
    case ov::element::i16:
        compare_data<ov::element_type_traits<ov::element::i16>::value_type>(src, dst);
        break;
    case ov::element::i32:
        compare_data<ov::element_type_traits<ov::element::i32>::value_type>(src, dst);
        break;
    case ov::element::i64:
        compare_data<ov::element_type_traits<ov::element::i64>::value_type>(src, dst);
        break;
    case ov::element::u8:
        compare_data<ov::element_type_traits<ov::element::u8>::value_type>(src, dst);
        break;
    case ov::element::u16:
        compare_data<ov::element_type_traits<ov::element::u16>::value_type>(src, dst);
        break;
    case ov::element::u32:
        compare_data<ov::element_type_traits<ov::element::u32>::value_type>(src, dst);
        break;
    case ov::element::u64:
        compare_data<ov::element_type_traits<ov::element::u64>::value_type>(src, dst);
        break;
    default:
        OPENVINO_THROW("Unsupported data type");
    }
}
}  // namespace

TEST_P(OVTensorTestCopy, copy_to) {
    ov::element::Type type;
    TestParams p;
    std::tie(type, p) = GetParam();
    // Source tensors
    ov::Tensor full_src_tensor;
    ov::Tensor src_tensor;
    if (!p.src_strides.empty()) {
        full_src_tensor = ov::Tensor(type, ov::Shape{p.src_shape[0] * p.src_strides[0]});
        src_tensor = ov::Tensor(type, p.src_shape, full_src_tensor.data(), p.src_strides);
    } else {
        src_tensor = full_src_tensor = ov::Tensor(type, p.src_shape);
    }
    init_tensor(full_src_tensor, true);

    ov::Tensor full_dst_tensor;
    ov::Tensor dst_tensor;
    if (!p.dst_strides.empty()) {
        full_dst_tensor = ov::Tensor(type, ov::Shape{p.dst_shape[0] * p.dst_strides[0]});
        dst_tensor = ov::Tensor(type, p.dst_shape, full_dst_tensor.data(), p.dst_strides);
    } else {
        dst_tensor = full_dst_tensor = ov::Tensor(type, p.dst_shape);
    }
    init_tensor(full_src_tensor, false);

    src_tensor.copy_to(dst_tensor);
    compare_tensors(src_tensor, dst_tensor);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(copy_tests,
                         OVTensorTestCopy,
                         ::testing::Combine(::testing::Values(
                                                              ov::element::bf16,
                                                              ov::element::f16,
                                                              ov::element::f32,
                                                              ov::element::f64,
                                                              ov::element::i8,
                                                              ov::element::i16,
                                                              ov::element::i32,
                                                              ov::element::i64,
                                                              ov::element::u8,
                                                              ov::element::u16,
                                                              ov::element::u32,
                                                              ov::element::u64
                                            ),
                                            ::testing::Values(
                                                              TestParams {
                                                                  ov::Shape{1, 3, 4, 8}, {}, 
                                                                  {0}, {}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{3, 2, 2}, {},
                                                                  ov::Shape{3, 2, 2}, ov::Strides{128, 24, 8}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{3, 2, 2}, ov::Strides{64, 16, 8},
                                                                  ov::Shape{3, 2, 2}, ov::Strides{}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{3, 2, 2}, ov::Strides{64, 16, 8},
                                                                  ov::Shape{3, 2, 2}, ov::Strides{128, 24, 8}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{}, {}, 
                                                                  {}, {}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{1}, {}, 
                                                                  {}, {}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{}, {}, 
                                                                  {1}, {}
                                                              }
                                           )));
// clang-format on
