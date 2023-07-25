// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include "ngraph/runtime/reference/matmul.hpp"

#include "compilation_context.hpp"
#include "gemm_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace  {

const std::vector<cldnn::format> f_blocked_4d_formats = {
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
};

const std::vector<cldnn::format> f_blocked_5d_formats = {
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
};

const std::vector<cldnn::format> b_blocked_4d_formats = {
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
};

const std::vector<cldnn::format> b_blocked_5d_formats = {
    format::bs_fs_zyx_bsv16_fsv32,
    format::bs_fs_zyx_bsv16_fsv16,
    format::bs_fs_zyx_bsv32_fsv32,
    format::bs_fs_zyx_bsv32_fsv16,
};

// TODO: uncomment in scope of CVS-85940
const std::vector<cldnn::format> planar_formats = {
    format::bfyx,
    /*
    format::bfzyx,
    format::bfwzyx,
     */
};


const std::vector<data_types> float_types = {
    data_types::f16, data_types::f32 };

const std::vector<data_types> all_types = {
    data_types::f16, data_types::f32 , data_types::i8, data_types::u8, data_types::i32
};

typedef std::tuple<
std::vector<std::vector<int32_t>>,
std::vector<std::vector<float>>,
format,
data_types,
std::vector<float>,
bool,
bool,
float,
float
>
GemmParams;

class GemmGPUTest : public ::testing::TestWithParam<GemmParams> {
protected:
    std::vector<std::vector<float>> input_data;
    std::vector<float> out_data;
    std::vector<std::vector<int32_t>> shapes;
    format fmt{format::bfyx};
    data_types type;
    bool transpose_input0;
    bool transpose_input1;
    float alpha;
    float beta;

    virtual void fill_gemm_params() {
        GemmParams params = testing::TestWithParam<GemmParams>::GetParam();
        std::tie(shapes, input_data, fmt, type, out_data, transpose_input0,
                 transpose_input1, alpha, beta) = params;
    }

    virtual void process_program(program::ptr) {
    }

public:
    virtual ~GemmGPUTest() {}
    void test(bool is_caching_test = false) {

        fill_gemm_params();

        topology tp;

        std::vector<std::pair<primitive_id, memory_ptr>> network_inputs;
        std::vector<input_info> gemm_inputs;

        auto &engine = get_test_engine();
        for (size_t i = 0; i < shapes.size(); ++i) {
            tensor t{shapes[i]};
            layout l{data_types::f32, format::bfyx, t};
            auto input = engine.allocate_memory(l);
            set_values(input, input_data[i]);
            primitive_id prim_id = std::string("input") + std::to_string(i);
            network_inputs.emplace_back(prim_id, input);

            primitive_id prim_id_reordered = prim_id + "_reordered";
            // tp.add(data(prim_id, input));
            tp.add(input_layout(prim_id, input->get_layout()));
            tp.add(reorder(prim_id_reordered, prim_id, fmt, type));
            gemm_inputs.push_back(input_info(prim_id_reordered));
        }

        auto g = gemm("gemm_output", gemm_inputs, type, transpose_input0, transpose_input1, alpha, beta);
        tp.add(g);
        tp.add(reorder("output", input_info("gemm_output"), format::bfyx, data_types::f32));

        cldnn::network::ptr network;
        if (is_caching_test) {
            membuf mem_buf;
            {
                cldnn::network _network(engine, tp, get_test_default_config(engine));
                process_program(_network.get_program());
                std::ostream out_mem(&mem_buf);
                BinaryOutputBuffer ob = BinaryOutputBuffer(out_mem);
                _network.save(ob);
            }
            {
                std::istream in_mem(&mem_buf);
                BinaryInputBuffer ib = BinaryInputBuffer(in_mem, engine);
                network = std::make_shared<cldnn::network>(ib, get_test_stream_ptr(), engine, true, 0);
            }
        } else {
            network = std::make_shared<cldnn::network>(engine, tp, get_test_default_config(engine));
            process_program(network->get_program());
        }

        for (auto &input : network_inputs) {
            network->set_input_data(input.first, input.second);
        }
        auto outputs = network->execute();
        auto output = outputs.at("output").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), out_data.size());
        const auto abs_error = type == data_types::f16 ? 0.1 : 0.0001;
        for (uint32_t i = 0; i < out_data.size(); ++i) {
            ASSERT_NEAR(output_ptr[i], out_data[i], abs_error);
        }
    }
};

class GemmGPUTestRandom : public GemmGPUTest {

    ov::Shape input0_shape;
    ov::Shape input1_shape;
    ov::Shape output_shape;

    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void generated_inputs() {
        for (size_t i = 0; i < shapes.size(); ++i) {
            size_t size = ngraph::shape_size(shapes[i]);
            input_data[i] = rg.generate_random_1d<float>(size, -1, 1, 10);
        }
    }

    void process_program(program::ptr p) override {
        std::vector<program_node*>& prog_nodes = p->get_outputs();
        auto inputs = p->get_inputs();
        auto input_it = inputs.begin();
        input0_shape = (*input_it)->get_output_layout().get_shape();
        ++input_it;
        input1_shape = (*input_it)->get_output_layout().get_shape();
        layout output_layout = prog_nodes[0]->get_output_layout();
        output_shape = output_layout.get_shape();
        out_data.resize(ngraph::shape_size(output_shape));
        calculate_output_data();
    }

    void calculate_output_data() {
        ngraph::runtime::reference::matmul<float>(
                input_data[0].data(),
                input_data[1].data(),
                out_data.data(),
                input0_shape,
                input1_shape,
                output_shape,
                transpose_input0,
                transpose_input1);
    }

protected:

    void fill_gemm_params() override {
        GemmGPUTest::fill_gemm_params();
        // this class support only simple gemm case: 2 inputs, alpha eq 1.f and beta eq 0.f
        ASSERT_THAT(input_data.size(), 2ul);
        ASSERT_THAT(alpha, 1.f);
        ASSERT_THAT(beta, 0.f);
        generated_inputs();
    }

};

TEST_P(GemmGPUTest, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(GemmGPUTestRandom, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_t1, GemmGPUTestRandom,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{{1, 1, 3, 4},
                                                            {1, 1, 1, 4}}),
        ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(float_types),
        ::testing::Values(std::vector<float>{}),
        ::testing::Values(true), ::testing::Values(false),
        ::testing::Values(1.0f), ::testing::Values(0.0f)));

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_t2, GemmGPUTestRandom,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{{1, 1, 4, 3},
                                                            {1, 1, 4, 1}}),
        ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(float_types),
        ::testing::Values(std::vector<float>{}),
        ::testing::Values(false), ::testing::Values(true),
        ::testing::Values(1.0f), ::testing::Values(0.0f)));

template <typename T>
void test_basic_bfyx_t2_inplace_crop_with_pad(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } });

    std::vector<T> input_data = {
        1.f, -2.f,  3.f, -4.f,
        5.f,  6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,

        1.f, -2.f,  3.f, -4.f,
        5.f,  6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<T> input_data2 = {
        2.f, 5.f, -4.f, -7.f,
    };
    set_values(input, input_data);
    set_values(input2, input_data2);

    std::vector<T> out_data = {
        8.f, 22.f, 20.f
    };

    topology topology;
    topology.add(
        input_layout("input", input->get_layout())
    );
    topology.add(
        input_layout("input2", input2->get_layout())
    );
    topology.add(
        crop("crop.1", input_info("input"), { 1, 1, 4, 3 }, { 0, 1, 0, 0 })
    );
    topology.add(
        gemm("output", { input_info("crop.1"), input_info("input2") }, data_types::f32, false, true)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("input", input);
    network->set_input_data("input2", input2);
    auto outputs = network->execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<T> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(gemm_gpu, basic_bfyx_t2_inplace_crop_with_pad) {
    test_basic_bfyx_t2_inplace_crop_with_pad<float>(false);
}

TEST(gemm_gpu, dynamic) {
    auto& engine = get_test_engine();
    ov::Shape in1_shape = { 1, 1, 3, 4 };
    ov::Shape in2_shape = { 1, 4 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});

    std::vector<float> input1_data = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f,
    };
    set_values(input1, input1_data);
    set_values(input2, input2_data);

    std::vector<float> out_data = {
        8.f, 22.f, 20.f
    };

    topology topology;
    topology.add(input_layout("input1", in1_layout),
                 input_layout("input2", in2_layout),
                 gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, true, 1.0f, 0.0f, 4, 2)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("gemm");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gemm").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(gemm_gpu, dynamic_multi_inference_same_shape) {
    auto& engine = get_test_engine();

    auto in1_dyn_layout = layout{ ov::PartialShape{ 1, 1, ov::Dimension(1, 10), 4 }, data_types::f32, format::bfyx };
    auto in1_actual_layout = layout{ ov::PartialShape{ 1, 1, 3, 4 }, data_types::f32, format::bfyx };
    auto in2_dyn_layout = layout{ ov::PartialShape{ 4, ov::Dimension(1, 10) }, data_types::f32, format::bfyx };
    auto in2_actual_layout = layout{ ov::PartialShape{ 4, 1 }, data_types::f32, format::bfyx };
    auto input1_1 = engine.allocate_memory(in1_actual_layout);
    auto input1_2 = engine.allocate_memory(in1_actual_layout);
    auto input2_1 = engine.allocate_memory(in2_actual_layout);
    auto input2_2 = engine.allocate_memory(in2_actual_layout);

    std::vector<float> input1_data1 = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };
    std::vector<float> input1_data2 = {
        -1.f, 2.f, -3.f, 4.f,
        5.f, 6.f, -1.f, 2.f,
        3.f, -3.f, 2.f, 1.f,
    };
    std::vector<float> input2_data1 = {
        2.f, 5.f, -4.f, -7.f,
    };
    std::vector<float> input2_data2 = {
        4.f, 7.f, 2.f, 5.f,
    };
    set_values(input1_1, input1_data1);
    set_values(input1_2, input1_data2);
    set_values(input2_1, input2_data1);
    set_values(input2_2, input2_data2);

    std::vector<float> out_data1 = {
        8.f, 22.f, 20.f
    };
    std::vector<float> out_data2 = {
        24.f, 70.f, 0.f
    };

    topology topology;
    topology.add(input_layout("input1", in1_dyn_layout),
                 input_layout("input2", in2_dyn_layout),
                 gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, false, 1.0f, 0.0f, 4, 2)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    {
        network.set_input_data("input1", input1_1);
        network.set_input_data("input2", input2_1);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gemm");

        auto prog = network.get_program();
        auto& node = prog->get_node("gemm");
        auto impl = node.get_selected_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)3);
        for (uint32_t i = 0; i < out_data1.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data1[i]);
        }
    }

    {
        network.set_input_data("input1", input1_2);
        network.set_input_data("input2", input2_2);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gemm");

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)3);
        for (uint32_t i = 0; i < out_data2.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data2[i]);
        }
    }
}

TEST(gemm_gpu, dynamic_multi_inference_different_shape) {
    auto& engine = get_test_engine();

    auto in1_dyn_layout = layout{ ov::PartialShape{ 1, 1, ov::Dimension(1, 10), 4 }, data_types::f32, format::bfyx };
    auto in1_actual_layout1 = layout{ ov::PartialShape{ 1, 1, 3, 4 }, data_types::f32, format::bfyx };
    auto in1_actual_layout2 = layout{ ov::PartialShape{ 1, 1, 4, 4 }, data_types::f32, format::bfyx };
    auto in2_dyn_layout = layout{ ov::PartialShape{ 4, ov::Dimension(1, 10) }, data_types::f32, format::bfyx };
    auto in2_actual_layout = layout{ ov::PartialShape{ 4, 1 }, data_types::f32, format::bfyx };
    auto input1_1 = engine.allocate_memory(in1_actual_layout1);
    auto input1_2 = engine.allocate_memory(in1_actual_layout2);
    auto input2 = engine.allocate_memory(in2_actual_layout);

    std::vector<float> input1_data1 = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };
    std::vector<float> input1_data2 = {
        -1.f, 2.f, -3.f, 4.f,
        5.f, 6.f, -1.f, 2.f,
        3.f, -3.f, 2.f, 1.f,
        1.f, 2.f, -5.f, 6.f,
    };
    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f,
    };
    set_values(input1_1, input1_data1);
    set_values(input1_2, input1_data2);
    set_values(input2, input2_data);

    std::vector<float> out_data1 = {
        8.f, 22.f, 20.f
    };
    std::vector<float> out_data2 = {
        -8.f, 30.f, -24.f, -10.f
    };

    topology topology;
    topology.add(input_layout("input1", in1_dyn_layout),
                 input_layout("input2", in2_dyn_layout),
                 gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, false, 1.0f, 0.0f, 4, 2)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    {
        network.set_input_data("input1", input1_1);
        network.set_input_data("input2", input2);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gemm");

        auto prog = network.get_program();
        auto& node = prog->get_node("gemm");
        auto impl = node.get_selected_impl();
        ASSERT_TRUE(impl != nullptr);
        ASSERT_TRUE(impl->is_dynamic());

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)3);
        for (uint32_t i = 0; i < out_data1.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data1[i]);
        }
    }

    {
        network.set_input_data("input1", input1_2);
        network.set_input_data("input2", input2);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "gemm");

        auto output_prim_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

        ASSERT_EQ(output_ptr.size(), (uint32_t)4);
        for (uint32_t i = 0; i < out_data2.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], out_data2[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_t1t2,
        GemmGPUTestRandom,
        ::testing::Combine(
            ::testing::Values(std::vector<std::vector<int32_t>>{{2, 1, 3, 4}, {2, 1, 4, 1}}),
            ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
            ::testing::ValuesIn(planar_formats),
            ::testing::ValuesIn(float_types),
            ::testing::Values(std::vector<float>{}),
            ::testing::Values(true),
            ::testing::Values(true),
            ::testing::Values(1.0f),
            ::testing::Values(0.0f)
            )
        );

INSTANTIATE_TEST_SUITE_P(
    GemmGPUTest_basic_input3, GemmGPUTest,
    ::testing::Combine(
        ::testing::Values(std::vector<std::vector<int32_t>>{
            {1, 1, 3, 2}, {1, 1, 2, 3}, {1, 1, 2, 2}}),
        ::testing::Values(std::vector<std::vector<float>>{
            {1.0f, 2.0f, 3.0f, 1.0f, 0.0f, 1.0f},
            {
                3.0f,
                3.0f,
                1.0f,
                2.0f,
                1.0f,
                2.0f,
            },
            {
                1.0f,
                0.0f,
                2.0f,
                0.0f,
            }}),
        ::testing::ValuesIn(planar_formats), ::testing::ValuesIn(all_types),
        ::testing::Values(std::vector<float>{26.0f, 26.0f, 28.0f, 10.0f}),
        ::testing::Values(false), ::testing::Values(false),
        ::testing::Values(2.0f), ::testing::Values(10.0f)));

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t1t2,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 4, 3 }, { 1, 1, 3, 2 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f
                        },
                        {
                            3.0f, 3.0f, 1.0f,
                            2.0f, 1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(true),
                    ::testing::Values(true),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_1,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 3, 4 }, { 1, 1, 2, 3 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 1.0f, 0.0f,
                            2.0f, 0.0f, 0.0f,
                            3.0f, 1.0f, 0.0f,
                            4.0f, 0.0f, 0.0f

                        },
                        {
                            3.0f, 2.0f,
                            3.0f, 1.0f,
                            1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t2,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 3, 4 }, { 1, 1, 3, 2 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 1.0f, 0.0f,
                            2.0f, 0.0f, 0.0f,
                            3.0f, 1.0f, 0.0f,
                            4.0f, 0.0f, 0.0f

                        },
                        {
                            3.0f, 3.0f, 1.0f,
                            2.0f, 1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(false),
                    ::testing::Values(true),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_input3_t1,
        GemmGPUTest,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 1, 4, 3 }, { 1, 1, 2, 3 }, { 1, 1, 2, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{
                        {
                            1.0f, 2.0f, 3.0f, 4.0f,
                            1.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f
                        },
                        {
                            3.0f, 2.0f,
                            3.0f, 1.0f,
                            1.0f, 2.0f,
                        },
                        {
                            1.0f, 0.0f,
                            1.0f, 0.0f,
                            2.0f, 2.0f,
                            1.0f, 1.0f,

                        }
                    }),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(all_types),
                    ::testing::Values(std::vector<float>{
                       15.0f, 6.0f,
                       15.0f, 8.0f,
                       30.0f, 20.0f,
                       27.0f, 19.0f
                    }),
                    ::testing::Values(true),
                    ::testing::Values(false),
                    ::testing::Values(2.0f),
                    ::testing::Values(3.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 2, 1, 4, 3 }, { 2, 1, 1, 4 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);


INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic3_bfyx,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 5, 1, 500, 9 }, { 5, 1, 1, 500 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_basic_smarcink2,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 2, 1, 3, 2 }, { 2, 1, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(planar_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_f_block_4d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 32, 3, 2 }, { 1, 32, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(f_blocked_4d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        GemmGPUTest_b_block_4d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 32, 1, 3, 2 }, { 32, 1, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(b_blocked_4d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);
// TODO: enable in scope of CVS-85940
INSTANTIATE_TEST_SUITE_P(
        DISABLED_GemmGPUTest_f_block_5d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 1, 16, 2, 3, 2 }, { 1, 16, 2, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(f_blocked_5d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_GemmGPUTest_b_block_5d_formats,
        GemmGPUTestRandom,
                ::testing::Combine(
                    ::testing::Values(std::vector<std::vector<int32_t>>{{ 16, 1, 2, 3, 2 }, { 16, 1, 2, 2, 3 }}),
                    ::testing::Values(std::vector<std::vector<float>>{{}, {}}),
                    ::testing::ValuesIn(b_blocked_5d_formats),
                    ::testing::ValuesIn(float_types),
                    ::testing::Values(std::vector<float>{}),
                    ::testing::Values(false),
                    ::testing::Values(false),
                    ::testing::Values(1.0f),
                    ::testing::Values(0.0f)
            )
);

struct gemm_base_test_params {
    size_t m_size;
    size_t n_size;
    size_t k_size;
    size_t b0_num;
    size_t f0_num;
    size_t b1_num;
    size_t f1_num;
    size_t b2_num;
    size_t f2_num;
    size_t b_out_num;
    size_t f_out_num;
    bool transpose_input0;
    bool transpose_input1;
    float alpha;
    float beta;
    cldnn::data_types allocate0_type;
    cldnn::data_types allocate1_type;
    cldnn::data_types allocate2_type;
    cldnn::data_types output_type;
    std::vector <int> range0;
    std::vector <int> range1;
    std::vector <int> range2;
    std::string kernel_name;
};

#ifdef ENABLE_ONEDNN_FOR_GPU

#define CASE_GEMM_INT8_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_ONEDNN_1 1, 64, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_2 64, 1, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_3 1, 1, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_ONEDNN_4 64, 64, 64, 2, 2, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_ONEDNN_1 1, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_2 64, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_3 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_ONEDNN_4 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_NN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 0.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_NN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_NT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TN_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TT_TRANSPOSITION_ONEDNN 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN 13, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN 32, 16, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_NT_TRANSPOSITION_ONEDNN 16, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TN_TRANSPOSITION_ONEDNN 32, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TT_TRANSPOSITION_ONEDNN 32, 64, 96, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_NN_TRANSPOSITION_ONEDNN 32, 16, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_NT_TRANSPOSITION_ONEDNN 16, 64, 128, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TN_TRANSPOSITION_ONEDNN 32, 64, 96, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TT_TRANSPOSITION_ONEDNN 32, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32,  { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_1 32, 32, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_3 64, 32, 64, 1, 2, 1, 1, 1, 2, 1, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_BROADCASTING_ONEDNN_4 32, 64, 64, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_COMBO_ONEDNN_1 5, 18, 99, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_2 1, 32, 65, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_3 13, 4, 64, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_ONEDNN_4 128, 126, 127, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::i8, data_types::i8, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#define CASE_GEMM_UINT8_COMBO_ONEDNN_1 11, 16, 65, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_2 13, 14, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_3 16, 16, 99, 1, 2, 1, 2, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_UINT8_COMBO_ONEDNN_4 3, 1, 77, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::u8, data_types::i8, data_types::f32, data_types::f32, { 0, 255, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

// Currently broadcasting support wasn't implemented for f16 cases with biases
#define CASE_GEMM_FP16_COMBO_ONEDNN_1 5, 7, 65, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_2 32, 8, 128, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_3 14, 2, 69, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_COMBO_ONEDNN_4 1, 1, 64, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP32_COMBO_ONEDNN_1 7, 17, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_2 26, 22, 79, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_3 5, 7, 81, 1, 2, 1, 1, 1, 2, 1, 2, true, false, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_COMBO_ONEDNN_4 61, 1, 99, 1, 1, 2, 2, 2, 2, 2, 2, true, true, \
1.0f, 1.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -128, 127, 1 }, { -128, 127, 1 }, { -10, 10, 8 }

#endif  // ENABLE_ONEDNN_FOR_GPU

#define CASE_GEMM_INT8_NN_TRANSPOSITION 64, 64, 64, 1, 2, 1, 2, 1, 2, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_NT_TRANSPOSITION 32, 64, 32, 2, 1, 2, 1, 2, 1, 2, 1, false, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TN_TRANSPOSITION 128, 64, 32, 2, 2, 2, 2, 2, 2, 2, 2, true, false, \
1.0f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_TT_TRANSPOSITION 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.2f, 0.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_BROADCAST_1 32, 32, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_2 32, 32, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_3 64, 32, 32, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_BROADCAST_4 32, 64, 32, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.2f, 0.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_LEFTOVERS_1 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_2 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_3 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_4 13, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_5 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_6 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_7 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_8 32, 13, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_9 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_10 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.6f, 1.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_11 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_LEFTOVERS_12 32, 32, 13, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 1.3f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_COMBO_1 8, 8, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_2 16, 16, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, true, \
1.7f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_3 11, 31, 21, 7, 15, 7, 15, 7, 15, 7, 15, true, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_COMBO_4 32, 32, 32, 3, 6, 3, 6, 3, 6, 3, 6, true, true, \
1.2f, 4.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_INT8_SLM_COMBO_1 64, 64, 64, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_2 384, 384, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_3 128, 128, 64, 2, 3, 2, 3, 2, 3, 2, 3, false, false, \
1.0f, 1.5f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }
#define CASE_GEMM_INT8_SLM_COMBO_4 256, 64, 64, 3, 6, 3, 6, 3, 6, 3, 6, false, false, \
1.2f, 4.0f, data_types::i8, data_types::u8, data_types::f32, data_types::f32, { -128, 127, 1 }, { 0, 255, 1 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NN_1 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_2 64, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_3 31, 47, 65, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_4 65, 31, 47, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_TN_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TN_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_TT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_TT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP32_TILED_NN_BROADCAST_1 64, 96, 32, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_2 32, 16, 16, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_3 5, 1, 3, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }
#define CASE_GEMM_FP32_TILED_NN_BROADCAST_4 64, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 4.0f, data_types::f32, data_types::f32, data_types::f32, data_types::f32, { -10, 10, 8 }, { -10, 10, 8 }, { -10, 10, 8 }

#define CASE_GEMM_FP16_TILED_NN_1 64, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_2 128, 64, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_3 131, 17, 15, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_4 33, 17, 17, 1, 1, 1, 1, 1, 1, 1, 1, false, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_NT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, false, true, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_TN_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TN_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_TT_1 16, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_2 32, 32, 32, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_3 64, 32, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_TT_4 16, 128, 64, 1, 1, 1, 1, 1, 1, 1, 1, true, true, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

#define CASE_GEMM_FP16_TILED_NN_BROADCAST_1 64, 96, 128, 1, 2, 1, 1, 1, 1, 1, 2, false, false, \
1.5f, 2.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_2 64, 16, 64, 2, 1, 1, 1, 1, 1, 2, 1, false, false, \
1.7f, 0.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_3 1, 2, 3, 1, 2, 2, 1, 1, 2, 2, 2, false, false, \
1.0f, 1.5f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }
#define CASE_GEMM_FP16_TILED_NN_BROADCAST_4 8, 8, 8, 1, 1, 2, 2, 2, 2, 2, 2, false, false, \
1.0f, 4.0f, data_types::f16, data_types::f16, data_types::f16, data_types::f16, { -1, 1, 1 }, { -1, 1, 1 }, { -1, 1, 1 }

template <typename gemm_params, typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
class GemmBaseTest : public ::testing::TestWithParam<gemm_params> {
public:
    virtual ov::intel_gpu::ImplementationDesc getImplementationDesc(gemm_params& p) {
         return { format::bfyx, p.kernel_name };
    }

    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    inline size_t getGemmIndex(size_t x, size_t y, size_t f, size_t b, size_t x_size, size_t y_size, size_t f_num, size_t b_num,
                               size_t x_pitch, size_t y_pitch, size_t f_pitch, size_t b_pitch) {
        return (x % x_size) * x_pitch + (y % y_size) * y_pitch + (f % f_num) * f_pitch + (b % b_num) * b_pitch;
    }

    void execute(gemm_params& p, bool is_caching_test = false) {
        auto y0_size = p.m_size;
        auto y0_pitch = p.k_size;
        auto x0_size = p.k_size;
        auto x0_pitch = 1;
        auto f0_pitch = y0_size * x0_size;
        auto b0_pitch = p.f0_num * f0_pitch;

        auto y1_size = p.k_size;
        auto y1_pitch = p.n_size;
        auto x1_size = p.n_size;
        auto x1_pitch = 1;
        auto f1_pitch = y1_size * x1_size;
        auto b1_pitch = p.f1_num * f1_pitch;

        auto y2_size = p.m_size;
        auto y2_pitch = p.n_size;
        auto x2_size = p.n_size;
        auto x2_pitch = 1;
        auto f2_pitch = y2_size * x2_size;
        auto b2_pitch = p.f2_num * f2_pitch;

        auto y_out_size = p.m_size;
        auto y_out_pitch = p.n_size;
        auto x_out_size = p.n_size;
        auto x_out_pitch = 1;
        auto f_out_pitch = y_out_size * x_out_size;
        auto b_out_pitch = p.f_out_num * f_out_pitch;

        if (p.transpose_input0) {
            y0_size = p.k_size;
            y0_pitch = p.m_size;
            x0_size = p.m_size;
            x0_pitch = 1;
        }

        if (p.transpose_input1) {
            y1_size = p.n_size;
            y1_pitch = p.k_size;
            x1_size = p.k_size;
            x1_pitch = 1;
        }

        auto& engine = get_test_engine();
        auto input0_size = tensor((int)p.b0_num, (int)p.f0_num, (int)x0_size, (int)y0_size);
        VVVVF<input0_type> input0_data = rg.generate_random_4d<input0_type>(p.b0_num, p.f0_num, x0_size, y0_size, p.range0[0], p.range0[1], p.range0[2]);
        auto input0_data_bfyx = flatten_4d(format::bfyx, input0_data);
        auto input0_mem = engine.allocate_memory({ p.allocate0_type, format::bfyx, input0_size });
        set_values(input0_mem, input0_data_bfyx);

        auto input1_size = tensor((int)p.b1_num, (int)p.f1_num, (int)x1_size, (int)y1_size);
        VVVVF<input1_type> input1_data = rg.generate_random_4d<input1_type>(p.b1_num, p.f1_num, x1_size, y1_size, p.range1[0], p.range1[1], p.range1[2]);
        auto input1_data_bfyx = flatten_4d(format::bfyx, input1_data);
        auto input1_mem = engine.allocate_memory({ p.allocate1_type, format::bfyx, input1_size });
        set_values(input1_mem, input1_data_bfyx);

        auto input2_size = tensor((int)p.b2_num, (int)p.f2_num, (int)x2_size, (int)y2_size);
        VVVVF<input2_type> input2_data = rg.generate_random_4d<input2_type>(p.b2_num, p.f2_num, x2_size, y2_size, p.range2[0], p.range2[1], p.range2[2]);
        auto input2_data_bfyx = flatten_4d(format::bfyx, input2_data);
        auto input2_mem = engine.allocate_memory({ p.allocate2_type, format::bfyx, input2_size });
        set_values(input2_mem, input2_data_bfyx);

        std::vector<output_type> out_data(p.b_out_num * p.f_out_num * p.m_size * p.n_size);

        for (size_t b = 0; b < p.b_out_num; ++b) {
            for (size_t f = 0; f < p.f_out_num; ++f) {
                for (size_t y = 0; y < p.m_size; ++y) {
                    for (size_t x = 0; x < p.n_size; ++x) {
                        size_t input2_data_index = getGemmIndex(x, y, f, b, x2_size, y2_size, p.f2_num, p.b2_num, x2_pitch, y2_pitch, f2_pitch, b2_pitch);
                        size_t out_data_index = getGemmIndex(x, y, f, b, x_out_size, y_out_size, p.f_out_num, p.b_out_num,
                                                             x_out_pitch, y_out_pitch, f_out_pitch, b_out_pitch);
                        accumulator_type acc = 0;

                        for (size_t k = 0; k < p.k_size; ++k) {
                            size_t input0_data_index = getGemmIndex(k * (!p.transpose_input0) + y * p.transpose_input0, y * (!p.transpose_input0) +
                            k * p.transpose_input0, f, b, x0_size, y0_size, p.f0_num, p.b0_num, x0_pitch, y0_pitch, f0_pitch, b0_pitch);
                            size_t input1_data_index = getGemmIndex(x * (!p.transpose_input1) + k * p.transpose_input1, k * (!p.transpose_input1) +
                            x * p.transpose_input1, f, b, x1_size, y1_size, p.f1_num, p.b1_num, x1_pitch, y1_pitch, f1_pitch, b1_pitch);
                            acc += (accumulator_type)input0_data_bfyx[input0_data_index] * (accumulator_type)input1_data_bfyx[input1_data_index];
                        }

                        out_data[out_data_index] = (output_type)acc;
                        out_data[out_data_index] *= (output_type)p.alpha;
                        if (p.beta)
                            out_data[out_data_index] += (output_type)p.beta * (output_type)input2_data_bfyx[input2_data_index];
                    }
                }
            }
        }

        topology topology;
        topology.add(input_layout("input0", input0_mem->get_layout()));
        topology.add(input_layout("input1", input1_mem->get_layout()));
        if (p.beta != 0) {
            topology.add(input_layout("input2", input2_mem->get_layout()));
            topology.add(gemm("gemm_bfyx", { input_info("input0"), input_info("input1"), input_info("input2") }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        } else {
            topology.add(gemm("gemm_bfyx", { input_info("input0"), input_info("input1") }, p.output_type, p.transpose_input0, p.transpose_input1, p.alpha, p.beta));
        }
        topology.add(reorder("reorder_bfyx", input_info("gemm_bfyx"), format::bfyx, data_types::f32));

        ov::intel_gpu::ImplementationDesc gemm_impl = getImplementationDesc(p);

        ExecutionConfig cfg = get_test_default_config(engine);
        cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm_bfyx", gemm_impl} }));

        cldnn::network::ptr network = get_network(engine, topology, cfg, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input0", input0_mem);
        network->set_input_data("input1", input1_mem);
        if (p.beta != 0) {
            network->set_input_data("input2", input2_mem);
        }
        auto outputs = network->execute();
        auto output = outputs.at("reorder_bfyx").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());

        const float threshold_int8 = 1.f;
        const float threshold_fp16 = 1e-1;
        const float threshold_fp32 = 3e-4;

        ASSERT_EQ(output_ptr.size(), (size_t)(p.b_out_num * p.f_out_num * p.m_size * p.n_size));
        if (sizeof(input0_type) == 1) {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_int8) << "index = " << i;
            }
        } else if (sizeof(input0_type) == 2) {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp16) << "index = " << i;
            }
        } else {
            for (size_t i = 0; i < out_data.size(); ++i) {
                ASSERT_NEAR(float(output_ptr[i]), float(out_data[i]), threshold_fp32) << "index = " << i;
            }
        }
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
struct gemm_onednn_test_params {
    std::vector<tensor> in_shapes;
    tensor out_shape;
    tensor kernel;
    tensor pad;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
};

template <typename T>
class GemmOneDNNTest : public ::testing::TestWithParam<T> {
public:
    tests::random_generator rg;
    cldnn::engine& engine = get_test_engine();
    topology topology_ocl;
    topology topology_onednn;

    ExecutionConfig config_ocl;
    ExecutionConfig config_onednn;

    float tolerance = 0.0f;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
        config_ocl.set_property(ov::intel_gpu::optimize_data(true));
        config_ocl.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        if (engine.get_device_info().supports_immad) {
            config_onednn.set_property(ov::intel_gpu::optimize_data(true));
            config_onednn.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        }
    }

    void execute(T& p) {
        auto input0_prim = get_generated_random_1d_mem(engine, get_input_layout(p, 0));
        auto input1_prim = get_generated_random_1d_mem(engine, get_input_layout(p, 1));

        network network_ocl(engine, topology_ocl, config_ocl);
        network network_onednn(engine, topology_onednn, config_onednn);

        network_ocl.set_input_data("input0", input0_prim);
        network_ocl.set_input_data("input1", input1_prim);
        network_onednn.set_input_data("input0", input0_prim);
        network_onednn.set_input_data("input1", input1_prim);

        compare(network_ocl, network_onednn, p);
    }

    void compare(network& network_ocl, network& network_onednn, T& p) {
        auto outputs_ocl = network_ocl.execute();
        auto outputs_onednn = network_onednn.execute();

        ASSERT_EQ(outputs_ocl.size(), outputs_onednn.size());
        ASSERT_EQ(outputs_ocl.size(), size_t(1));

        auto val_ocl = get_output_values_to_float(network_ocl, outputs_ocl.begin()->second);
        auto val_onednn = get_output_values_to_float(network_onednn, outputs_onednn.begin()->second);

        ASSERT_EQ(val_ocl.size(), val_onednn.size());

        for (size_t i = 0; i < val_ocl.size(); i++) {
            ASSERT_NEAR(val_ocl[i], val_onednn[i], tolerance)
                << "tolerance = " << tolerance
                << "\ni = " << i
                << "\nocl[i] = " << val_ocl[i]
                << "\nonednn[i] = " << val_onednn[i];
        }
    }

    layout get_input_layout(T& p, int in_no) {
        auto pad = p.pad;
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        if (in_no == 0)
            return layout{ p.data_type_in0, p.input_format, p.in_shapes.at(0), padding{ pad_ } };
        else if (in_no == 1)
            return layout{ p.data_type_in1, p.input_format, p.in_shapes.at(1), padding{ pad_ } };
        else
            return layout{ p.data_type_in2, p.input_format, p.in_shapes.at(2), padding{ pad_ } };
    }

    cldnn::memory::ptr get_generated_random_1d_mem(cldnn::engine& engine, cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        cldnn::tensor s = l.get_tensor();
        if (l.data_type == cldnn::data_types::i8 || l.data_type == cldnn::data_types::u8) {
            VF<uint8_t> rnd_vec = rg.generate_random_1d<uint8_t>(s.count(), -200, 200);
            set_values(prim, rnd_vec);
        } else if (l.data_type == cldnn::data_types::f16) {
            VF<FLOAT16> rnd_vec = rg.generate_random_1d<FLOAT16>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = rg.generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }
};

class gemm_onednn_ndims : public GemmOneDNNTest<gemm_onednn_test_params> {};
TEST_P(gemm_onednn_ndims, basic) {
    if (!engine.get_device_info().supports_immad)
        return;

    auto p = GetParam();

    auto in_layout0 = get_input_layout(p, 0);
    auto in_layout1 = get_input_layout(p, 1);

    topology_ocl.add(input_layout("input0", in_layout0));
    topology_ocl.add(input_layout("input1", in_layout1));
    topology_ocl.add(gemm("gemm0_ocl", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()));
    topology_ocl.add(reorder("reorder0", input_info("gemm0_ocl"), p.default_format, data_types::f32));

    topology_onednn.add(input_layout("input0", get_input_layout(p, 0)));
    topology_onednn.add(input_layout("input1", get_input_layout(p, 1)));
    topology_onednn.add(gemm("gemm0_onednn", { input_info("input0"), input_info("input1") }, data_types::f32, false, false, 1.f, 0.f, in_layout0.get_rank(), in_layout1.get_rank()));
    topology_onednn.add(reorder("reorder0", input_info("gemm0_onednn"), p.default_format, data_types::f32));

    ov::intel_gpu::ImplementationDesc gemm_impl_ocl = { p.default_format, "", impl_types::ocl };
    config_ocl.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm0_ocl", gemm_impl_ocl } }));

    ov::intel_gpu::ImplementationDesc gemm_impl_onednn = { p.default_format, "", impl_types::onednn };
    config_onednn.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "gemm0_onednn", gemm_impl_onednn } }));

    tolerance = default_tolerance(p.default_type);
    execute(p);
}
#define CASE_GEMM_ONEDNN_FP16_4D { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ONEDNN_FP16_5D { { 1, 3, 4, 4, 4 }, { 1, 3, 4, 4, 4 } }, { 1, 3, 4, 4, 4 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GEMM_ONEDNN_FP16_6D { { 2, 3, 5, 4, 3, 2 }, { 2, 3, 4, 5, 3, 2 } }, { 2, 3, 5, 5, 3, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::f16, data_types::f16, data_types::f16, format::bfwzyx, data_types::f16, format::bfwzyx
#define CASE_GEMM_ONEDNN_I8_4D { { 2, 3, 2, 2 }, { 2, 3, 2, 2 } }, { 2, 3, 2, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::i8, format::bfyx
#define CASE_GEMM_ONEDNN_I8_5D { { 1, 3, 4, 4, 4 }, { 1, 3, 4, 4, 4 } }, { 1, 3, 4, 4, 4 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx
#define CASE_GEMM_ONEDNN_I8_6D { { 2, 3, 5, 4, 3, 2 }, { 2, 3, 4, 5, 3, 2 } }, { 2, 3, 5, 5, 3, 2 }, tensor{ 1 }, tensor{ 0 }, \
data_types::i8, data_types::i8, data_types::i8, format::bfwzyx, data_types::i8, format::bfwzyx

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_onednn_ndims, ::testing::ValuesIn(std::vector<gemm_onednn_test_params>{
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_4D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_5D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_FP16_6D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_4D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_5D },
    gemm_onednn_test_params{ CASE_GEMM_ONEDNN_I8_6D },
}));

TEST(gemm_onednn, impl_replacement_with_cldnn) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    ov::Shape in1_shape = { 1, 1, 3, 4 };
    ov::Shape in2_shape = { 1, 4 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});

    std::vector<float> input1_data = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f,
    };
    set_values(input1, input1_data);
    set_values(input2, input2_data);

    std::vector<float> out_data = {
        8.f, 22.f, 20.f
    };

    topology topology;
    topology.add(input_layout("input1", in1_layout),
                 input_layout("input2", in2_layout),
                 gemm("gemm", { input_info("input1"), input_info("input2") }, data_types::f32, false, true, 1.0f, 0.0f, 4, 2)
    );

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
    ExecutionConfig cfg{ ov::intel_gpu::queue_type(QueueTypes::in_order),
                         ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gemm", fc_impl} }),
                         ov::intel_gpu::optimize_data(true),
                         ov::intel_gpu::allow_new_shape_infer(true) };

    network network(engine, topology, cfg);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("gemm");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gemm").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_FLOAT_EQ(output_ptr[i], out_data[i]);
    }

    // WA: Call cancel() to wait for all queued kernels compilation finish
    network.get_program()->get_compilation_context().cancel();

    // Check if OneDNN's impl is used for the next execute() call
    network.execute();
    inst = network.get_primitive("gemm");
    impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_FALSE(impl->is_dynamic());
}

template <typename gemm_params, typename input0_type, typename input1_type, typename input2_type, typename output_type, typename accumulator_type>
class GemmBaseOneDNNTest : public ::GemmBaseTest<gemm_params, input0_type, input1_type, input2_type, output_type, accumulator_type> {
public:
    virtual ov::intel_gpu::ImplementationDesc getImplementationDesc(gemm_params& p) {
        return { format::bfyx, "", impl_types::onednn };
    }

    void execute(gemm_params& p, bool is_caching_test = false) {
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            return;
        GemmBaseTest<gemm_params, input0_type, input1_type, input2_type, output_type, accumulator_type>::execute(p, is_caching_test);
    }
};

class gemm_int8_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_ONEDNN_4, "" },
}));

class gemm_uint8_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_ONEDNN_4, "" },
}));

class gemm_fp16_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_ONEDNN_4, "" },
}));

class gemm_fp32_simple_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_simple_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_simple_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_ONEDNN_4, "" },
}));

class gemm_int8_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION_ONEDNN, "" },

    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
}));

class gemm_uint8_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TT_TRANSPOSITION_ONEDNN, "" },

    gemm_base_test_params{ CASE_GEMM_UINT8_NN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_NT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TN_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_TT_TRANSPOSITION_LEFTOVERS_ONEDNN, "" },
}));

class gemm_fp16_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_TT_TRANSPOSITION_ONEDNN, "" },
}));

class gemm_fp32_transposition_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_transposition_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_transposition_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_NN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_NT_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_TN_TRANSPOSITION_ONEDNN, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_TT_TRANSPOSITION_ONEDNN, "" },
}));

class gemm_int8_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_fp16_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_fp32_broadcasting_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, int32_t> {};
TEST_P(gemm_fp32_broadcasting_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_broadcasting_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_BROADCASTING_ONEDNN_4, "" },
}));

class gemm_int8_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, int8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_int8_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_ONEDNN_4, "" },
}));

class gemm_uint8_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, uint8_t, int8_t, float, float, int32_t> {};
TEST_P(gemm_uint8_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_uint8_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_UINT8_COMBO_ONEDNN_4, "" },
}));

class gemm_fp16_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP16_COMBO_ONEDNN_4, "" },
}));

class gemm_fp32_combo_tests_onednn : public ::GemmBaseOneDNNTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_combo_tests_onednn, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_combo_tests_onednn, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_1, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_2, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_3, "" },
    gemm_base_test_params{ CASE_GEMM_FP32_COMBO_ONEDNN_4, "" },
}));

#endif // ENABLE_ONEDNN_FOR_GPU

class gemm_int8_transposition_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_transposition_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_transposition_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_NN_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_NT_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_TN_TRANSPOSITION, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_TT_TRANSPOSITION, "gemm_mmad_int8" },
}));

class gemm_int8_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_BROADCAST_4, "gemm_mmad_int8" },
}));

class gemm_int8_leftovers_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_leftovers_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_leftovers_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_4, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_5, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_6, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_7, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_8, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_9, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_10, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_11, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_LEFTOVERS_12, "gemm_mmad_int8" },
}));

class gemm_int8_combo_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_combo_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_combo_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_1, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_2, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_3, "gemm_mmad_int8" },
    gemm_base_test_params{ CASE_GEMM_INT8_COMBO_4, "gemm_mmad_int8" },
}));

class gemm_int8_slm_combo_tests : public ::GemmBaseTest<gemm_base_test_params, int8_t, uint8_t, float, float, int32_t> {};
TEST_P(gemm_int8_slm_combo_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_int8_slm_combo_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_1, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_2, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_3, "gemm_mmad_int8_slm" },
    gemm_base_test_params{ CASE_GEMM_INT8_SLM_COMBO_4, "gemm_mmad_int8_slm" },
}));

class gemm_fp32_tiled_nn_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_nt_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NT_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_tn_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_tn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_tn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TN_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_tt_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_tt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_tt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_TT_4, "gemm_tiled_opt" },
}));

class gemm_fp32_tiled_nn_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, float, float, float, float, float> {};
TEST_P(gemm_fp32_tiled_nn_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp32_tiled_nn_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP32_TILED_NN_BROADCAST_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nn_tests : public ::GemmBaseTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_tiled_nn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nt_tests : public ::GemmBaseTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_tiled_nt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NT_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_tn_tests : public ::GemmBaseTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_tiled_tn_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_tn_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TN_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_tt_tests : public ::GemmBaseTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_tiled_tt_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_tt_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_TT_4, "gemm_tiled_opt" },
}));

class gemm_fp16_tiled_nn_broadcast_tests : public ::GemmBaseTest<gemm_base_test_params, FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16> {};
TEST_P(gemm_fp16_tiled_nn_broadcast_tests, basic) { auto p = GetParam(); execute(p); }

INSTANTIATE_TEST_SUITE_P(gemm_gpu, gemm_fp16_tiled_nn_broadcast_tests, ::testing::ValuesIn(std::vector <gemm_base_test_params> {
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_1, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_2, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_3, "gemm_tiled_opt" },
    gemm_base_test_params{ CASE_GEMM_FP16_TILED_NN_BROADCAST_4, "gemm_tiled_opt" },
}));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(GemmGPUTest, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(GemmGPUTestRandom, basic_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST_P(gemm_int8_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_simple_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_transposition_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_broadcasting_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_uint8_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_combo_tests_onednn, basic_cached) { auto p = GetParam(); execute(p, true); }
#endif // ENABLE_ONEDNN_FOR_GPU

TEST_P(gemm_int8_transposition_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_leftovers_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_combo_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_int8_slm_combo_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_tn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_tt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp32_tiled_nn_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_tn_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_tt_tests, basic_cached) { auto p = GetParam(); execute(p, true); }
TEST_P(gemm_fp16_tiled_nn_broadcast_tests, basic_cached) { auto p = GetParam(); execute(p); }

#endif // RUN_ALL_MODEL_CACHING_TESTS
TEST(gemm_gpu, basic_bfyx_t2_inplace_crop_with_pad_cached) {
    test_basic_bfyx_t2_inplace_crop_with_pad<float>(true);
}
} // namespace
