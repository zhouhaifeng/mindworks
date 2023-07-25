// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "network_test.h"
#include <intel_gpu/runtime/utils.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "compilation_context.hpp"
#include "fully_connected_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
cldnn::format::type layout_4d(cldnn::format f) {
    switch (f.value) {
    case cldnn::format::bfyx:
        return cldnn::format::bfyx;
    case cldnn::format::yxfb:
        return cldnn::format::yxfb;
    default:
        return f.value;
    }
}

template <typename T>
VVVVF<T> fully_connected_reference(VVVVF<T> &input, VVVVF<T> &weights, VF<T> &bias, bool relu = false, T slope = 0.0f) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();         // input is assumed to be bfyx
    size_t output_f = weights.size();       // weights are assumed to be bfyx
    VVVVF<T> output(output_b, VVVF<T>(1, VVF<T>(1, VF<T>(output_f))));
    float res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t n = 0; n < output_f; ++n) {
            res = bias[n];
            for (size_t f = 0; f < input_f; ++f) {
                for (size_t y = 0; y < input_y; ++y) {
                    for (size_t x = 0; x < input_x; ++x) {
                        res += (float)input[b][f][y][x] * (float)weights[n][f][y][x];
                    }
                }
            }
            if (relu && res < (float)0)
                res *= (float)slope;
            output[b][0][0][n] = (T)res;
        }
    }
    return output;
}

template <typename T>
void generic_fully_connected_test(cldnn::format test_input_fmt, cldnn::format test_weights_fmt,
                                  int input_b, int f, int y, int x, int output_f, bool relu, T slope = 0) {
    tests::random_generator rg(GET_SUITE_NAME);
    int min_random = -2, max_random = 2;
    VVVVF<T> input_rnd = rg.generate_random_4d<T>(input_b, f, y, x, min_random, max_random);
    VVVVF<T> weights_rnd = rg.generate_random_4d<T>(output_f, f, y, x, min_random, max_random);
    VF<T> bias_rnd_vec = rg.generate_random_1d<T>(output_f, min_random, max_random);
    VF<T> input_rnd_vec = flatten_4d<T>(test_input_fmt, input_rnd);
    VF<T> weights_rnd_vec = flatten_4d<T>(test_weights_fmt, weights_rnd);

    auto& engine = get_test_engine();
    tensor input_tensor(input_b, f, x, y);
    tensor weights_tensor(output_f, f, x, y);
    auto input = engine.allocate_memory({ type_to_data_type<T>::value, test_input_fmt, input_tensor });
    auto weights = engine.allocate_memory({ type_to_data_type<T>::value, test_weights_fmt, weights_tensor });
    auto bias = engine.allocate_memory({ type_to_data_type<T>::value, format::bfyx, { 1, 1, output_f, 1 } });
    set_values(input, input_rnd_vec);
    set_values(weights, weights_rnd_vec);
    set_values(bias, bias_rnd_vec);

    primitive_id out_id = "fully_connected";
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("bias", bias),
        fully_connected(out_id, input_info("input"), "weights", "bias")
    );
    if (relu)
    {
        topology.add(activation("out", input_info(out_id), activation_func::relu, { slope, 0.0f }));
        out_id = "out";
    }
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, out_id);

    auto output_memory = outputs.at(out_id).get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());

    //ASSERT_EQ(output_layout.format.value, test_input_fmt);
    tensor output_tensor = output_layout.get_tensor();
    int b_size = output_tensor.batch[0];
    int x_size = output_tensor.feature[0];
    ASSERT_EQ(b_size, input_b);
    ASSERT_EQ(x_size, output_f);
    unsigned num_of_operations = f * x * y * 2;
    float ulp = (1.0f / 1024.0f) * num_of_operations;
    bool test_is_correct = true;
    VVVVF<T> output_cpu = fully_connected_reference<T>(input_rnd, weights_rnd, bias_rnd_vec, relu, slope);
    VF<T> output_cpu_vec = flatten_4d<T>(layout_4d(output_layout.format), output_cpu);
    for (size_t i = 0; i < output_cpu_vec.size(); ++i) {
        if (std::abs(float(output_cpu_vec[i]) - float(output_ptr[i])) > ulp) {
            ASSERT_FLOAT_EQ(output_cpu_vec[i], output_ptr[i]); // to print the problematic values
            test_is_correct = false;
            break;
        }
    }

    ASSERT_EQ(test_is_correct, true) << std::endl
        << "failing test parameters:" << std::endl
        << "test_input_fmt = " << format::traits(test_input_fmt).order << std::endl
        << "test_weights_fmt = " << format::traits(test_weights_fmt).order << std::endl
        << "input_b = " << input_b << std::endl
        << "f = " << f << std::endl
        << "y = " << y << std::endl
        << "x = " << x << std::endl
        << "output_f = " << output_f << std::endl
        << "relu = " << relu << std::endl
        << "slope = " << (float)slope << std::endl
        << "type = " << (sizeof(T) == 2 ? "float16" : "float32") << std::endl;
}
}  // namespace

TEST(DISABLED_fully_connected_gpu, generic_random_short) {
    VF<cldnn::format> test_input_fmts = { cldnn::format::bfyx, cldnn::format::yxfb };
    VF<cldnn::format> test_weights_fmts = { cldnn::format::yxfb };
    VF<bool> relu = { true, false };
    std::vector<int> batches = { 1, 2, 4, 8, 16 };
    std::vector<int> features = { 1, 2 };
    std::vector<std::pair<int, int>> input_sizes = { { 28, 28 }, { 64, 64 }, { 100, 100 }, { 227, 227 }, { 1000, 1 }, { 1, 4096 } };
    VF<int> outputs_x = { 5, 16 };

    auto& engine = get_test_engine();
    bool f16_supported = !!engine.get_device_info().supports_fp16;
    if (!f16_supported) {
        std::cout << "[ SKIPPED ] float16 combinations are skipped (cl_khr_fp16 is not supported)." << std::endl;
    }

    for (cldnn::format test_input_fmt : test_input_fmts) {
        for (cldnn::format test_weights_fmt : test_weights_fmts) {
            for (const auto& b : batches) {
                for(const auto& f : features) {
                    for (const auto& sizes : input_sizes) {
                        for (int output_f : outputs_x) {
                            for (bool relu_activated : relu) {
                                generic_fully_connected_test<float>(test_input_fmt, test_weights_fmt,
                                                                    b, f, sizes.second, sizes.first, output_f, relu_activated);
                                if (!f16_supported) continue;
                                generic_fully_connected_test<FLOAT16>(test_input_fmt, test_weights_fmt,
                                                                      b, f, sizes.second, sizes.first, output_f, relu_activated);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(fully_connected_gpu, no_biases) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   no biases
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t input_f = 3, input_b = 1,    // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights");
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(1.5f, output_ptr[0]);
    ASSERT_EQ(0.75f, output_ptr[1]);
    ASSERT_EQ(-2.25f, output_ptr[2]);
    ASSERT_EQ(3.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, no_biases_int8) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  8.0f, 2.0f, -4.0f
    //
    //  Weights:
    //   2.0f    1.0f  0.0f
    //  -3.0f   -2.0f  1.0f
    //   0.0f   -2.0f -4.0f
    //  -5.0f   10.0f  8.0f
    //
    //
    //  Biases:
    //   no biases
    //
    //  Output:
    //  18    -32    12   -52

    const int32_t input_f = 3, input_b = 1,    // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, 1 } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights");
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { input_b, weight_b, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(18.0f, output_ptr[0]);
    ASSERT_EQ(-32.0f, output_ptr[1]);
    ASSERT_EQ(12.0f, output_ptr[2]);
    ASSERT_EQ(-52.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_1) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1} });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.5f, output_ptr[0]);
    ASSERT_EQ(2.75f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(7.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //
    //  Output:
    //   2.5    2.75     0.75   7
    //   4      1        2.75   5

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 2,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(4.00f, output_ptr[1]);
    ASSERT_EQ(2.75f, output_ptr[2]);
    ASSERT_EQ(1.00f, output_ptr[3]);
    ASSERT_EQ(0.75f, output_ptr[4]);
    ASSERT_EQ(2.75f, output_ptr[5]);
    ASSERT_EQ(7.00f, output_ptr[6]);
    ASSERT_EQ(5.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, 2.0, 3.0, 4.0
    //  Output:
    //   2.5    2.75    0.75   7

    const int32_t output_f = 4,                // size of the whole output buffer
                  input_f = 3,                 // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, 2.0f, 3.0f, 4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias")
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc_prim");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(2.75f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(7.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_1_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //
    //  Biases:
    //   1.0,  -2.0,  3.0,  -4.0
    //
    //  Output:
    //   2.5   0      0.75  0

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(0.00f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, xb_f32_batch_2_relu) {
    //  Input  : 3x2
    //  Output : 4x2
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //   1       1.5  0
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //
    //  Output:
    //   2.5    0   0.75   0
    //   4      0   2.75   0

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3, input_b = 2,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::xb_f32, { output_b, { { output_f } }, { 1 } } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 1.0f, 2.0f, 1.5f, 0.5f, 0.0f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(4.00f, output_ptr[1]);
    ASSERT_EQ(0.00f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
    ASSERT_EQ(0.75f, output_ptr[4]);
    ASSERT_EQ(2.75f, output_ptr[5]);
    ASSERT_EQ(0.00f, output_ptr[6]);
    ASSERT_EQ(0.00f, output_ptr[7]);
}

TEST(fully_connected_gpu, x_f32_relu) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   0    0.75  0

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3,                  // size of the whole input buffer
                  weight_b = 4, weight_y = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32, { 1       , { { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_y, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(0.00f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(0.00f, output_ptr[3]);
}

TEST(fully_connected_gpu, x_f32_relu_with_negative_slope) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3
    //  Negative Slope: 0.1
    //
    //  Input:
    //  -0.5     2    0.5
    //
    //  Weights:
    //   1.5     1    0.5
    //  -1       0    0.5
    //   0.5    -0.5 -2
    //  -0.5     1    1.5
    //
    //  Biases:
    //   1.0, -2.0, 3.0, -4.0
    //  Output:
    //   2.5   -0.125    0.75  -0.1

    const int32_t output_f = 4,                 // size of the whole output buffer
                  input_f = 3,                  // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, input_f, 1, 1 } });
    //auto output_prim = memory::allocate({ memory::format::x_f32, { 1       , { { output_f } }, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, output_f, 1 } });

    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });
    set_values(bias_prim, { 1.0f, -2.0f, 3.0f, -4.0f });

    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected("fc_prim", input_info("input"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu_negative_slope, { 0.1f })
    );

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "out");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(2.50f, output_ptr[0]);
    ASSERT_EQ(-0.125f, output_ptr[1]);
    ASSERT_EQ(0.75f, output_ptr[2]);
    ASSERT_EQ(-0.1f, output_ptr[3]);
}

TEST(fully_connected_gpu, b_fs_yx_fsv4)
{
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    const int in_B = 2;
    const int in_F = 2048;
    const int in_Y = 1;
    const int in_X = 1;

    const int W_B = 1000;
    const int W_F = in_F;
    const int W_Y = in_Y;
    const int W_X = in_X;

    const int O_F = W_B;

    // Input data
    std::vector<char> Data(in_F * in_B); // in_X = in_Y = 1
    int i = 0;
    std::generate(Data.begin(), Data.end(), [i]() mutable { return i++ % 9; });
    auto input = engine.allocate_memory({ data_types::i8, format::bfyx, { in_B, in_F, in_X, in_Y } });
    set_values(input, std::move(Data));

    // Create a topology
    topology topology(input_layout("input", input->get_layout()));

    // Reorder
    topology.add(reorder("reorder_in",
                         input_info("input"),
                         layout(data_types::i8, format::b_fs_yx_fsv4, { in_B, in_F, in_X, in_Y })));

    // Weights
    std::vector<char> Weights(W_B * W_F);
    i = 0;
    std::generate(Weights.begin(), Weights.end(), [=]() mutable {
        return i % 2 ? -(i++) / W_F - 1 : (i++) / W_F + 1;
    });
    auto weights_gold =
        engine.allocate_memory({ data_types::i8, format::bfyx, { W_B, W_F, W_X, W_Y } });
    auto weights_imad =
        engine.allocate_memory({ data_types::i8, format::bfyx, { W_B, W_F, W_X, W_Y } });
    set_values(weights_gold, Weights);
    set_values(weights_imad, std::move(Weights));
    topology.add(data("weights_gold", weights_gold), data("weights_imad", weights_imad));

    auto bias_gold = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, O_F, 1, 1 } });
    auto bias_imad = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, O_F, 1, 1 } });

    std::vector<float> bias_data(O_F, 0);
    set_values(bias_gold, bias_data);
    set_values(bias_imad, bias_data);

    topology.add(data("bias_gold", bias_gold));
    topology.add(data("bias_imad", bias_imad));

    // Fully connected
    fully_connected fullc_gold(
        "fullc_gold", input_info("input"), "weights_gold", "bias_gold");
    fully_connected fullc_imad(
        "fullc_imad", input_info("reorder_in"), "weights_imad", "bias_imad");
    topology.add(fullc_gold, fullc_imad);


    auto input_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, W_B, 1, 1 } });
    auto input_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, W_B, 1, 1 } });
    auto output_low_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    auto output_high_mem = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
    set_values(input_low_mem,  rg.generate_random_1d<float>(W_B, -200, 0));
    set_values(input_high_mem, rg.generate_random_1d<float>(W_B, 1, 200));
    set_values(output_low_mem, { -127.0f });
    set_values(output_high_mem, { 127.0f });

    topology.add(data("in_lo", input_low_mem),
        data("in_hi", input_high_mem),
        data("out_lo", output_low_mem),
        data("out_hi", output_high_mem),
        quantize("quant_gold", input_info("fullc_gold"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        quantize("quant_imad", input_info("fullc_imad"), input_info("in_lo"), input_info("in_hi"), input_info("out_lo"), input_info("out_hi"), 255, data_types::i8)
    );

    // Output reorder
    auto reorder_gold =
        reorder("reorder_gold", input_info("quant_gold"), layout(data_types::i8, format::bfyx, { in_B, W_B, 1, 1 }));
    auto reorder_imad =
        reorder("reorder_imad", input_info("quant_imad"), layout(data_types::i8, format::bfyx, { in_B, W_B, 1, 1 }));
    topology.add(reorder_gold, reorder_imad);

    // Network build
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);

    // Network execuiton
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto out_gold = outputs.find("reorder_gold");
    auto out_test = outputs.find("reorder_imad");

    ASSERT_NE(out_gold, outputs.end());
    ASSERT_NE(out_test, outputs.end());
    cldnn::mem_lock<char> gold_ptr(out_gold->second.get_memory(), get_test_stream());
    cldnn::mem_lock<char> test_ptr(out_test->second.get_memory(), get_test_stream());

    ASSERT_EQ(gold_ptr.size(), test_ptr.size());
    for (size_t i = 0; i < gold_ptr.size(); i++) {
        ASSERT_EQ(gold_ptr[i], test_ptr[i]);
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b12) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }
    // Test parameters
    const int batch_num = 12;
    const int output_f = 40;
    const int input_x = 3;
    const int input_y = 3;
    const int input_f = 64;

    // Allocate memory
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_num, input_f, input_y, input_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, output_f, 1 } });

    // Generate random input data and set values
    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_4d<FLOAT16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<FLOAT16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = rg.generate_random_1d<FLOAT16>(output_f, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    set_values(input_prim, input_data_bfyx);
    set_values(weights_prim, weights_data_bfyx);
    set_values(bias_prim, bias_data);

    // Calculate CPU reference
    auto reference_output = fully_connected_reference(input_data, weights_data, bias_data, true);

    // Create topology to test
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc_prim", input_info("input_fsv"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            ASSERT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}

TEST(fully_connected_gpu, DISABLED_fs_byx_fsv32_b34)
{
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_fp16)
    {
        std::cout << "[ SKIPPED ] The test is skipped (cl_khr_fp16 is not supported)." << std::endl;
        ASSERT_EQ(1, 1);
        return;
    }
    // Test parameters
    const int batch_num = 34;
    const int output_f = 40;
    const int input_x = 3;
    const int input_y = 3;
    const int input_f = 64;

    // Allocate memory
    auto input_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { batch_num, input_f, input_y, input_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { output_f, input_f, input_y, input_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 1, output_f, 1 } });

    // Generate random input data and set values
    tests::random_generator rg(GET_SUITE_NAME);
    auto input_data = rg.generate_random_4d<FLOAT16>(batch_num, input_f, input_y, input_x, -1, 1);
    auto weights_data = rg.generate_random_4d<FLOAT16>(output_f, input_f, input_y, input_x, -1, 1);
    auto bias_data = rg.generate_random_1d<FLOAT16>(output_f, -1, 1);

    auto input_data_bfyx = flatten_4d(format::bfyx, input_data);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);

    set_values(input_prim, input_data_bfyx);
    set_values(weights_prim, weights_data_bfyx);
    set_values(bias_prim, bias_data);

    // Calculate CPU reference
    auto reference_output = fully_connected_reference(input_data, weights_data, bias_data, true);

    // Create topology to test
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        reorder("input_fsv", input_info("input"), { data_types::f16, format::fs_b_yx_fsv32, { batch_num, input_f, input_y, input_x } }),
        fully_connected("fc_prim", input_info("input_fsv"), "weights", "bias"),
        activation("out", input_info("fc_prim"), activation_func::relu)
    );

    // Set data optimization to allow weights reordering to optimal format
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();

    auto output_prim = outputs.at("out").get_memory();
    cldnn::mem_lock<FLOAT16> output_ptr(output_prim, get_test_stream());

    for (size_t bi = 0; bi < batch_num; ++bi)
    {
        for (size_t fi = 0; fi < output_f; ++fi)
        {
            auto ref_val = reference_output[bi][0][0][fi];
            auto val = output_ptr[bi * output_f + fi];
            auto equal = floating_point_equal(ref_val, val);

            ASSERT_TRUE(equal);
            if (!equal)
            {
                std::cout << "At b = " << bi << ", f = " << fi << std::endl;
            }
        }
    }
}

using shared_dims = std::tuple<size_t, size_t, size_t>;
using fully_connected_test_params = std::tuple<
    size_t,        // batch_num
    shared_dims,   // input_f input_x input_y
    size_t,        // output_f
    format::type,  // input format
    format::type,  // output format
    std::string,   // kernel
    bool           // is_caching_test
>;

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test : ::testing::TestWithParam<fully_connected_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test() {
        shared_dims dims;
        size_t batch, input_f, input_x, input_y, output_f;
        format::type input_format, output_format;
        std::string kernel;
        bool is_caching_test;

        std::tie(batch, dims, output_f, input_format, output_format, kernel, is_caching_test) = GetParam();
        std::tie(input_f, input_x, input_y) = dims;

        auto input_data = rg.generate_random_4d<InputT>(batch, input_f, input_y, input_x, type_test_ranges<InputT>::min, type_test_ranges<InputT>::max, type_test_ranges<InputT>::k);
        auto weights_data = rg.generate_random_4d<WeightsT>(output_f, input_f, input_y, input_x, type_test_ranges<WeightsT>::min, type_test_ranges<WeightsT>::max, type_test_ranges<WeightsT>::k);
        auto bias_data = rg.generate_random_2d<BiasT>(1, output_f, type_test_ranges<BiasT>::min, type_test_ranges<BiasT>::max, type_test_ranges<BiasT>::k);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected<OutputT>("fc_prim", input, weights, bias, ov::intel_gpu::ImplementationDesc{ output_format, kernel });

        net.run(get_test_default_config(eng, ov::intel_gpu::optimize_data(true)), is_caching_test);
    }
};

using fully_connected_random_test_f32 = fully_connected_random_test<float, float, float, float>;
using fully_connected_random_test_f16 = fully_connected_random_test<FLOAT16, FLOAT16, FLOAT16, FLOAT16>;

TEST_P(fully_connected_random_test_f32, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f32,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx, format::yxfb),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_batched,
    fully_connected_random_test_f32,
    ::testing::Combine(
        ::testing::Values(2, 8),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""),
        ::testing::Values(false))
);

TEST_P(fully_connected_random_test_f16, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_b2,
    fully_connected_random_test_f16,
    ::testing::Combine(
        // Batch 1 is disabled due to sporadic failures in `fully_connected_gpu_bs_f_bsv16_b1`
        // - there are nans in output.
        ::testing::Values(2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::yxfb),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_bfyx_batched,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(2, 8),
        ::testing::Values(shared_dims{3, 1, 1},
                          shared_dims{32, 1, 1}),
        ::testing::Values(3, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    export_import,
    fully_connected_random_test_f16,
    ::testing::Combine(
        ::testing::Values(2),
        ::testing::Values(shared_dims{32, 1, 1}),
        ::testing::Values(32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::bfyx),
        ::testing::Values(""),
        ::testing::Values(true))
);

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct fully_connected_random_test_3d : ::testing::TestWithParam<fully_connected_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test() {
        shared_dims dims;
        size_t batch, input_f, input_x, input_y, output_y;
        format::type input_format, output_format;
        std::string kernel;
        bool is_caching_test;

        std::tie(batch, dims, output_y, input_format, output_format, kernel, is_caching_test) = GetParam();
        std::tie(input_f, input_x, input_y) = dims;

        auto input_data = rg.generate_random_4d<InputT>(batch, input_f, input_y, input_x, type_test_ranges<InputT>::min, type_test_ranges<InputT>::max, type_test_ranges<InputT>::k);
        auto weights_data = rg.generate_random_4d<WeightsT>(output_y, input_y, 1, 1, type_test_ranges<WeightsT>::min, type_test_ranges<WeightsT>::max, type_test_ranges<WeightsT>::k);
        auto bias_data = rg.generate_random_2d<BiasT>(1, output_y, type_test_ranges<BiasT>::min, type_test_ranges<BiasT>::max, type_test_ranges<BiasT>::k);

        auto& eng = get_test_engine();
        auto net = network_test(eng);
        auto input = net.add_input_layout<InputT, 4>("input", input_format, std::move(input_data));
        auto weights = net.add_data<WeightsT, 4>("weights", format::oiyx, std::move(weights_data));
        auto bias = net.add_data<BiasT, 2>("bias", format::bfyx, std::move(bias_data));
        auto fc = net.add_fully_connected_3d<OutputT>("fc_prim", input, weights, bias, ov::intel_gpu::ImplementationDesc{ output_format, kernel }, 3);

        ExecutionConfig config = get_test_default_config(eng);
        config.set_property(ov::intel_gpu::optimize_data(true));
        net.run(config, is_caching_test);
    }
};


using fully_connected_random_test_f32_3d = fully_connected_random_test_3d<float, float, float, float>;
using fully_connected_random_test_f16_3d = fully_connected_random_test_3d<FLOAT16, FLOAT16, FLOAT16, FLOAT16>;
using fully_connected_random_test_i8_3d = fully_connected_random_test_3d<int8_t, int8_t, int8_t, float>;


TEST_P(fully_connected_random_test_f32_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 3},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 3}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_tiled,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{64, 1, 65},
                          shared_dims{64, 1, 128},
                          shared_dims{65, 1, 65},
                          shared_dims{65, 1, 128}),
        ::testing::Values(1, 32, 64),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_big,
    fully_connected_random_test_f32_3d,
    ::testing::Combine(
        ::testing::Values(3),
        ::testing::Values(shared_dims{16, 1, 17},
                          shared_dims{16, 1, 32},
                          shared_dims{32, 1, 17},
                          shared_dims{32, 1, 32}),
        ::testing::Values(17, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

TEST_P(fully_connected_random_test_f16_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_f16_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 16},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 16}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

TEST_P(fully_connected_random_test_i8_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{1, 1, 2},
                          shared_dims{1, 1, 16},
                          shared_dims{3, 1, 2},
                          shared_dims{3, 1, 16}),
        ::testing::Values(1, 3, 16),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_imad,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 2),
        ::testing::Values(shared_dims{64, 1, 65},
                          shared_dims{64, 1, 128},
                          shared_dims{65, 1, 65},
                          shared_dims{65, 1, 128}),
        ::testing::Values(1, 32, 64),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);

INSTANTIATE_TEST_SUITE_P(
    smoke_big,
    fully_connected_random_test_i8_3d,
    ::testing::Combine(
        ::testing::Values(1, 3),
        ::testing::Values(shared_dims{16, 1, 17},
                          shared_dims{16, 1, 32},
                          shared_dims{32, 1, 17},
                          shared_dims{32, 1, 32}),
        ::testing::Values(17, 32),
        ::testing::Values(format::bfyx),
        ::testing::Values(format::any),
        ::testing::Values(""),
        ::testing::Values(false))
);


struct quantization_t {
    VF<float> input_low;
    VF<float> input_high;
    float output_low;
    float output_high;
    int levels;
};

using fully_connected_quantized_test_params = std::tuple<
        size_t,  // batch_num
        size_t,  // input_f
        size_t,  // input_x
        size_t,  // input_y
        size_t,  // output_f
        format::type  // format
>;

template <typename InputT, typename OutputT>
class fully_connected_quantized_test : public ::testing::Test {
private:
    size_t batch_num() { return _input.size(); }
    size_t input_f() { return _input[0].size(); }
    size_t input_y() { return _input[0][0].size(); }
    size_t input_x() { return _input[0][0][0].size(); }
    size_t output_f() { return _weights.size(); }

    data_types input_data_type() {
        return type_to_data_type<InputT>::value;
    }

    data_types output_data_type() {
        return type_to_data_type<OutputT>::value;
    }

    bool has_bias() { return _bias.size() > 0; }

public:
    static std::string PrintToStringParamName(testing::TestParamInfo<fully_connected_quantized_test_params> param_info) {
        // construct a readable name
        return std::to_string(param_info.index) + "_in_" + std::to_string(testing::get<0>(param_info.param))
               + "x" + std::to_string(testing::get<1>(param_info.param))
               + "x" + std::to_string(testing::get<2>(param_info.param))
               + "x" + std::to_string(testing::get<3>(param_info.param))
               + "_of_" + std::to_string(testing::get<4>(param_info.param))
               + "_" + fmt_to_str(testing::get<5>(param_info.param));
    }

    void set_input(VVVVF<InputT> _data) {
        _input = std::move(_data);
    }

    void set_weights(VVVVF<int8_t> _data) {
        _weights = std::move(_data);
    }

    void set_bias(VF<int> _data) {
        _bias = std::move(_data);
    }

    void set_quantization(quantization_t quant_data) {
        _quantization = std::move(quant_data);
    }

    void set_input_format(format::type fmt) {
        _fmt = fmt;
    }

    void run_test(VVF<OutputT> expected) {
        auto& engine = get_test_engine();

        auto input_size = tensor(TensorValue(batch_num()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));
        auto weights_size = tensor(TensorValue(output_f()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));

        auto input_prim = engine.allocate_memory({ input_data_type(), _fmt, input_size });
        auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, weights_size });
        auto quantization_input_low = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(output_f())) });
        auto quantization_input_high = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(output_f())) });
        auto quantization_output_low = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(1)) });
        auto quantization_output_high = engine.allocate_memory({ data_types::f32, format::bfyx, tensor(feature(1)) });

        VF<InputT> input_flattened(input_prim->get_layout().get_linear_size());
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_f(); ++fi)
                for (size_t yi = 0; yi < input_y(); ++yi)
                    for (size_t xi = 0; xi < input_x(); ++xi) {
                        auto idx = tensor((int32_t)bi, (int32_t)fi, (int32_t)xi, (int32_t)yi);
                        auto offset = input_size.get_linear_offset(idx, _fmt);
                        input_flattened[offset] = _input[bi][fi][yi][xi];
                    }

        set_values(input_prim, input_flattened);
        set_values(weights_prim, flatten_4d(format::bfyx, _weights));
        set_values(quantization_input_low, _quantization.input_low);
        set_values(quantization_input_high, _quantization.input_high);
        set_values(quantization_output_low, { _quantization.output_low });
        set_values(quantization_output_high, { _quantization.output_high });

        auto bias_prim = engine.allocate_memory({ data_types::i32, format::bfyx, tensor(feature(output_f())) });
        set_values(bias_prim, _bias);

        topology topo;
        topo.add(data("weights", weights_prim));
        topo.add(data("bias", bias_prim));

        topo.add(input_layout("input", input_prim->get_layout()));

        auto input_sizes = input_size.sizes();
        auto last_dim = std::find_if(input_sizes.rbegin(), input_sizes.rend(),
                                     [](tensor::value_type x) { return x != 1l; });
        size_t input_rank = std::distance(input_sizes.begin(), last_dim.base());
        auto fc_prim = fully_connected("fc_prim", input_info("input"), "weights", "bias", cldnn::padding(), input_rank);
        fc_prim.output_data_types = {type_to_data_type<OutputT>::value};
        topo.add(fc_prim);

        topo.add(data("quant_input_low", quantization_input_low));
        topo.add(data("quant_input_high", quantization_input_high));
        topo.add(data("quant_output_low", quantization_output_low));
        topo.add(data("quant_output_high", quantization_output_high));
        topo.add(quantize("quantization_prim",
            input_info("fc_prim"),
            input_info("quant_input_low"),
            input_info("quant_input_high"),
            input_info("quant_output_low"),
            input_info("quant_output_high"),
            _quantization.levels,
            output_data_type()
            ));

        topo.add(reorder("output", input_info("quantization_prim"), format::bfyx, output_data_type()));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network net(engine, topo, config);
        net.set_input_data("input", input_prim);

        auto output = net.execute();
        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<OutputT> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num(); ++bi) {
            for (size_t fi = 0; fi < output_f(); ++fi) {
                ASSERT_NEAR(out_ptr[bi * output_f() + fi], expected[bi][fi], 1) << "at b = " << bi << ", fi = " << fi;
            }
        }
    }

private:
    VVVVF<InputT> _input;
    VVVVF<int8_t> _weights;
    VF<int> _bias;
    quantization_t _quantization;
    format::type _fmt;
};


template <typename OutputT, typename AccT, typename InputT, typename WeightsT, typename BiasT>
VVF<OutputT> ref_fully_connected(
    const VVVVF<InputT>& input,
    const VVVVF<WeightsT>& weights,
    const VF<BiasT>& bias,
    const quantization_t& quantization) {

    auto batch_num = input.size();
    auto input_f = input[0].size();
    auto input_y = input[0][0].size();
    auto input_x = input[0][0][0].size();

    auto output_f = weights.size();

    auto output = VVF<OutputT>(batch_num, VF<OutputT>(output_f));

    for (size_t bi = 0; bi < batch_num; ++bi)
    for (size_t fi = 0; fi < output_f; ++fi) {
        AccT acc = static_cast<AccT>(0);
        for (size_t ifi = 0; ifi < input_f; ++ifi)
        for (size_t iyi = 0; iyi < input_y; ++iyi)
        for (size_t ixi = 0; ixi < input_x; ++ixi) {
            auto input_val = static_cast<AccT>(input[bi][ifi][iyi][ixi]);
            auto weights_val = static_cast<AccT>(weights[fi][ifi][iyi][ixi]);
            acc += input_val * weights_val;
        }
        acc += static_cast<AccT>(bias[fi]);

        //quantization
        auto input_low = quantization.input_low[fi];
        auto input_high = quantization.input_high[fi];
        auto output_low = quantization.output_low;
        auto output_high = quantization.output_high;
        float levels = static_cast<float>(quantization.levels); // just to get correct output values
        if (acc <= input_low)
            output[bi][fi] = static_cast<OutputT>(output_low);
        else if (acc > input_high)
            output[bi][fi] = static_cast<OutputT>(output_high);
        else {
            if (std::is_same<OutputT, float>::value) {
                output[bi][fi] = static_cast<OutputT>(
                    std::round((acc - input_low) / (input_high - input_low) * (levels - 1))
                        * (1 / (levels - 1) * (output_high - output_low))
                        + output_low);
            }
            else {
                output[bi][fi] = static_cast<OutputT>(std::round(
                    std::round((acc - input_low) / (input_high - input_low) * (levels - 1))
                        * (1 / (levels - 1) * (output_high - output_low))
                        + output_low));
            }
        }
    }
    return output;
}


template <typename InputT, typename OutputT>
class fc_quantized_random_test
    : public fully_connected_quantized_test<InputT, OutputT>
    , public ::testing::WithParamInterface< fully_connected_quantized_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

public:
    void run_random_test() {
        size_t b, in_f, in_x, in_y, out_f;
        format::type in_fmt;

        std::tie(b, in_f, in_x, in_y, out_f, in_fmt) = GetParam();

        VVVVF<InputT> input_data = rg.generate_random_4d<InputT>(b, in_f, in_y, in_x, -127, 127);
        VVVVF<int8_t> weights_data = rg.generate_random_4d<int8_t>(out_f, in_f, in_y, in_x, -127, 127);
        VF<int> bias_data = rg.generate_random_1d<int>(out_f, -127, 127);
        bool is_unsigned = std::is_same<OutputT, uint8_t>::value;
        quantization_t quant_data;
        quant_data.input_low   = rg.generate_random_1d<float>(out_f, -200, 0);
        quant_data.input_high  = rg.generate_random_1d<float>(out_f, 1, 200);
        quant_data.output_low  = is_unsigned ? 0.0f   : -127.0f;
        quant_data.output_high = is_unsigned ? 255.0f : 127.0f;
        quant_data.levels      = is_unsigned ? 256    : 255;

        this->set_input(input_data);
        this->set_weights(weights_data);
        this->set_bias(bias_data);
        this->set_quantization(quant_data);
        this->set_input_format(in_fmt);

        this->run_test(ref_fully_connected<OutputT, float>(input_data, weights_data, bias_data, quant_data));
    }
};

using fully_connected_i8_i8_test = fc_quantized_random_test<int8_t, int8_t>;
using fully_connected_i8_u8_test = fc_quantized_random_test<int8_t, uint8_t>;
using fully_connected_i8_f32_test = fc_quantized_random_test<int8_t, float>;

using fully_connected_u8_i8_test = fc_quantized_random_test<uint8_t, int8_t>;
using fully_connected_u8_u8_test = fc_quantized_random_test<uint8_t, uint8_t>;
using fully_connected_u8_f32_test = fc_quantized_random_test<uint8_t, float>;

TEST_P(fully_connected_i8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_i8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_i8_f32_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_u8_f32_test, random) {
    run_random_test();
}

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(16, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    imad,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(64, 128),
        testing::Values(1),
        testing::Values(1),
        testing::Values(1, 31, 64, 65),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    mmad,
    fully_connected_i8_i8_test,
    testing::Combine(
        testing::Values(1),
        testing::Values(16, 43, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32, 64),
        testing::Values(format::bfyx, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(16, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(16, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_i8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_i8_f32_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_u8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 32),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv32)
    ),
    fully_connected_u8_f32_test::PrintToStringParamName
);

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST(fully_connected_onednn, impl_replacement_with_cldnn) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    const int32_t input_f = 3, input_b = 1, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx });
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

    set_values(input_data, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
    ExecutionConfig cfg{ ov::intel_gpu::queue_type(QueueTypes::in_order),
                         ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl} }),
                         ov::intel_gpu::optimize_data(true),
                         ov::intel_gpu::allow_new_shape_infer(true) };

    network network(engine, topology, cfg);
    network.set_input_data("input", input_data);

    // Check if shape agnostic kernel is used as default impl or not
    auto inst = network.get_primitive("fc");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, 8)); // fake_alignment
    ASSERT_EQ(out_l.batch(), input_b);
    ASSERT_EQ(out_l.feature(), weight_b);
    ASSERT_EQ(out_l.spatial(0), 1);
    ASSERT_EQ(out_l.spatial(1), 1);

    cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

    ASSERT_EQ(1.5f, output_ptr[0]);
    ASSERT_EQ(0.75f, output_ptr[1]);
    ASSERT_EQ(-2.25f, output_ptr[2]);
    ASSERT_EQ(3.0f, output_ptr[3]);

    // WA: Call cancel() to wait for all queued kernels compilation finish
    network.get_program()->get_compilation_context().cancel();

    // Check if OneDNN's impl is used for the next execute() call
    network.execute();
    inst = network.get_primitive("fc");
    impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_FALSE(impl->is_dynamic());
}

TEST(fully_connected_onednn_gpu, no_biases_int8) {
    //  Input  : 3x1
    //  Output : 4x1
    //  Weights: 4x3

    const int32_t input_f = 3, input_b = 1,     // size of the whole input buffer
                  weight_b = 4, weight_f = 3;   // size of the whole weights buffer

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    // Change input data of fully-connected node from bx to bf
    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_b, weight_f, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, 1 } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights");
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { input_b, 4, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl} }));
    network network(engine, topology, cfg);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    ASSERT_EQ(18.0f, output_ptr[0]);
    ASSERT_EQ(-32.0f, output_ptr[1]);
    ASSERT_EQ(12.0f, output_ptr[2]);
    ASSERT_EQ(-52.0f, output_ptr[3]);
}

TEST(fully_connected_3d_onednn_gpu, no_biases_int8) {
    //  Input  : 1x2x3x1 (3D FC case)
    //  Output : 2x4
    //  Weights: 4x3

    const int32_t input_y = 3, input_f = 2, input_b = 1,  // size of the whole input buffer
                  weight_o = 4, weight_i = 3,             // size of the whole weights buffer
                  output_b = 2, output_f = 4;

    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { input_b, input_f, 1, input_y } });
    auto weights_prim = engine.allocate_memory({ data_types::i8, format::bfyx, { weight_o, weight_i, 1, 1 } });

    set_values(input_prim, { 8.4f, 2.3f, -4.49f, 8.4f, 2.3f, -4.49f });
    set_values<int8_t>(weights_prim, { 2, 1, 0, -3, -2, 1, 0, -2, -4, -5, 10, 8 });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto ri = reorder("reorder_to_int", input_info("input"), { data_types::i8, format::bfyx, { input_b, input_f, 1, input_y } });
    auto fc = fully_connected("fc_prim", input_info("reorder_to_int"), "weights", "", padding(), 3);
    auto rf = reorder("reorder_to_float", input_info("fc_prim"), { data_types::f32, format::bfyx, { output_b, output_f, 1, 1 } });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(ri);
    topology.add(rf);

    ov::intel_gpu::ImplementationDesc fc_impl = { format::bfyx, "", impl_types::onednn };
    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc_prim", fc_impl} }));

    network network(engine, topology, cfg);
    network.set_input_data("input", input_prim);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reorder_to_float");

    auto output_prim = outputs.begin()->second.get_memory();

    cldnn::mem_lock<float> output_ptr (output_prim, get_test_stream());

    for (int b = 0; b < output_b; b++) {
        ASSERT_EQ(18.0f, output_ptr[b * output_f + 0]);
        ASSERT_EQ(-32.0f, output_ptr[b * output_f + 1]);
        ASSERT_EQ(12.0f, output_ptr[b * output_f + 2]);
        ASSERT_EQ(-52.0f, output_ptr[b * output_f + 3]);
    }
}
#endif

TEST(fully_connected_gpu, dynamic) {
    auto& engine = get_test_engine();

    const int32_t input_f = 3, input_b = 1, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx });
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

    set_values(input_data, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_data);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, 8)); // fake_alignment
    ASSERT_EQ(out_l.batch(), input_b);
    ASSERT_EQ(out_l.feature(), weight_b);
    ASSERT_EQ(out_l.spatial(0), 1);
    ASSERT_EQ(out_l.spatial(1), 1);

    cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

    ASSERT_EQ(1.5f, output_ptr[0]);
    ASSERT_EQ(0.75f, output_ptr[1]);
    ASSERT_EQ(-2.25f, output_ptr[2]);
    ASSERT_EQ(3.0f, output_ptr[3]);
}

TEST(fully_connected_gpu, dynamic_6d_input) {
    auto& engine = get_test_engine();

    const int32_t input_b = 1, input_f = 3, input_w = 2, input_z = 1, input_y = 1, input_x = 4;
    const int32_t weight_b = 2;

    auto input_dyn_layout = layout{ov::PartialShape{ov::Dimension(-1), input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx};
    auto input_data = engine.allocate_memory(layout{ov::PartialShape{input_b, input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx});
    auto weights_data = engine.allocate_memory({ov::PartialShape{weight_b, input_x}, data_types::f32, format::bfyx });

    set_values(input_data, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                            -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                            -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});
    set_values(weights_data, {1.5f, 1.0f, -1.0f, 0.0f,
                              0.5f, -0.5f, -0.5f, 1.0f, });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights", "", cldnn::padding(), input_dyn_layout.get_rank())
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_data);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ASSERT_EQ(output_prim_mem->get_layout().batch(), 1);
    ASSERT_EQ(out_l.batch(), 1);
    ASSERT_EQ(out_l.feature(), 3);
    ASSERT_EQ(out_l.spatial(0), 2);
    ASSERT_EQ(out_l.spatial(1), 1);
    ASSERT_EQ(out_l.spatial(2), 1);
    ASSERT_EQ(out_l.spatial(3), 2);

    std::vector<float> expected_output = {
        0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5
    };

    cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

    for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
        ASSERT_EQ(expected_output[i], output_ptr[i]);
    }
}

TEST(fully_connected_gpu, static_6d_input) {
    auto& engine = get_test_engine();

    const int32_t input_b = 1, input_f = 3, input_w = 2, input_z = 1, input_y = 1, input_x = 4;
    const int32_t weight_b = 2;

    auto input_dyn_layout = layout{ov::PartialShape{input_b, input_f, input_w, input_z, input_y, input_x}, data_types::f32, format::bfwzyx};
    auto input_data = engine.allocate_memory(input_dyn_layout);
    auto weights_data = engine.allocate_memory({ov::PartialShape{weight_b, input_x}, data_types::f32, format::bfyx });

    set_values(input_data, {-0.5f, 2.0f, 0.5f, 1.f,  -1.5f, 2.0f, 0.5f, 1.f,
                            -0.5f, 2.5f, 0.5f, 1.f,  -0.5f, 3.0f, 0.5f, 1.f,
                            -0.5f, 2.0f, 0.5f, 1.f,  -0.5f, 2.0f, 2.5f, 1.f});
    set_values(weights_data, {1.5f, 1.0f, -1.0f, 0.0f,
                              0.5f, -0.5f, -0.5f, 1.0f, });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights", "", cldnn::padding(), input_dyn_layout.get_rank()),
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_data);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ASSERT_EQ(output_prim_mem->get_layout().batch(), 6);
    ASSERT_EQ(out_l.batch(), 6);
    ASSERT_EQ(out_l.feature(), 2);
    ASSERT_EQ(out_l.spatial(0), 1);
    ASSERT_EQ(out_l.spatial(1), 1);

    std::vector<float> expected_output = {
        0.75, -0.5, -0.75, -1, 1.25, -0.75, 1.75, -1, 0.75, -0.5, -1.25, -1.5
    };

    cldnn::mem_lock<float> output_ptr(output_prim_mem, get_test_stream());

    for (size_t i = 0 ; i < out_l.get_linear_size(); i++) {
        ASSERT_EQ(expected_output[i], output_ptr[i]);
    }
}

TEST(fully_connected_gpu, dynamic_multi_inference_same_shape) {
    auto& engine = get_test_engine();
    const int32_t input_f = 3, input_b = 1, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx };
    auto input_data1 = engine.allocate_memory(input_actual_layout);
    auto input_data2 = engine.allocate_memory(input_actual_layout);
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

    set_values(input_data1, { 0.5f, -2.0f, -0.5f });
    set_values(input_data2, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                              -1.0f, 0.0f, 0.5f,
                              0.5f, -0.5f, -2.0f,
                              -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    {
        network.set_input_data("input", input_data1);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network.get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, 8)); // fake_alignment
        ASSERT_EQ(out_l.batch(), input_b);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(-1.5f, output_ptr[0]);
        ASSERT_EQ(-0.75f, output_ptr[1]);
        ASSERT_EQ(2.25f, output_ptr[2]);
        ASSERT_EQ(-3.0f, output_ptr[3]);
    }

    {
        network.set_input_data("input", input_data2);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network.get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, 8)); // fake_alignment
        ASSERT_EQ(out_l.batch(), input_b);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(1.5f, output_ptr[0]);
        ASSERT_EQ(0.75f, output_ptr[1]);
        ASSERT_EQ(-2.25f, output_ptr[2]);
        ASSERT_EQ(3.0f, output_ptr[3]);
    }
}

TEST(fully_connected_gpu, dynamic_multi_inference_different_shape) {
    auto& engine = get_test_engine();

    const int32_t input_f = 3, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_actual_layout1 = layout{ ov::PartialShape{ 2, input_f }, data_types::f32,format::bfyx};
    auto input_actual_layout2 = layout{ ov::PartialShape{ 1, input_f }, data_types::f32,format::bfyx};
    auto input_data1 = engine.allocate_memory(input_actual_layout1);
    auto input_data2 = engine.allocate_memory(input_actual_layout2);
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});

    set_values(input_data1, { 0.5f, -2.0f, -0.5f,
                              -0.5f, 2.0f, 0.5f });
    set_values(input_data2, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                              -1.0f, 0.0f, 0.5f,
                              0.5f, -0.5f, -2.0f,
                              -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto inst = network.get_primitive("fc");
    ASSERT_TRUE(inst->is_dynamic());

    {
        network.set_input_data("input", input_data1);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network.get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(2, 8)); // fake_alignment
        ASSERT_EQ(out_l.batch(), 2);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(-1.5f, output_ptr[0]);
        ASSERT_EQ(-0.75f, output_ptr[1]);
        ASSERT_EQ(2.25f, output_ptr[2]);
        ASSERT_EQ(-3.0f, output_ptr[3]);

        ASSERT_EQ(1.5f, output_ptr[4]);
        ASSERT_EQ(0.75f, output_ptr[5]);
        ASSERT_EQ(-2.25f, output_ptr[6]);
        ASSERT_EQ(3.0f, output_ptr[7]);
    }

    {
        network.set_input_data("input", input_data2);

        auto outputs = network.execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "fc");

        auto output_prim_mem = outputs.begin()->second.get_memory();

        auto out_l = network.get_output_layout(outputs.begin()->first);
        ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(1, 8)); // fake_alignment
        ASSERT_EQ(out_l.batch(), 1);
        ASSERT_EQ(out_l.feature(), weight_b);
        ASSERT_EQ(out_l.spatial(0), 1);
        ASSERT_EQ(out_l.spatial(1), 1);

        cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

        ASSERT_EQ(1.5f, output_ptr[0]);
        ASSERT_EQ(0.75f, output_ptr[1]);
        ASSERT_EQ(-2.25f, output_ptr[2]);
        ASSERT_EQ(3.0f, output_ptr[3]);
    }
}

TEST(fully_connected_gpu, dynamic_multi_inference_multiple_shapes) {
    auto& engine = get_test_engine();

    const int32_t input_f = 3, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_actual_layout1 = layout{ ov::PartialShape{ 2, input_f }, data_types::f32,format::bfyx};
    auto input_actual_layout2 = layout{ ov::PartialShape{ 1, input_f }, data_types::f32,format::bfyx};
    auto input_data1 = engine.allocate_memory(input_actual_layout1);
    auto input_data2 = engine.allocate_memory(input_actual_layout2);
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx});

    set_values(input_data1, { 0.5f, -2.0f, -0.5f,
                              -0.5f, 2.0f, 0.5f });
    set_values(input_data2, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f,
                              -1.0f, 0.0f, 0.5f,
                              0.5f, -0.5f, -2.0f,
                              -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    // Call different shape multiple times to ensure caching works fine
    for (size_t i = 0; i < 2; i++) {
        {
            network.set_input_data("input", input_data1);

            auto outputs = network.execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network.get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(2, 8)); // fake_alignment
            ASSERT_EQ(out_l.batch(), 2); // fake_alignment
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(-1.5f, output_ptr[0]);
            ASSERT_EQ(-0.75f, output_ptr[1]);
            ASSERT_EQ(2.25f, output_ptr[2]);
            ASSERT_EQ(-3.0f, output_ptr[3]);

            ASSERT_EQ(1.5f, output_ptr[4]);
            ASSERT_EQ(0.75f, output_ptr[5]);
            ASSERT_EQ(-2.25f, output_ptr[6]);
            ASSERT_EQ(3.0f, output_ptr[7]);
        }

        {
            network.set_input_data("input", input_data2);

            auto outputs = network.execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network.get_output_layout(outputs.begin()->first);
            ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(1, 8)); // fake_alignment
            ASSERT_EQ(out_l.batch(), 1); // fake_alignment
            ASSERT_EQ(out_l.feature(), weight_b);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), 1);

            cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

            ASSERT_EQ(1.5f, output_ptr[0]);
            ASSERT_EQ(0.75f, output_ptr[1]);
            ASSERT_EQ(-2.25f, output_ptr[2]);
            ASSERT_EQ(3.0f, output_ptr[3]);
        }
    }
}

namespace {
    template <typename OutputT,
              typename InputT,
              typename WeightsT,
              typename BiasT,
              typename AccT = OutputT>
    VF<OutputT> dynamic_fully_connected_reference_calc(ov::Dimension::value_type batch,
                                                       ov::Dimension::value_type input_f,
                                                       ov::Dimension::value_type output_f,
                                                       VF<InputT>& input,
                                                       VF<WeightsT>& weights,
                                                       VF<BiasT>& bias) {
        VF<OutputT> result(batch * output_f);
        for (int b = 0; b < batch; b++) {
            for (int ofm = 0; ofm < output_f; ofm++) {
                AccT acc = static_cast<AccT>(bias[ofm]);
                for (int ifm = 0; ifm < input_f; ifm++) {
                    acc += weights[ofm * input_f + ifm] * input[b * input_f + ifm];
                }
                result[b * output_f + ofm] = acc;
            }
        }

        return result;
    }
} // namespace

using fully_connected_dynamic_test_params = std::tuple<
    std::vector<ov::Dimension::value_type>, // batch_sizes
    ov::Dimension::value_type,              // input_f
    ov::Dimension::value_type,              // output_f
    bool                                    // 3D case
>;

template <typename InputT, typename WeightsT, typename BiasT, typename OutputT>
struct dynamic_fully_connected_gpu : ::testing::TestWithParam<fully_connected_dynamic_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void run_test() {
        std::vector<ov::Dimension::value_type> batch_sizes;
        ov::Dimension::value_type input_f;
        ov::Dimension::value_type output_f;
        bool fc_3d = false;

        std::tie(batch_sizes, input_f, output_f, fc_3d) = GetParam();

        auto input_dt = cldnn::type_to_data_type<InputT>::value;
        auto weights_dt = cldnn::type_to_data_type<WeightsT>::value;
        auto output_dt = cldnn::type_to_data_type<OutputT>::value;

        auto& engine = get_test_engine();
        auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), input_f }, input_dt, format::bfyx };
        if (fc_3d)
            input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(), ov::Dimension(), input_f }, input_dt, format::bfyx };

        auto weights_mem = engine.allocate_memory({ ov::PartialShape{ output_f, input_f }, weights_dt, format::bfyx });
        auto weights_data_vec = rg.generate_random_1d<WeightsT>(output_f * input_f, -1, 1);

        auto bias_mem = engine.allocate_memory({ ov::PartialShape{ output_f }, output_dt, format::bfyx });
        auto bias_data_vec = rg.generate_random_1d<OutputT>(output_f, 0, 1);

        set_values(weights_mem, weights_data_vec);
        set_values(bias_mem, bias_data_vec);

        cldnn::topology topology{
            input_layout("input", input_dyn_layout),
            data("weights", weights_mem),
            data("bias", bias_mem),
        };

        if (fc_3d)
            topology.add(fully_connected("fc", input_info("input"), "weights", "bias", output_dt, padding(), 3));
        else
            topology.add(fully_connected("fc", input_info("input"), "weights", "bias", output_dt));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);

        for (const auto& batch_size : batch_sizes) {
            auto input_actual_layout = layout{ ov::PartialShape{ batch_size, input_f }, input_dt, format::bfyx };
            if (fc_3d)
                input_actual_layout = layout{ ov::PartialShape{ 1, batch_size, input_f }, input_dt, format::bfyx };
            cldnn::memory_ptr input_mem = engine.allocate_memory(input_actual_layout);
            std::vector<InputT> input_data_vec = rg.generate_random_1d<InputT>(batch_size * input_f, 0, 1);
            set_values(input_mem, input_data_vec);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            ASSERT_EQ(outputs.size(), size_t(1));
            ASSERT_EQ(outputs.begin()->first, "fc");

            auto output_prim_mem = outputs.begin()->second.get_memory();

            auto out_l = network.get_output_layout(outputs.begin()->first);
            ASSERT_EQ(out_l.batch(), fc_3d ? 1 : batch_size);
            ASSERT_EQ(out_l.feature(), fc_3d ? batch_size : output_f);
            ASSERT_EQ(out_l.spatial(0), 1);
            ASSERT_EQ(out_l.spatial(1), fc_3d ? output_f : 1);

            cldnn::mem_lock<OutputT> output_ptr(output_prim_mem, get_test_stream());

            auto ref_result = dynamic_fully_connected_reference_calc<OutputT>(batch_size,
                                                                              input_f,
                                                                              output_f,
                                                                              input_data_vec,
                                                                              weights_data_vec,
                                                                              bias_data_vec);

            if (engine.get_device_info().supports_immad) {
                for (int b = 0; b < batch_size; b++) {
                    for (int ofm = 0; ofm < output_f; ofm++) {
                        EXPECT_NEAR(ref_result[b * output_f + ofm], output_ptr[b * output_f + ofm],
                                    default_tolerance(input_dt));
                    }
                }
            } else {
                for (int b = 0; b < batch_size; b++) {
                    for (int ofm = 0; ofm < output_f; ofm++) {
                        ASSERT_EQ(ref_result[b * output_f + ofm], output_ptr[b * output_f + ofm]);
                    }
                }
            }
        }
    }
};

using dynamic_fully_connected_gpu_f32_3d = dynamic_fully_connected_gpu<float, float, float, float>;
using dynamic_fully_connected_gpu_f16_3d = dynamic_fully_connected_gpu<FLOAT16, FLOAT16, FLOAT16, FLOAT16>;
using dynamic_fully_connected_gpu_i8_3d = dynamic_fully_connected_gpu<int8_t, int8_t, int8_t, float>;

static const std::vector<ov::Dimension::value_type>
    dyn_batches_full = {1, 2, 4, 7, 8, 9, 15, 16, 31, 32, 33, 47, 48, 49, 58, 63, 64};
static const std::vector<ov::Dimension::value_type>
    dyn_batches_smoke = {1, 2, 7, 8, 9, 16, 32, 33, 47, 48, 58};

TEST_P(dynamic_fully_connected_gpu_f32_3d, basic) {
    run_test();
}

TEST_P(dynamic_fully_connected_gpu_f16_3d, basic) {
    run_test();
}

TEST_P(dynamic_fully_connected_gpu_i8_3d, basic) {
    run_test();
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_f32_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_f16_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    smoke,
    dynamic_fully_connected_gpu_i8_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_smoke),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_f32_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_f16_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

INSTANTIATE_TEST_SUITE_P(
    full,
    dynamic_fully_connected_gpu_i8_3d,
    ::testing::Combine(
        ::testing::Values(dyn_batches_full),
        ::testing::Values(10, 32, 42, 53, 64, 128),
        ::testing::Values(2, 9, 16, 32, 64, 128),
        ::testing::Values(false, true))
);

TEST(fully_connected_gpu, has_cached_weights_reorder) {
    auto& engine = get_test_engine();

    const int32_t input_f = 3, input_b = 1, weight_b = 4;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), input_f }, data_types::f32,format::bfyx };
    auto input_data = engine.allocate_memory(layout{ ov::PartialShape{ input_b, input_f }, data_types::f32,format::bfyx });
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ weight_b, input_f }, data_types::f32,format::bfyx });

    set_values(input_data, { -0.5f, 2.0f, 0.5f });
    set_values(weights_data, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl_desc} })),
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input_data);

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "fc");

    auto output_prim_mem = outputs.begin()->second.get_memory();

    auto inst = network.get_primitive("fc");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto reorder_kernel_params = impl->get_weights_reorder_kernel_params();
    ASSERT_TRUE(reorder_kernel_params != nullptr);
    auto reorder_impl = network.get_program()->get_implementations_cache().get(*reorder_kernel_params);
    ASSERT_TRUE(reorder_impl != nullptr);

    auto out_l = network.get_output_layout(outputs.begin()->first);
    ASSERT_EQ(output_prim_mem->get_layout().batch(), align_to(input_b, 8)); // fake_alignment
    ASSERT_EQ(out_l.batch(), input_b);
    ASSERT_EQ(out_l.feature(), weight_b);
    ASSERT_EQ(out_l.spatial(0), 1);
    ASSERT_EQ(out_l.spatial(1), 1);

    cldnn::mem_lock<float> output_ptr (output_prim_mem, get_test_stream());

    ASSERT_EQ(1.5f, output_ptr[0]);
    ASSERT_EQ(0.75f, output_ptr[1]);
    ASSERT_EQ(-2.25f, output_ptr[2]);
    ASSERT_EQ(3.0f, output_ptr[3]);
}

template <typename InputT, typename T>
VVF<T> fully_connected_types_reference(VVVVF<InputT> &input, VVVVF<T> &weights, VF<T> &bias, const quantization_t& quantization, bool relu = false, T slope = 0.0f) {
    size_t input_f = input[0].size();
    size_t input_y = input[0][0].size();
    size_t input_x = input[0][0][0].size();
    size_t output_b = input.size();         // input is assumed to be bfyx
    size_t output_f = weights.size();       // weights are assumed to be bfyx
    VVF<T> output(output_b, VF<T>(output_f));
    float res;
    for (size_t b = 0; b < output_b; ++b) {
        for (size_t n = 0; n < output_f; ++n) {
            res = bias[n];
            for (size_t f = 0; f < input_f; ++f) {
                for (size_t y = 0; y < input_y; ++y) {
                    for (size_t x = 0; x < input_x; ++x) {
                        res += (float)input[b][f][y][x] * (float)weights[n][f][y][x];
                    }
                }
            }
            if (relu && res < (float)0)
                res *= (float)slope;
            if (res > quantization.output_high)
                output[b][n] = quantization.output_high;
            else {
                if (res < quantization.output_low)
                    output[b][n] = quantization.output_low;
                else
                    output[b][n] = (T)res;
            }
        }
    }
    return output;
}

using fully_connected_types_test_params = std::tuple<
        size_t,  // batch_num
        size_t,  // input_f
        size_t,  // input_x
        size_t,  // input_y
        size_t,  // output_f
        format::type  // format
>;

template <typename InputT, typename WeightsT>
class fully_connected_types_test : public ::testing::Test {
private:
    size_t batch_num() { return _input.size(); }
    size_t input_f() { return _input[0].size(); }
    size_t input_y() { return _input[0][0].size(); }
    size_t input_x() { return _input[0][0][0].size(); }
    size_t output_f() { return _weights.size(); }

    data_types input_data_type() {
        return type_to_data_type<InputT>::value;
    }

    data_types weights_data_type() {
        return type_to_data_type<WeightsT>::value;
    }

    bool has_bias() { return _bias.size() > 0; }

public:
    static std::string PrintToStringParamName(testing::TestParamInfo<fully_connected_types_test_params> param_info) {
        // construct a readable name
        return std::to_string(param_info.index) + "_in_" + std::to_string(testing::get<0>(param_info.param))
               + "x" + std::to_string(testing::get<1>(param_info.param))
               + "x" + std::to_string(testing::get<2>(param_info.param))
               + "x" + std::to_string(testing::get<3>(param_info.param))
               + "_of_" + std::to_string(testing::get<4>(param_info.param))
               + "_" + fmt_to_str(testing::get<5>(param_info.param));
    }

    void set_input(VVVVF<InputT> _data) {
        _input = std::move(_data);
    }

    void set_weights(VVVVF<WeightsT> _data) {
        _weights = std::move(_data);
    }

    void set_bias(VF<WeightsT> _data) {
        _bias = std::move(_data);
    }

    void set_input_format(format::type fmt) {
        _fmt = fmt;
    }

    void run_test(VVF<WeightsT> expected) {
        auto& engine = get_test_engine();

        auto input_size = tensor(TensorValue(batch_num()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));
        auto weights_size = tensor(TensorValue(output_f()), TensorValue(input_f()), TensorValue(input_x()), TensorValue(input_y()));

        auto input_prim = engine.allocate_memory({ input_data_type(), _fmt, input_size });
        auto weights_prim = engine.allocate_memory({ weights_data_type(), format::bfyx, weights_size });

        VF<InputT> input_flattened(input_prim->get_layout().get_linear_size());
        for (size_t bi = 0; bi < batch_num(); ++bi)
            for (size_t fi = 0; fi < input_f(); ++fi)
                for (size_t yi = 0; yi < input_y(); ++yi)
                    for (size_t xi = 0; xi < input_x(); ++xi) {
                        auto idx = tensor((int32_t)bi, (int32_t)fi, (int32_t)xi, (int32_t)yi);
                        auto offset = input_size.get_linear_offset(idx, _fmt);
                        input_flattened[offset] = _input[bi][fi][yi][xi];
                    }

        set_values(input_prim, input_flattened);
        set_values(weights_prim, flatten_4d(format::bfyx, _weights));

        auto bias_prim = engine.allocate_memory({ weights_data_type(), format::bfyx, tensor(feature(output_f()))});
        set_values(bias_prim, _bias);

        topology topo;
        topo.add(data("weights", weights_prim));
        topo.add(data("bias", bias_prim));

        topo.add(input_layout("input", input_prim->get_layout()));

        auto input_sizes = input_size.sizes();
        auto last_dim = std::find_if(input_sizes.rbegin(), input_sizes.rend(),
                                     [](tensor::value_type x) { return x != 1l; });
        size_t input_rank = std::distance(input_sizes.begin(), last_dim.base());
        auto fc_prim = fully_connected("output", input_info("input"), "weights", "bias", cldnn::padding(), input_rank);
        fc_prim.output_data_types = {type_to_data_type<WeightsT>::value};
        topo.add(fc_prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        network net(engine, topo, config);
        net.set_input_data("input", input_prim);

        auto output = net.execute();
        auto out_mem = output.at("output").get_memory();
        cldnn::mem_lock<WeightsT> out_ptr(out_mem, get_test_stream());

        for (size_t bi = 0; bi < batch_num(); ++bi) {
            for (size_t fi = 0; fi < output_f(); ++fi) {
                ASSERT_NEAR(out_ptr[bi * output_f() + fi], expected[bi][fi], 1) << "at b = " << bi << ", fi = " << fi << ", output_f() = " << output_f();
            }
        }
    }

private:
    VVVVF<InputT> _input;
    VVVVF<WeightsT> _weights;
    VF<WeightsT> _bias;
    format::type _fmt;
};

template <typename InputT, typename WeightsT>
class fc_random_types_test
    : public fully_connected_types_test<InputT, WeightsT>
    , public ::testing::WithParamInterface< fully_connected_types_test_params> {
public:
    void run_random_test() {
        tests::random_generator rg(GET_SUITE_NAME);
        size_t b, in_f, in_x, in_y, out_f;
        format::type in_fmt;

        std::tie(b, in_f, in_x, in_y, out_f, in_fmt) = GetParam();

        quantization_t quant_data;
        quant_data.output_low  = std::numeric_limits<WeightsT>::lowest();
        quant_data.output_high = std::numeric_limits<WeightsT>::max();

        VVVVF<InputT> input_data = rg.template generate_random_4d<InputT>(b, in_f, in_y, in_x, 0, 127);
        VVVVF<WeightsT> weights_data = rg.template generate_random_4d<WeightsT>(out_f, in_f, in_y, in_x, quant_data.output_low , quant_data.output_high);
        VF<WeightsT> bias_data = rg.template generate_random_1d<WeightsT>(out_f, quant_data.output_low , quant_data.output_high);

        this->set_input(input_data);
        this->set_weights(weights_data);
        this->set_bias(bias_data);
        this->set_input_format(in_fmt);

        //this->run_test(ref_fully_connected<WeightsT, float,  InputT, WeightsT>(input_data, weights_data, bias_data, quant_data));
        this->run_test(fully_connected_types_reference<InputT, WeightsT>(input_data, weights_data, bias_data, quant_data));
    }
};

using fully_connected_types_i8_i8_test = fc_random_types_test<int8_t, int8_t>;
using fully_connected_types_i8_u8_test = fc_random_types_test<int8_t, uint8_t>;
using fully_connected_types_i8_f32_test = fc_random_types_test<int8_t, float>;

using fully_connected_types_u8_i8_test = fc_random_types_test<uint8_t, int8_t>;
using fully_connected_types_u8_u8_test = fc_random_types_test<uint8_t, uint8_t>;
using fully_connected_types_u8_f32_test = fc_random_types_test<uint8_t, float>;

TEST_P(fully_connected_types_i8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_i8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_i8_f32_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_i8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_u8_test, random) {
    run_random_test();
}

TEST_P(fully_connected_types_u8_f32_test, random) {
    run_random_test();
}

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_i8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_i8_f32_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_i8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_i8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_u8_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_u8_test::PrintToStringParamName
);

INSTANTIATE_TEST_SUITE_P(
    basic,
    fully_connected_types_u8_f32_test,
    testing::Combine(
        testing::Values(1, 2),
        testing::Values(3, 64),
        testing::Values(1),
        testing::Values(1),
        testing::Values(3, 32),
        testing::Values(format::bfyx, format::b_fs_yx_fsv4, format::b_fs_yx_fsv16, format::b_fs_yx_fsv32)
    ),
    fully_connected_types_u8_f32_test::PrintToStringParamName
);
