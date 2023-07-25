// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_nd.hpp>

#include "gather_nd_inst.h"

using namespace cldnn;
using namespace ::tests;

inline void DoTestBase(engine& engine,
    const cldnn::memory::ptr input0,
    const cldnn::memory::ptr input1,
    const std::vector<float>& expected_results,
    const int indices_rank,
    const int batch_dims,
    const cldnn::format fmt,
    const tensor ts,
    const bool batch_merged_output,
    bool is_caching_test=false) {
    topology topology;

    int input_rank = 0;
    if (input0->get_layout().format == format::bfyx) {
        input_rank = 4;
    } else if (input0->get_layout().format == format::bfzyx) {
        input_rank = 5;
    } else if (input0->get_layout().format == format::bfwzyx) {
        input_rank = 6;
    } else {
        FAIL();
    }

    auto gather_nd_inst = gather_nd("gather_nd", input_info("InputData"), input_info("InputIndices"), input_rank, indices_rank, batch_dims, batch_merged_output);
    topology.add(input_layout("InputData", input0->get_layout()));
    topology.add(input_layout("InputIndices", input1->get_layout()));
    topology.add(gather_nd_inst);

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("InputData", input0);
    network->set_input_data("InputIndices", input1);
    auto outputs = network->execute();
    auto output = outputs.at("gather_nd").get_memory();

    // Compare output shape
    auto output_format = output->get_layout().format;
    auto output_shape = output->get_layout().get_tensor();

    ASSERT_EQ(fmt, output_format);

    int32_t dim_size = 6;
    if (fmt == format::bfyx) {
        dim_size = 4;
    } else if (fmt == format::bfzyx) {
        dim_size = 5;
    }

    for (int32_t i = 0; i < dim_size; i++)
    {
        ASSERT_EQ(ts.sizes()[i], output_shape.sizes()[i]);
    }

    // Compare output value
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_EQ(expected_results[i], half_to_float(output_ptr[i]));
    }
}

inline void DoTestV5(engine& engine,
    const cldnn::memory::ptr input0,
    const cldnn::memory::ptr input1,
    const std::vector<float>& expected_results,
    const int indices_rank,
    const int batch_dims,
    const cldnn::format fmt,
    const tensor size,
    bool is_caching_test=false) {
    DoTestBase(engine, input0, input1, expected_results, indices_rank, batch_dims, fmt, size, true, is_caching_test);
}

inline void DoTestV8(engine& engine,
    const cldnn::memory::ptr input0,
    const cldnn::memory::ptr input1,
    const std::vector<float>& expected_results,
    const int indices_rank,
    const int batch_dims,
    const cldnn::format fmt,
    const tensor size,
    bool is_caching_test=false) {
    DoTestBase(engine, input0, input1, expected_results, indices_rank, batch_dims, fmt, size, false, is_caching_test);
}

TEST(gather_nd_gpu_fp16, d23322_i231312_ir6_batch2) {
    auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 2;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 2, 2, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 2, 1, 3, 1 } }); // indices
    // expected output dim: v5{6,1,3,1,2}, v8{2,3,1,3,1,2}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),  FLOAT16(13), FLOAT16(14),    FLOAT16(15), FLOAT16(16),  FLOAT16(11), FLOAT16(12),    FLOAT16(13), FLOAT16(14),  FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),  FLOAT16(23), FLOAT16(24),    FLOAT16(25), FLOAT16(26),  FLOAT16(21), FLOAT16(22),    FLOAT16(23), FLOAT16(24),  FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),  FLOAT16(33), FLOAT16(34),    FLOAT16(35), FLOAT16(36),  FLOAT16(31), FLOAT16(32),    FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12),  FLOAT16(13), FLOAT16(14),    FLOAT16(15), FLOAT16(16),  FLOAT16(11), FLOAT16(12),    FLOAT16(13), FLOAT16(14),  FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),  FLOAT16(23), FLOAT16(24),    FLOAT16(25), FLOAT16(26),  FLOAT16(21), FLOAT16(22),    FLOAT16(23), FLOAT16(24),  FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),  FLOAT16(33), FLOAT16(34),    FLOAT16(35), FLOAT16(36),  FLOAT16(31), FLOAT16(32),    FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),
        });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),    FLOAT16(2), FLOAT16(0),    FLOAT16(2), FLOAT16(0),
        FLOAT16(0), FLOAT16(1),    FLOAT16(0), FLOAT16(1),    FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(0),    FLOAT16(1), FLOAT16(0),    FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),    FLOAT16(2), FLOAT16(1),    FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(1), FLOAT16(0),    FLOAT16(2), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(15), FLOAT16(16),   FLOAT16(11), FLOAT16(12),   FLOAT16(11), FLOAT16(12),
        FLOAT16(25), FLOAT16(26),   FLOAT16(23), FLOAT16(24),   FLOAT16(23), FLOAT16(24),
        FLOAT16(33), FLOAT16(34),   FLOAT16(33), FLOAT16(34),   FLOAT16(33), FLOAT16(34),

        FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),   FLOAT16(25), FLOAT16(26),   FLOAT16(25), FLOAT16(26),
        FLOAT16(31), FLOAT16(32),   FLOAT16(35), FLOAT16(36),   FLOAT16(33), FLOAT16(34),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfzyx, {6, 1, 2, 1, 3});
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfwzyx, { 2, 3, 2, 1, 3, 1 });
}

TEST(gather_nd_gpu_fp16, d231322_i231321_ir6_batch5) {
    auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 5;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 2, 2, 3, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 2, 3, 1, 2, 3, 1 } }); // indices
    // expected output dim: v5{36}, v8{2, 3, 2, 3, 1}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(21), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(31), FLOAT16(30),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(29), FLOAT16(30),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(31),

        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(29),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, {36, 1, 1, 1});
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfzyx, {2, 3, 2, 3, 1});
}

TEST(gather_nd_gpu_fp16, d23322_i23321_ir5_batch4) {
    auto& engine = get_test_engine();

    const int indices_rank = 5;
    const int batch_dims = 4;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 2, 2, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 1, 2, 3 } }); // indices
    // expected output dim: v5{36}, v8{2,3,2,3}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(21), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(31), FLOAT16(30),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(29), FLOAT16(30),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(31),

        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(29),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 36, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 3, 2, 3 });
}


TEST(gather_nd_gpu_fp16, d23223_i2321_ir4_batch3) {
    auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 3;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 2, 3, 3, 2, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 2 } }); // indices
    // expected output dim: v5{12,3} v8{2,3,3,2}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(17), FLOAT16(18),FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28),FLOAT16(25),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),FLOAT16(33),  FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(17), FLOAT16(18),FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28),FLOAT16(25),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(35), FLOAT16(36),FLOAT16(33),  FLOAT16(34), FLOAT16(35), FLOAT16(36),
        });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),

        FLOAT16(0), FLOAT16(0),
        FLOAT16(0), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(14), FLOAT16(15), FLOAT16(16),  FLOAT16(16), FLOAT16(17), FLOAT16(18),
        FLOAT16(24), FLOAT16(25), FLOAT16(26),  FLOAT16(27), FLOAT16(28), FLOAT16(25),
        FLOAT16(32), FLOAT16(33), FLOAT16(34),  FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12), FLOAT16(13),  FLOAT16(17), FLOAT16(18), FLOAT16(15),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),  FLOAT16(26), FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30), FLOAT16(31),  FLOAT16(35), FLOAT16(36), FLOAT16(33),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 12, 3, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 3, 3, 2 });
}

TEST(gather_nd_gpu_fp16, d2342_i2312_ir4_batch2) {
    auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 2;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 2, 4 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 2, 1 } }); // indices
    // expected output dim: v5{6,1}, v8(2,3,1)

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30),   FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),
        FLOAT16(29), FLOAT16(30),   FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),
    });

    set_values(input1, {
        FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),

        FLOAT16(0), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),
        FLOAT16(2), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(14),
        FLOAT16(21),
        FLOAT16(34),

        FLOAT16(11),
        FLOAT16(26),
        FLOAT16(33),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 6, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 3, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d234_i2311_ir4_batch2) {
    auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 2;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 1 } }); // indices
    // expected output dim: v5{6,1,1}, v8{2,3,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(4),
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(9), FLOAT16(10), FLOAT16(11), FLOAT16(12),

        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(17), FLOAT16(18), FLOAT16(19), FLOAT16(20),
        FLOAT16(21), FLOAT16(22), FLOAT16(23), FLOAT16(24),

        });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
        FLOAT16(2),

        FLOAT16(0),
        FLOAT16(2),
        FLOAT16(2),
        });

    std::vector<float> expected_results = {
        FLOAT16(2),
        FLOAT16(5),
        FLOAT16(11),

        FLOAT16(13),
        FLOAT16(19),
        FLOAT16(23),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 6, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 3, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d234_i21_ir2_batch1) {
    auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 1;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 3, 1, 4 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    // expected output dim: v5{2,4,1,1}, v8{2,4,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3), FLOAT16(4),
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(9), FLOAT16(10), FLOAT16(11), FLOAT16(12),

        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(17), FLOAT16(18), FLOAT16(19), FLOAT16(20),
        FLOAT16(21), FLOAT16(22), FLOAT16(23), FLOAT16(24),

    });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(5), FLOAT16(6), FLOAT16(7), FLOAT16(8),
        FLOAT16(13), FLOAT16(14), FLOAT16(15), FLOAT16(16),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 4, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 4, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d22_i21_ir2_batch1) {
    auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 1;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    // expected output dim: v5{2,1,1}, v8{2,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4),
    });

    set_values(input1, {
        FLOAT16(1),
        FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(2),
        FLOAT16(3),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 1, 1, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d3223_i321113_ir6_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 6;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 3, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 3, 2, 3, 1, 1, 1 } }); // indices
    // expected output dim: 323111

    set_values(input0, {
        FLOAT16(11), FLOAT16(12), FLOAT16(13),   FLOAT16(14), FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22), FLOAT16(23),   FLOAT16(24), FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32), FLOAT16(33),   FLOAT16(34), FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42), FLOAT16(43),   FLOAT16(44), FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52), FLOAT16(53),   FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),   FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0), FLOAT16(0),

        FLOAT16(0), FLOAT16(1), FLOAT16(0),
        FLOAT16(2), FLOAT16(0), FLOAT16(1),

        FLOAT16(1), FLOAT16(1), FLOAT16(0),
        FLOAT16(0), FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(64), FLOAT16(65), FLOAT16(66),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),

        FLOAT16(21), FLOAT16(22), FLOAT16(23),
        FLOAT16(54), FLOAT16(55), FLOAT16(56),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),
        FLOAT16(11), FLOAT16(12), FLOAT16(13),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfwzyx, { 3, 2, 3, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfwzyx, { 3, 2, 3, 1, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d3221_i32312_ir3_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 2, 1, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices
    // expected output dim: 32213

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),
        FLOAT16(1), FLOAT16(0),

        FLOAT16(0), FLOAT16(1),
        FLOAT16(2), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),

        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),
        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),

        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfzyx, { 3, 2, 2, 1, 3 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfzyx, { 3, 2, 2, 1, 3 });
}

TEST(gather_nd_gpu_fp16, d3231_i32312_ir3_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfzyx, { 3, 2, 2, 1, 3 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 3 } }); // indices
    // expected output dim: {3,2,2,1}

    set_values(input0, {
        FLOAT16(11), FLOAT16(12),     FLOAT16(13), FLOAT16(14),     FLOAT16(15), FLOAT16(16),
        FLOAT16(21), FLOAT16(22),     FLOAT16(23), FLOAT16(24),     FLOAT16(25), FLOAT16(26),

        FLOAT16(31), FLOAT16(32),     FLOAT16(33), FLOAT16(34),     FLOAT16(35), FLOAT16(36),
        FLOAT16(41), FLOAT16(42),     FLOAT16(43), FLOAT16(44),     FLOAT16(45), FLOAT16(46),

        FLOAT16(51), FLOAT16(52),     FLOAT16(53), FLOAT16(54),     FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62),     FLOAT16(63), FLOAT16(64),     FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(0), FLOAT16(2),

        FLOAT16(0), FLOAT16(1), FLOAT16(0),
        FLOAT16(2), FLOAT16(0), FLOAT16(1),

        FLOAT16(1), FLOAT16(1), FLOAT16(2),
        FLOAT16(0), FLOAT16(0), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(63), FLOAT16(64),
        FLOAT16(35), FLOAT16(36),

        FLOAT16(21), FLOAT16(22),
        FLOAT16(53), FLOAT16(54),

        FLOAT16(45), FLOAT16(46),
        FLOAT16(11), FLOAT16(12),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 2, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 2, 1 });
}

TEST(gather_nd_gpu_fp16, d3112_i3221_ir4_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 1, 2, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices
    // expected output dim: {3,2,2,1,1,2}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(7), FLOAT16(8),
        FLOAT16(13), FLOAT16(14),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),

        FLOAT16(2), FLOAT16(1),
        FLOAT16(0), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),

        FLOAT16(13), FLOAT16(14),       FLOAT16(7), FLOAT16(8),
        FLOAT16(1), FLOAT16(2),         FLOAT16(7), FLOAT16(8),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfwzyx, { 3, 2, 2, 1, 1, 2 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfwzyx, { 3, 2, 2, 1, 1, 2 });
}

TEST(gather_nd_gpu_fp16, d3332_i3223_ir4_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 4;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 3, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 3, 2 } }); // indices
    // expected output dim: {3,2,3,2}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3),     FLOAT16(4), FLOAT16(5), FLOAT16(6),
        FLOAT16(7), FLOAT16(8), FLOAT16(9),     FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),

        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),
        FLOAT16(25), FLOAT16(26), FLOAT16(27),     FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),     FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(2), FLOAT16(0),
        FLOAT16(1), FLOAT16(0), FLOAT16(0),        FLOAT16(1), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(0), FLOAT16(1),        FLOAT16(1), FLOAT16(1), FLOAT16(1),
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1), FLOAT16(1),        FLOAT16(0), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(2), FLOAT16(1),        FLOAT16(0), FLOAT16(2), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(61), FLOAT16(62), FLOAT16(63),
        FLOAT16(19), FLOAT16(20), FLOAT16(21),      FLOAT16(25), FLOAT16(26), FLOAT16(27),

        FLOAT16(22), FLOAT16(23), FLOAT16(24),      FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(51), FLOAT16(52), FLOAT16(53),

        FLOAT16(28), FLOAT16(29), FLOAT16(30),      FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(34), FLOAT16(35), FLOAT16(36),      FLOAT16(16), FLOAT16(17), FLOAT16(18),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 3, 2 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 3, 2 });
}

TEST(gather_nd_gpu_fp16, d3323_i322_ir3_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 3;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 3, 3, 2 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 2 } }); // indices
    // expected output dim: {3,2,3,2}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3),     FLOAT16(4), FLOAT16(5), FLOAT16(6),
        FLOAT16(7), FLOAT16(8), FLOAT16(9),     FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),

        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),
        FLOAT16(25), FLOAT16(26), FLOAT16(27),     FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),     FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input1, {
        FLOAT16(2), FLOAT16(0),
        FLOAT16(2), FLOAT16(1),

        FLOAT16(1), FLOAT16(2),
        FLOAT16(1), FLOAT16(0),

        FLOAT16(0), FLOAT16(1),
        FLOAT16(0), FLOAT16(2),
    });

    std::vector<float> expected_results = {
        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),

        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),
        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),

        FLOAT16(7), FLOAT16(8), FLOAT16(9),        FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),     FLOAT16(16), FLOAT16(17), FLOAT16(18),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 3, 2 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 2, 3, 2 });
}

TEST(gather_nd_gpu_fp16, d22_i21_ir2_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 1, 1, 1 } }); // indices
    // expected output dim: {2,2,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4)
    });

    set_values(input1, {
        FLOAT16(1), FLOAT16(0),
    });

    std::vector<float> expected_results = {
        FLOAT16(3), FLOAT16(4),
        FLOAT16(1), FLOAT16(2),
    };

    DoTestV5(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 2, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 2, 2, 1, 1 });
}

TEST(gather_nd_gpu_fp16, d22_i32_ir2_batch0) {
    auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    // expected output dim: {3,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4)
    });

    set_values(input1, {
        FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(1),
        FLOAT16(3),
        FLOAT16(4),
    };

    DoTestV5(engine,input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 1, 1, 1 });
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 1, 1, 1 });
}

TEST(gather_nd_gpu_fp16, export_import) {
    auto& engine = get_test_engine();

    const int indices_rank = 2;
    const int batch_dims = 0;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfyx, { 2, 2, 1, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, { 3, 2, 1, 1 } }); // indices
    // expected output dim: {3,1,1}

    set_values(input0, {
        FLOAT16(1), FLOAT16(2),
        FLOAT16(3), FLOAT16(4)
    });

    set_values(input1, {
        FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(1),
        FLOAT16(3),
        FLOAT16(4),
    };

    DoTestV5(engine,input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 1, 1, 1 }, true);
    DoTestV8(engine, input0, input1, expected_results, indices_rank, batch_dims, format::bfyx, { 3, 1, 1, 1 }, true);
}

TEST(gather_nd_gpu_fp16, dynamic_r4) {
    auto& engine = get_test_engine();

    ov::Shape in1_shape = {3, 3, 2, 3};
    ov::Shape in2_shape = {3, 2, 2, 3};
    const int batch_dims = 0;
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f16, format::bfyx};
    auto in2_layout = layout{ov::PartialShape(in2_shape), data_types::f16, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f16, format::bfyx}); // data
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f16, format::bfyx}); // Indexes

    set_values(input1, {
        FLOAT16(1), FLOAT16(2), FLOAT16(3),     FLOAT16(4), FLOAT16(5), FLOAT16(6),
        FLOAT16(7), FLOAT16(8), FLOAT16(9),     FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(13), FLOAT16(14), FLOAT16(15),  FLOAT16(16), FLOAT16(17), FLOAT16(18),

        FLOAT16(19), FLOAT16(20), FLOAT16(21),     FLOAT16(22), FLOAT16(23), FLOAT16(24),
        FLOAT16(25), FLOAT16(26), FLOAT16(27),     FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(31), FLOAT16(32), FLOAT16(33),     FLOAT16(34), FLOAT16(35), FLOAT16(36),

        FLOAT16(41), FLOAT16(42), FLOAT16(43),     FLOAT16(44), FLOAT16(45), FLOAT16(46),
        FLOAT16(51), FLOAT16(52), FLOAT16(53),     FLOAT16(54), FLOAT16(55), FLOAT16(56),
        FLOAT16(61), FLOAT16(62), FLOAT16(63),     FLOAT16(64), FLOAT16(65), FLOAT16(66),
    });

    set_values(input2, {
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(2), FLOAT16(0),
        FLOAT16(1), FLOAT16(0), FLOAT16(0),        FLOAT16(1), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(0), FLOAT16(1),        FLOAT16(1), FLOAT16(1), FLOAT16(1),
        FLOAT16(2), FLOAT16(0), FLOAT16(0),        FLOAT16(2), FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1), FLOAT16(1),        FLOAT16(0), FLOAT16(1), FLOAT16(1),
        FLOAT16(1), FLOAT16(2), FLOAT16(1),        FLOAT16(0), FLOAT16(2), FLOAT16(1),
    });

    std::vector<float> expected_results = {
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(61), FLOAT16(62), FLOAT16(63),
        FLOAT16(19), FLOAT16(20), FLOAT16(21),      FLOAT16(25), FLOAT16(26), FLOAT16(27),

        FLOAT16(22), FLOAT16(23), FLOAT16(24),      FLOAT16(28), FLOAT16(29), FLOAT16(30),
        FLOAT16(41), FLOAT16(42), FLOAT16(43),      FLOAT16(51), FLOAT16(52), FLOAT16(53),

        FLOAT16(28), FLOAT16(29), FLOAT16(30),      FLOAT16(10), FLOAT16(11), FLOAT16(12),
        FLOAT16(34), FLOAT16(35), FLOAT16(36),      FLOAT16(16), FLOAT16(17), FLOAT16(18),
    };

    auto expected_fmt = format::bfyx;
    const tensor expected_ts = { 3, 2, 3, 2 };
    uint8_t dim_size = 4;

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(gather_nd("gather_nd", input_info("input1"), input_info("input2"), static_cast<uint8_t>(in1_shape.size()), static_cast<uint8_t>(in2_shape.size()), batch_dims, true));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("gather_nd");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gather_nd").get_memory();
    // Compare output shape
    auto output_format = output->get_layout().format;
    auto output_shape = output->get_layout().get_tensor();

    ASSERT_EQ(expected_fmt, output_format);

    for (int32_t i = 0; i < dim_size; i++)
    {
        ASSERT_EQ(expected_ts.sizes()[i], output_shape.sizes()[i]);
    }
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], half_to_float(output_ptr[i])) << i;
    }
}

TEST(gather_nd_gpu_fp16, dynamic_r5) {
    auto& engine = get_test_engine();

    ov::Shape in1_shape = {2, 3, 3, 2, 2};
    ov::Shape in2_shape = {2, 3, 3, 2, 1};
    const int batch_dims = 4;
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f16, format::bfzyx};
    auto in2_layout = layout{ov::PartialShape(in2_shape), data_types::f16, format::bfzyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f16, format::bfzyx}); // data
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f16, format::bfzyx}); // Indexes

    set_values(input1, {
        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(21), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(31), FLOAT16(30),

        FLOAT16(11), FLOAT16(12),   FLOAT16(13), FLOAT16(14),   FLOAT16(15), FLOAT16(16),   FLOAT16(17), FLOAT16(18),   FLOAT16(19), FLOAT16(10),   FLOAT16(17), FLOAT16(18),
        FLOAT16(21), FLOAT16(22),   FLOAT16(23), FLOAT16(24),   FLOAT16(25), FLOAT16(26),   FLOAT16(27), FLOAT16(28),   FLOAT16(29), FLOAT16(20),   FLOAT16(27), FLOAT16(28),
        FLOAT16(31), FLOAT16(32),   FLOAT16(33), FLOAT16(34),   FLOAT16(35), FLOAT16(36),   FLOAT16(37), FLOAT16(38),   FLOAT16(39), FLOAT16(30),   FLOAT16(29), FLOAT16(30),
        });

    set_values(input2, {
        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),

        FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),    FLOAT16(1), FLOAT16(1),
        FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),    FLOAT16(0), FLOAT16(0),
        FLOAT16(1), FLOAT16(0),    FLOAT16(0), FLOAT16(1),    FLOAT16(1), FLOAT16(0),
        });

    std::vector<float> expected_results = {
        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(31),

        FLOAT16(12), FLOAT16(14),   FLOAT16(16), FLOAT16(18),   FLOAT16(10), FLOAT16(18),
        FLOAT16(21), FLOAT16(23),   FLOAT16(25), FLOAT16(27),   FLOAT16(29), FLOAT16(27),
        FLOAT16(32), FLOAT16(33),   FLOAT16(35), FLOAT16(38),   FLOAT16(30), FLOAT16(29),
    };

    auto expected_fmt = format::bfyx;
    const tensor expected_ts = { 2, 3, 2, 3 };
    uint8_t dim_size = 4;

    topology topology;
    topology.add(input_layout("input1", in1_layout));
    topology.add(input_layout("input2", in2_layout));
    topology.add(gather_nd("gather_nd", input_info("input1"), input_info("input2"), static_cast<uint8_t>(in1_shape.size()), static_cast<uint8_t>(in2_shape.size()), batch_dims, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("gather_nd");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("gather_nd").get_memory();

    // Compare output shape
    auto output_format = output->get_layout().format;
    auto output_shape = output->get_layout().get_tensor();

    ASSERT_EQ(expected_fmt, output_format);

    for (int32_t i = 0; i < dim_size; i++)
    {
        ASSERT_EQ(expected_ts.sizes()[i], output_shape.sizes()[i]);
    }

    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    ASSERT_EQ(expected_results.size(), output_ptr.size());
    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], half_to_float(output_ptr[i])) << i;
    }
}