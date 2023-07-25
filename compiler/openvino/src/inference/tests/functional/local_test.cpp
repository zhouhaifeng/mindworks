// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>

#include "ngraph/ops.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class LocaleTests : public ::testing::Test {
    std::string originalLocale;
    std::string _model = R"V0G0N(
<net name="model" version="10">
	<layers>
		<layer id="0" name="input" type="Parameter" version="opset1">
			<data shape="1,256,200,272" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="input">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>272</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="rois" type="Parameter" version="opset1">
			<data shape="1000,4" element_type="f16"/>
			<output>
				<port id="0" precision="FP16" names="rois">
					<dim>1000</dim>
					<dim>4</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="indices" type="Parameter" version="opset1">
			<data shape="1000" element_type="i32"/>
			<output>
				<port id="0" precision="I32" names="indices">
					<dim>1000</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="output" type="ROIAlign" version="opset3">
			<data mode="avg" pooled_h="7" pooled_w="7" sampling_ratio="2" spatial_scale="0.25"/>
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
					<dim>200</dim>
					<dim>272</dim>
				</port>
				<port id="1">
					<dim>1000</dim>
					<dim>4</dim>
				</port>
				<port id="2">
					<dim>1000</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP16" names="output">
					<dim>1000</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="output/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1000</dim>
					<dim>256</dim>
					<dim>7</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
		<edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
		<edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
	</edges>
</net>
)V0G0N";

    std::string _model_LSTM = R"V0G0N(
 <net name="LSTMCell" version="10">
    <layers>
        <layer id="0" name="in0" type="Parameter" version="opset1">
            <data shape="1,512" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in1" type="Parameter" version="opset1">
            <data shape="1,256" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in2" type="Parameter" version="opset1">
            <data shape="1,256" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="in3" type="Const" version="opset1">
            <data offset="22223012" size="2097152" shape="1024,512" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="in4" type="Const" version="opset1">
            <data offset="24320164" size="1048576" shape="1024,256" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="in5" type="Const" version="opset1">
            <data offset="25368740" size="4096" shape="1024" element_type="f32"/>
            <output>
                <port id="1" precision="FP32">
                    <dim>1024</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="LSTMCell" type="LSTMCell" version="opset1" precision="FP32">
            <data hidden_size="256" element_type="f32" clip="0.00000"/>
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="3" precision="FP32">
                    <dim>1024</dim>
                    <dim>512</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1024</dim>
                    <dim>256</dim>
                </port>
                <port id="5" precision="FP32">
                    <dim>1024</dim>
                </port>
            </input>
        <output>
            <port id="6" precision="FP32">
                <dim>1</dim>
                <dim>256</dim>
            </port>
            <port id="7" precision="FP32">
                <dim>1</dim>
                <dim>256</dim>
            </port>
        </output>
        </layer>
		<layer id="7" name="485/sink_port_0" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
				</port>
			</input>
		</layer>
		<layer id="8" name="485/sink_port_1" type="Result" version="opset1">
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>256</dim>
				</port>
			</input>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
        <edge from-layer="3" from-port="1" to-layer="6" to-port="3"/>
        <edge from-layer="4" from-port="1" to-layer="6" to-port="4"/>
        <edge from-layer="5" from-port="1" to-layer="6" to-port="5"/>
        <edge from-layer="6" from-port="6" to-layer="7" to-port="0"/>
        <edge from-layer="6" from-port="7" to-layer="8" to-port="0"/>
    </edges>
</net>
)V0G0N";

protected:
    void SetUp() override {
        originalLocale = setlocale(LC_ALL, nullptr);
    }
    void TearDown() override {
        setlocale(LC_ALL, originalLocale.c_str());
    }

    void testBody(bool isLSTM = false) const {
        InferenceEngine::Core core;

        std::string model = isLSTM ? _model_LSTM : _model;
        auto blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {26000000}, Layout::C));
        blob->allocate();
        auto net = core.ReadNetwork(model, blob);

        auto funcs = net.getFunction();

        for (const auto& op : funcs->get_ops()) {
            if (!isLSTM) {
                if (op->get_friendly_name() == "output") {
                    const auto roi = std::dynamic_pointer_cast<ngraph::op::v3::ROIAlign>(op);
                    ASSERT_EQ(roi->get_pooled_h(), 7);
                    ASSERT_EQ(roi->get_pooled_w(), 7);
                    ASSERT_EQ(roi->get_sampling_ratio(), 2);
                    ASSERT_EQ(roi->get_spatial_scale(), 0.25f);
                }
            } else {
                if (op->get_friendly_name() == "LSTMCell") {
                    const auto lstm_seq = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
                    ASSERT_EQ(lstm_seq->get_clip(), 0.0f);
                    ASSERT_EQ(lstm_seq->get_hidden_size(), 256);
                }
            }
        }
    }
};

#if defined(ENABLE_OV_IR_FRONTEND)
TEST_F(LocaleTests, WithRULocale) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithUSLocale) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody();
}

TEST_F(LocaleTests, WithRULocaleOnLSTM) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, WithUSLocaleOnLSTM) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody(true);
}

TEST_F(LocaleTests, DISABLED_WithRULocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("ru_RU.UTF-8"));
    testBody();
    std::locale::global(prev);
}

TEST_F(LocaleTests, DISABLED_WithUSLocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("en_US.UTF-8"));
    testBody();
    std::locale::global(prev);
}
#endif  // defined(ENABLE_OV_IR_FRONTEND)
