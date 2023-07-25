// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <legacy/graph_tools.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <common_test_utils/common_utils.hpp>
#include <unordered_set>
#include <gmock/gmock.h>
#include "ie_common.h"
#include <ie_core.hpp>
#include "graph_test_base.hpp"
#include <memory>
#include <frontend/model_quantizer.hpp>

using namespace testing;
using namespace InferenceEngine;
using namespace std;
using namespace GraphTest;
using namespace ov::intel_gna::frontend;

class GraphCopyTests : public GraphTestsBase {

protected:
    MockCopier mc;
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model,
                                         std::vector<float> scale_factors) const {
        GnaInputs inputs;
        InferenceEngine::InputsDataMap inputs_map = model.getInputsInfo();
        size_t sf_id = 0;
        for (auto&& input_data : inputs_map) {
            auto input_layer = getCreatorLayer(input_data.second->getInputData()).lock();
            if (scale_factors.size() <= sf_id) {
                THROW_GNA_EXCEPTION << "Scale factors are not set for some of the inputs";
            }
            inputs[input_layer->name].scale_factor = scale_factors[sf_id++];
        }

        Config gna_config;
        gna_config.gnaPrecision = InferenceEngine::Precision::I16;
        gna_config.gnaFlags.input_low_precision = false;

        auto transformer = ov::intel_gna::TransformationsPipeline(gna_config);

        return ModelQuantizer(transformer)
            .quantize(
            model,
            inputs);
    }

    void SetUp() override {
        GraphTestsBase::_batchSize = 12;
        GraphTestsBase::SetUp();
        CONNECT(1, 2);
        CONNECT(3, 4);
        CONNECT(4, 2);
        CONNECT(3, 5);
        CONNECT(5, 2);

        EXPECT_CALL(*mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap &maps) {
            prepareInputs(maps, 12);
        })));

        EXPECT_CALL(*mockNet, getOutputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](OutputsDataMap &maps) {
            prepareOutputs(maps);
        })));

        EXPECT_CALL(*mockNet, getBatchSize()).WillRepeatedly(Return(12));
        EXPECT_CALL(*mockNet, getName()).WillRepeatedly(ReturnRefOfCopy(std::string("nm")));

        EXPECT_CALL(mc, copyLayer(_)).WillRepeatedly(Invoke([](CNNLayerPtr ptr) {
            return ptr;
        }));
    }
};

TEST_F(GraphCopyTests, canPreserveBatchWhenCopyNetwork) {
    auto clone = CNNNetCopy<MockCopier>(CNNNetwork(mockNet), mc);
    auto icnnnet = static_cast<ICNNNetwork::Ptr>(clone);

    //network was copied not just assigned
    ASSERT_NE(icnnnet.get(), mockNet.get());

    ASSERT_EQ(clone.getBatchSize(), 12);
}


TEST_F(GraphCopyTests, canPreserveInputs) {
    auto clone = CNNNetCopy<MockCopier>(CNNNetwork(mockNet), mc);

    InputsDataMap inputs = clone.getInputsInfo(), inputsTarget;
    InputsDataMap heads, headsTarget;

    mockNet->getInputsInfo(inputsTarget);
    ASSERT_INPUTS_INFO_EQ(inputs, inputsTarget);
}

TEST_F(GraphCopyTests, canPreserveOutputs) {
    auto clone = CNNNetCopy<MockCopier>(CNNNetwork(mockNet), mc);

    OutputsDataMap outTarget = clone.getOutputsInfo(), outSource;
    mockNet->getOutputsInfo(outSource);

    ASSERT_OUTPUTS_INFO_EQ(outSource, outTarget);
}

TEST_F(GraphCopyTests, canPreserveAttributes) {
    auto clone = CNNNetCopy<MockCopier>(CNNNetwork(mockNet), mc);
    ADD_ATTR(1, "id", "r-1-2-3");
    ADD_ATTR(2, "id", "r-1-2-3");
    auto idMemOutput = CommonTestUtils::getLayerByName(clone, "1")->GetParamAsString("id");
    auto idMemInput  = CommonTestUtils::getLayerByName(clone, "2")->GetParamAsString("id");

    ASSERT_STREQ(idMemInput.c_str(), idMemOutput.c_str());
    ASSERT_STREQ(idMemInput.c_str(), "r-1-2-3");
}

TEST_F(GraphCopyTests, canQuantizeTopology) {
    auto clone = quantize(CNNNetwork(mockNet), std::vector<float >({1.0f, 1.0f}));

    CNNNetBFS(CommonTestUtils::getLayerByName(clone, "1"), [&](CNNLayerPtr layer) {
        auto params = getInjectedData<QuantizedLayerParams>(layer);
        ASSERT_NE(params, nullptr);
    });

    CNNNetBFS(CommonTestUtils::getLayerByName(clone, "3"), [&](CNNLayerPtr layer) {
        auto params = getInjectedData<QuantizedLayerParams>(layer);
        ASSERT_NE(params, nullptr);
    });
}

TEST(CNNSpecificGraphCopyTests, copyNetworkWithClampLayer) {
    //define minimal network with Clamp layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>3</dim>
                                        <dim>224</dim>
                                        <dim>224</dim>
                                </port>
                        </output>
                </layer>
                <layer id="1" name="ClampLayer" precision="FP16" type="Clamp">
                    <data max="6" min="0"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </input>
                    <output>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </output>
                </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
    )V0G0N";

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(SINGLE_LAYER_MODEL, InferenceEngine::Blob::CPtr()));

    //copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    auto copied_net = CNNNetwork(copied_net_ptr);

    //check that Clamp layer was properly copied
    auto layer = std::dynamic_pointer_cast<ClampLayer>(CommonTestUtils::getLayerByName(copied_net, "ClampLayer"));
    ASSERT_NE(layer, nullptr) << "Could not perform dynamic cast from base pointer to Clamp layer pointer. "
            "Net copy could be incorrect.";
}

TEST(CNNSpecificGraphCopyTests, copyPreprocess) {
    //define minimal network with Clamp layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>3</dim>
                                        <dim>224</dim>
                                        <dim>224</dim>
                                </port>
                        </output>
                </layer>
                <layer id="1" name="ClampLayer" precision="FP16" type="Clamp">
                    <data max="6" min="0"/>
                    <input>
                            <port id="0">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </input>
                    <output>
                            <port id="1">
                                    <dim>1</dim>
                                    <dim>3</dim>
                                    <dim>224</dim>
                                    <dim>224</dim>
                            </port>
                    </output>
                </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
        <pre-process reference-layer-name="InputLayer">
            <channel id="0">
                <mean value="104"/>
            </channel>
            <channel id="1">
                <mean value="116"/>
            </channel>
            <channel id="2">
                <mean value="122"/>
            </channel>
        </pre-process>
    </net>
    )V0G0N";

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;

    ASSERT_NO_THROW(network = core.ReadNetwork(SINGLE_LAYER_MODEL, InferenceEngine::Blob::CPtr()));

    //copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    auto copied_net = CNNNetwork(copied_net_ptr);

    //check that pre process Info existed in copied network
    auto &pp = copied_net.getInputsInfo().begin()->second->getPreProcess();
    ASSERT_EQ(MEAN_VALUE, pp.getMeanVariant());
    ASSERT_EQ(3, pp.getNumberOfChannels());


    ASSERT_FLOAT_EQ(pp[0]->meanValue, 104);
    ASSERT_FLOAT_EQ(pp[1]->meanValue, 116);
    ASSERT_FLOAT_EQ(pp[2]->meanValue, 122);
}

TEST(CNNSpecificGraphCopyTests, copyNetworkWithDeconvolution) {
    //define minimal network with deconvolution layer
    const std::string SINGLE_LAYER_MODEL = R"V0G0N(
    <net name="SingleLayer" version="2" batch="1">
        <layers>
                <layer id="0" name="InputLayer" precision="FP16" type="Input">
                        <output>
                                <port id="0">
                                        <dim>1</dim>
                                        <dim>384</dim>
                                        <dim>4</dim>
                                        <dim>2</dim>
                                </port>
                        </output>
                </layer>
            <layer name="upsample_merged" type="Deconvolution" precision="FP16" id="1">
            <deconvolution_data stride-x="2" stride-y="2" pad-x="1" pad-y="1" kernel-x="4" kernel-y="4" output="384" group="384"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>4</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>384</dim>
                    <dim>8</dim>
                    <dim>4</dim>
                </port>
            </output>
            <weights offset="0" size="12288"/>
        </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
    )V0G0N";

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    auto blob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {12288}, Layout::C));
    blob->allocate();
    ASSERT_NO_THROW(network = core.ReadNetwork(SINGLE_LAYER_MODEL, blob));

    // copy the network
    struct EmptyStruct {};
    auto visitor = [&](CNNLayerPtr lp) { return injectData<EmptyStruct>(lp); };
    auto copied_net_ptr = CNNNetCopy(network, visitor);
    auto copied_net = CNNNetwork(copied_net_ptr);

    // check that Clamp layer was properly copied
    auto layer = std::dynamic_pointer_cast<DeconvolutionLayer>(CommonTestUtils::getLayerByName(copied_net, "upsample_merged"));
    ASSERT_NE(layer, nullptr) << "Could not perform dynamic cast from base pointer to Deconvolution layer pointer. "
                                 "Net copy could be incorrect.";
}
