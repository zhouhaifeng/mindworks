# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

# test.BACKEND_NAME is a configuration variable determining which
# nGraph backend tests will use. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more details.
BACKEND_NAME = None

# test.MODEL_ZOO_DIR is a configuration variable providing the path
# to the ZOO of ONNX models to test. It's set during pytest configuration time.
# See `pytest_configure` hook in `conftest.py` for more
# details.
MODEL_ZOO_DIR = None

# test.MODEL_ZOO_XFAIL is a configuration variable which enable xfails for model zoo.
MODEL_ZOO_XFAIL = False


def xfail_test(reason="Mark the test as expected to fail", strict=True):
    return pytest.mark.xfail(reason=reason, strict=strict)


skip_segfault = pytest.mark.skip(reason="Segmentation fault error")
xfail_accuracy = xfail_test(reason="Accuracy")
xfail_unsupported_by_legacy_api = xfail_test(reason="RuntimeError: This feature is not supported via legacy API.")
xfail_issue_69444 = xfail_test(reason="ONNX Resize - AssertionError: Mismatched elements.")
xfail_issue_67415 = xfail_test(reason="RuntimeError: Unsupported data type for when filling blob!")
xfail_issue_33488 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "MaxUnpool")
skip_issue_38084 = pytest.mark.skip(reason="Aborted (core dumped) Assertion "
                                           "`(layer->get_output_partial_shape(i).is_static())' failed.")
xfail_issue_33595 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "Unique")
xfail_issue_33596 = xfail_test(reason="RuntimeError: nGraph does not support different sequence operations: "
                                      "ConcatFromSequence, SequenceConstruct, SequenceAt, SplitToSequence, "
                                      "SequenceEmpty, SequenceInsert, SequenceErase, SequenceLength ")
xfail_issue_33606 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "Det")
xfail_issue_33651 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "TfIdfVectorizer")
xfail_issue_33581 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "GatherElements")
xfail_issue_35923 = xfail_test(reason="RuntimeError: PReLU without weights is not supported")
xfail_issue_35927 = xfail_test(reason="RuntimeError: B has zero dimension that is not allowable")
xfail_issue_38084 = xfail_test(reason="RuntimeError: AssertionFailed: layer->get_output_partial_shape(i)."
                                      "is_static() nGraph <value> operation with name: <value> cannot be "
                                      "converted to <value> layer with name: <value> because output "
                                      "with index 0 contains dynamic shapes: {<value>}. Try to use "
                                      "CNNNetwork::reshape() method in order to specialize shapes "
                                      "before the conversion.")
xfail_issue_38091 = xfail_test(reason="AssertionError: Mismatched elements")
xfail_issue_38699 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Gradient")
xfail_issue_38701 = xfail_test(reason="RuntimeError: unsupported element type: STRING")
xfail_issue_38706 = xfail_test(reason="RuntimeError: output_3.0 has zero dimension which is not allowed")
xfail_issue_38708 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Slice): y>': "
                                      "Axes input must be constant")
xfail_issue_38710 = xfail_test(reason="RuntimeError: data has zero dimension which is not allowed")
xfail_issue_38713 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Momentum")
xfail_issue_38724 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Resize): Y>': "
                                      "tf_crop_and_resize - this type of coordinate transformation mode "
                                      "is not supported. Choose one of the following modes: "
                                      "tf_half_pixel_for_nn, asymmetric, align_corners, pytorch_half_pixel, "
                                      "half_pixel")
xfail_issue_38725 = xfail_test(reason="RuntimeError: While validating ONNX node '<Node(Loop): "
                                      "value info has no element type specified")
xfail_issue_38726 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "LessOrEqual")
xfail_issue_38732 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "ConvInteger")
xfail_issue_38734 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Adam")
xfail_issue_38735 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "ai.onnx.preview.training.Adagrad")
xfail_issue_48052 = xfail_test(reason="Dropout op is not supported in traning mode")
xfail_issue_45180 = xfail_test(reason="RuntimeError: Unsupported dynamic op: ReduceSum")
xfail_issue_44851 = xfail_test(reason="Expected: Unsupported dynamic op: Broadcast")
xfail_issue_44858 = xfail_test(reason="Expected: Unsupported dynamic op: Unsqueeze")
xfail_issue_44957 = xfail_test(reason="Expected: Unsupported dynamic op: NonZero")
xfail_issue_44958 = xfail_test(reason="Expected: Unsupported dynamic op: Interpolate")
xfail_issue_44965 = xfail_test(reason="Expected: RuntimeError: value info has no element")
xfail_issue_47323 = xfail_test(reason="RuntimeError: The plugin does not support FP64")
xfail_issue_73538 = xfail_test(reason="OneHot: Unsupported negative indices, "
                                      "AssertionError: Mismatched elements.")
xfail_issue_99949 = xfail_test(reason="Bitwise operators are not supported")
xfail_issue_99950 = xfail_test(reason="CenterCropPad func is not supported")
xfail_issue_99952 = xfail_test(reason="Col2Im operator is not supported")
xfail_issue_99954 = xfail_test(reason="Constant Pad - RuntimeError: Shape inference of Reference node with name y failed")
xfail_issue_99955 = xfail_test(reason="GroupNorm is not supported")
xfail_issue_99957 = xfail_test(reason="LayerNorm - RuntimeError: While validating node '<Node(Reshape): Mean>'")
xfail_issue_99958 = xfail_test(reason="LogSoftmax - Results mismatch")
xfail_issue_99959 = xfail_test(reason="Mish function is not supported")
xfail_issue_99960 = xfail_test(reason="MVN - Results mismatch")
xfail_issue_99961 = xfail_test(reason="Optional has/get element operators are not supported)'")
xfail_issue_99962 = pytest.mark.skip(reason="ReduceL1/L2 - Unrecognized attribute: axes for operator ReduceL1/L2")
xfail_issue_99968 = xfail_test(reason="ReduceL1/L2 - Results mismatch or unsupported ReduceSum with "
                                      "dynamic rank by CPU plugin")
xfail_issue_99969 = xfail_test(reason="Resize - Results mismatch / "
                                      "RuntimeError: While validating ONNX node '<Node(Resize): Y>' / "
                                      "RuntimeError: Check '(false)' failed at onnx/frontend/src/op/resize.cpp")
xfail_issue_99970 = xfail_test(reason="Scatter and ScatterND - RuntimeError: Check '(reduction == none)' failed at "
                                      "src/frontends/onnx/frontend/src/op/scatter_elements.cpp OR at "
                                      "src/frontends/onnx/frontend/src/op/scatter_nd")
xfail_issue_99972 = xfail_test(reason="Softmax - Results mismatch")
xfail_issue_99973 = xfail_test(reason="Split -  RuntimeError: While validating ONNX node "
                                      "'<Node(Split): output_1, output_2, output_3, output_4>'")

# Model MSFT issues:
xfail_issue_37957 = xfail_test(reason="RuntimeError: nGraph does not support the following ONNX operations: "
                                      "com.microsoft.CropAndResize, com.microsoft.GatherND, "
                                      "com.microsoft.Pad, com.microsoft.Range")
xfail_issue_39669 = xfail_test(reason="AssertionError: This model has no test data")
xfail_issue_36534 = xfail_test(reason="RuntimeError: node input index is out of range")
xfail_issue_36536 = xfail_test(reason="RuntimeError: can't protect")
xfail_issue_36538 = xfail_test(reason="RuntimeError: Check 'PartialShape::broadcast_merge_into( pshape, "
                                      "node->get_input_partial_shape(i), autob)' failed at "
                                      "/openvino/ngraph/src/ngraph/op/util/elementwise_args.cpp:48:")
xfail_issue_39658 = xfail_test(reason="RuntimeError: Tile operation has a form that is not supported."
                                      " z should be converted to TileIE operation.")
xfail_issue_39662 = xfail_test(reason="RuntimeError: 'ScatterElementsUpdate' layer with name 'y' have "
                                      "indices value that points to non-existing output tensor element")


xfail_issue_37973 = xfail_test(reason="TF Inception V2 - AssertionError: zoo models results mismatch")
xfail_issue_47430 = xfail_test(reason="FCN ResNet models - AssertionError: zoo models results mismatch")
xfail_issue_47495 = xfail_test(reason="BertSquad-10 from MSFT - AssertionError: zoo models results mismatch")
xfail_issue_48145 = xfail_test(reason="BertSquad-8 - AssertionError: Items are not equal: ACTUAL: 4 "
                                      "DESIRED: 3")
xfail_issue_48190 = xfail_test(reason="RobertaBase-11 - AssertionError: Items are not equal: "
                                      "ACTUAL: dtype('float64') DESIRED: dtype('float32')")
xfail_issue_49752 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::Pad")
xfail_issue_49753 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::StridedSlice")
xfail_issue_49754 = xfail_test(reason="RuntimeError: Unsupported dynamic ops: v1::TopKIE")
xfail_issue_52463 = xfail_test(reason="test_operator_add_size1_singleton_broadcast_cpu - "
                                      "Not equal to tolerance")
xfail_issue_58033 = xfail_test(reason="Einsum operation misses support for complex ellipsis equations")
xfail_issue_58676 = xfail_test(reason="AssertionError: Not equal to tolerance rtol=0.001, atol=1e-07")
xfail_issue_onnx_models_140 = xfail_test(reason="https://github.com/onnx/models/issues/140")

xfail_issue_63033 = xfail_test(reason="BatchNormalization: Training mode is not supported")
xfail_issue_63036 = xfail_test(reason="Changes in ConvTranspose padding")
xfail_issue_63043 = xfail_test(reason="Recurrent node expects constants as W, R, B inputs.")

skip_rng_tests = pytest.mark.skip(reason="Tests use random number generator with no seed.")
xfail_issue_63137 = xfail_test(reason="Unsupported operations: OptionalHasElement, OptionalGetElement")
xfail_issue_63138 = xfail_test(reason="Missing ONNX Shape-15 support")

xfail_issue_78843 = xfail_test(reason="Missing reference output files for ssd mobilenet models")
xfail_issue_78741 = xfail_test(reason="Cannot get dims for non-static shapes. "
                                      "Requires dynamism support enabled.")

xfail_issue_81976 = xfail_test(reason="RuntimeError: z node not found in graph cache")
xfail_issue_82038 = xfail_test(reason="ScatterElements, ScatterND, AssertionError: Result mismatch")
xfail_issue_82039 = xfail_test(reason="Unsupported data type Optional, RuntimeError: [ NOT_IMPLEMENTED ] "
                                      "CPU plugin: Input image format UNSPECIFIED is not supported yet...")
xfail_issue_90649 = xfail_test(reason="RuntimeError: OV does not support the following ONNX operations:"
                                      "BlackmanWindow, DFT, HammingWindow, HannWindow, LayerNormalization, "
                                      "MelWeightMatrix, SequenceMap, STFT")
xfail_issue_91151 = xfail_test(reason="RuntimeError: model input (shape={3,4}) and blob (shape=(1)) are incompatible")
xfail_issue_91490 = xfail_test(reason="y has zero dimension which is not allowed")
xfail_issue_101965 = xfail_test(reason="Mismatch with numpy-based expected results.")
xfail_issue_113506 = xfail_test(reason="Unsupported operation of type: LSTMSequence Node expects 7 inputs. Actual: 8")

skip_dynamic_model = pytest.mark.skip(reason="CPU plug-in can't load a model with dynamic output shapes via legacy API")
