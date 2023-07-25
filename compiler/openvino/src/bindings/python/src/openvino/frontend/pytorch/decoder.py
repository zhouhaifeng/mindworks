# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape, Tensor
from openvino.runtime import opset11 as ops

import typing
from packaging.version import parse
import torch
import numpy as np

wrapper_template="""
import torch
from typing import *

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, {input_sign}):
        return self.model({example_input})
"""


def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, bool):
        return OVType.boolean
    if isinstance(value, int):
        # Python int is 64 bit, but we will convert it to int32 except cases when it can't fit in 32 bits
        if torch.iinfo(torch.int).min <= value <= torch.iinfo(torch.int).max:
            return OVType.i32
        return OVType.i64
    return OVType.dynamic


def ivalue_to_constant(ivalue):
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, (list, tuple)):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if isinstance(ivalue, torch.Tensor):
        ivalue = ivalue.to(memory_format=torch.contiguous_format)
        if ivalue.dtype == torch.bfloat16:
            # reinterpret bfloat16 data as float16 to allow conversion to numpy
            ivalue = ivalue.view(torch.float16)
            narr = ivalue.numpy(force=True)
            if not narr.flags['C_CONTIGUOUS']:
                narr = np.ascontiguousarray(narr)
            # TODO: this tensor doesn't share memory with initial tensor
            tensor = Tensor(narr, ivalue.shape, OVType.bf16)
            ov_const = op.Constant(tensor, shared_memory=True)
        else:
            narr = ivalue.numpy(force=True)
            if not narr.flags['C_CONTIGUOUS']:
                narr = np.ascontiguousarray(narr)
            ov_const = op.Constant(narr, shared_memory=True)
        return ov_const.outputs()
    return None


def get_value_from_getattr(getattr_node, self_module):
    assert getattr_node.kind() == "prim::GetAttr", "Got node of kind not equal to prim::GetAttr"
    # GetAttr nodes can be nested
    stack = []
    while getattr_node.kind() == "prim::GetAttr":
        stack.append(getattr_node)
        inputs = list(getattr_node.inputs())
        if len(inputs) == 0:
            break
        getattr_node = inputs[0].node()
    module = self_module
    while len(stack) > 0:
        node = stack.pop()
        attr_name = node.s("name")
        assert hasattr(module, attr_name), f"No attribute with name \"{attr_name}\" found in module."
        module = getattr(module, attr_name)
    return module


pt_to_ov_type_map = {
    "float": OVType.f32,
    "int": OVType.i32,
    "bool": OVType.boolean,
    "torch.bfloat16": OVType.bf16,
    "torch.float16": OVType.f16,
    "torch.float32": OVType.f32,
    "torch.float64": OVType.f64,
    "torch.uint8": OVType.u8,
    "torch.int8": OVType.i8,
    "torch.int32": OVType.i32,
    "torch.int64": OVType.i64,
    "torch.bool": OVType.boolean,
    "torch.DoubleTensor": OVType.f64,
    "torch.FloatTensor": OVType.f32,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.BoolTensor": OVType.boolean,
    "torch.quint8": OVType.u8,
    "torch.qint8": OVType.i8,
    "torch.qint32": OVType.i32
}


class TorchScriptPythonDecoder (Decoder):
    def __init__(self, pt_module, graph_element=None, example_input=None, alias_db=None):
        Decoder.__init__(self)
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self._input_signature = None
        if graph_element is None:
            try:
                pt_module = self._get_scripted_model(pt_module, example_input)
            except Exception as e:
                if example_input is not None:
                    msg = "tracing or scripting"
                    help_msg = ""
                else:
                    msg = "scripting"
                    help_msg = "Tracing sometimes provide better results, "
                    "please provide valid 'example_input' argument. "
                raise RuntimeError(
                    f"Couldn't get TorchScript module by {msg}. {help_msg}"
                    "You can also provide TorchScript module that you obtained"
                    " yourself, please refer to PyTorch documentation: "
                    "https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html.")
            self.graph_element = pt_module.inlined_graph
            self.alias_db = self.graph_element.alias_db()
        else:
            self.graph_element = graph_element
            self.alias_db = alias_db
        self.pt_module = pt_module
        self.raw_inputs = list(self.graph_element.inputs())
        self.raw_outputs = list(self.graph_element.outputs())
        if self._input_signature is not None and "self" in self.raw_inputs[0].debugName():
            self._input_signature.insert(0, "self")

        if isinstance(self.graph_element, torch.Graph):
            self._transform_tensor_list_constants_to_listconstruct(self.graph_element)
            self._transform_optional_constants(self.graph_element)

    def _get_scripted_model(self, pt_module, example_inputs=None):
        import torch
        import inspect

        def process_dict_inputs(inputs, input_params, model):
            ordered_inputs = []
            for input_name in input_params:
                if input_name in inputs:
                    ordered_inputs.append(input_name)

            input_signature = list(input_params)
            if ordered_inputs == input_signature[:len(ordered_inputs)]:
                example_inputs = [inputs[input_name] for input_name in ordered_inputs]
                if all([isinstance(inp, torch.Tensor) for inp in example_inputs]):
                    return {"example_inputs": [inputs[name] for name in ordered_inputs]}, ordered_inputs, model
                return {"example_inputs": example_inputs}, ordered_inputs, model

            # PyTorch has some difficulties to trace models with named unordered parameters:
            # torch < 2.0.0 supports only positional arguments for tracing
            # pytorch == 2.0.0 supports input kwargs tracing, 
            # but does not support complex nested objects (e. g. tuple of tuples of tensors)
            # We will use wrapper for making them positional as workaround.

            input_sign_str = []
            input_params_str = []

            for input_name in ordered_inputs:
                if str(input_params[input_name].annotation).startswith("typing.Union"):
                    filter_custom_args = []
                    for arg in input_params[input_name].annotation.__args__:
                        str_arg = str(arg)
                        is_typing = str_arg.startswith("typing.")
                        is_torch = "torch." in str_arg
                        is_builten = str_arg in (str(int), str(float), str(type(None)))
                        if not (is_typing or is_torch or is_builten):
                            continue
                        filter_custom_args.append(arg)
                    input_params[input_name].annotation.__args__ = tuple(filter_custom_args)
                input_sign_str.append(str(input_params[input_name]).replace("NoneType", "None"))
                input_params_str.append(f"{input_name}={input_name}")

            wrapper_class = wrapper_template.format(input_sign=', '.join(input_sign_str), example_input=', '.join(input_params_str))
            result = {}
            try:
                exec(wrapper_class, result)

                wrapped_model = result["ModelWrapper"](model)
                wrapped_model.eval()
            # if wrapping failed, it is better to return original model for avoid user confusion regarding error message
            except Exception:
                wrapped_model = model

            return {"example_inputs": [inputs[name] for name in ordered_inputs]}, ordered_inputs, wrapped_model

        def prepare_example_inputs_and_model(inputs, input_params, model):
            if isinstance(inputs, dict):
                return process_dict_inputs(inputs, input_params, model)
            if isinstance(inputs, torch.Tensor):
                inputs = [inputs]
            input_signature = list(input_params)
            input_signature = input_signature[:len(inputs)]
            return {"example_inputs": inputs}, input_signature, model

        if isinstance(pt_module, torch.nn.Module):
            pt_module.eval()
        input_signature = None
        if isinstance(pt_module, torch.nn.Module) and not isinstance(pt_module, (torch.jit._trace.TopLevelTracedModule, torch.jit._script.RecursiveScriptModule)):
            # input params is dictionary contains input names and their signature values (type hints and default values if any)
            input_params = inspect.signature(pt_module.forward if hasattr(pt_module, "forward") else pt_module.__call__).parameters
            input_signature = list(input_params)
            if example_inputs is None:
                scripted = torch.jit.script(pt_module)
            else:
                input_parameters, input_signature, pt_module = prepare_example_inputs_and_model(example_inputs, input_params, pt_module)
                try:
                    scripted = torch.jit.trace(pt_module, **input_parameters)
                except Exception:
                    try:
                        scripted = torch.jit.script(pt_module)
                    except Exception:
                        scripted = torch.jit.trace(pt_module, **input_parameters, strict=False)
            skip_freeze = False
            for n in scripted.inlined_graph.nodes():
                # TODO: switch off freezing for all traced models
                if "quantize" in n.kind():
                    # do not freeze quantized models
                    skip_freeze = True
                    break
                elif "aten::to" in n.kind():
                    first_input = next(n.inputs())
                    if first_input.node().kind() == "prim::Constant":
                        ivalue = first_input.toIValue()
                        if ivalue is not None and ivalue.dtype in [torch.uint8, torch.int8, torch.bfloat16, torch.float16]:
                            # do not freeze models with compressed constants
                            skip_freeze = True
                            break
            if not skip_freeze:
                f_model = torch.jit.freeze(scripted)
            else:
                f_model = scripted
        else:
            f_model = pt_module

        self._input_signature = input_signature
        return f_model

    def inputs(self) -> list:
        return [x.unique() for x in self.raw_inputs]

    def get_input(self, index: int):
        return self.inputs()[index]

    def get_input_debug_name(self, index: int) -> str:
        return self._raw_input(index).debugName()

    def get_input_signature_name(self, index: int) -> str:
        if self._input_signature is not None and index < len(self._input_signature):
            return self._input_signature[index]
        return self.get_input_debug_name(index)

    def get_input_shape(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_shape_for_value(raw_input)

    def get_input_type(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_type_for_value(raw_input)

    def get_output_debug_name(self, index: int) -> str:
        return self._raw_output(index).debugName()

    def get_output_shape(self, index: int):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index: int):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, pt_type):
        """Returns known/unknown types wrapped as OVAny."""
        # Check for simple scalar types first
        if pt_type is None:
            return OVAny(OVType.dynamic)
        # TODO: Don't use str, use native types
        if str(pt_type) in pt_to_ov_type_map:
            return OVAny(pt_to_ov_type_map[str(pt_type)])
        elif isinstance(pt_type, torch.TensorType):
            # Tensor type, parse element type
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(pt_type.dtype())))
        elif isinstance(pt_type, torch.ListType):
            element_type = pt_type.getElementType()
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        elif isinstance(pt_type, (torch.StringType, torch.DeviceObjType)):
            return OVAny(DecoderType.Str())
        elif isinstance(pt_type, torch.NoneType):
            return OVAny(DecoderType.PyNone())
        else:
            # Not yet recognized
            return OVAny(OVType.dynamic)

    def get_shape_for_value(self, value: torch.Value):
        if value.isCompleteTensor():
            ps = PartialShape(value.type().sizes())
            return ps
        else:
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value: torch.Value):
        full_type = self._get_known_type_for_value(value.type())
        return full_type

    def get_input_transpose_order(self, index: int) -> list:
        raw_input = self._raw_input(index)
        if raw_input.type() is not None and raw_input.type().kind() == "TensorType":
            strides = raw_input.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_output_transpose_order(self, index: int) -> list:
        output = self._raw_output(index)
        if output.type() is not None and output.type().kind() == "TensorType":
            strides = output.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_subgraph_size(self) -> int:
        if isinstance(self.graph_element, torch.Node):
            return len(self.get_subgraphs())
        else:
            return 1

    def visit_subgraph(self, node_visitor) -> None:
        # make sure topological order is satisfied
        for node in self.graph_element.nodes():
            decoder = TorchScriptPythonDecoder(self.pt_module, node, alias_db=self.alias_db)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self) -> list:
        if self.graph_element.kind() == "prim::PythonOp":
            if "Subgraph" in self.graph_element.attributeNames():
                assert isinstance(self.graph_element, torch.Node), "Graph element must be of type torch.Node."
                return [getattr(self.graph_element, self.graph_element.kindOf("Subgraph"))("Subgraph")]
            else:
                # Attribute "Subgraph" is only available if Graph was created using tracing.
                # TODO Find way to extract subgraph for scripted Graph.
                return []
        return list(self.graph_element.blocks())

    def get_subgraph_decoder(self, index: int):
        decoder = TorchScriptPythonDecoder(self.pt_module, self.get_subgraphs()[index], alias_db=self.alias_db)
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self) -> str:
        assert isinstance(self.graph_element, torch.Node), "Function can be called only when self.graph_element is of type torch.Node"
        return self.graph_element.kind()

    def get_schema(self) -> str:
        return self.graph_element.schema()

    def outputs(self) -> list:
        return [x.unique() for x in self.raw_outputs]

    def _raw_output(self, index: int):
        return self.raw_outputs[index]

    def _raw_input(self, index: int):
        return self.raw_inputs[index]

    def num_of_outputs(self):
        return len(self.raw_outputs)

    def output(self, index: int):
        return self.outputs()[index]

    def mark_node(self, node):
        name = self.graph_element.kind()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        if self.graph_element.scopeName():
            node.set_friendly_name(self.graph_element.scopeName().split("/")[-1] + "/" + name)
        else:
            node.set_friendly_name(name)
        return node

    @staticmethod
    def convert_quantized_tensor(qtensor: torch.Tensor):
        # need to represent as Constant(u8) -> Convert(f32) -> Subtract(zero_point) -> Multiply (scale)
        qscheme = qtensor.qscheme()  # torch.per_channel_affine (per_tensor)
        if qscheme == torch.per_channel_affine:
            int8_tensor = qtensor.int_repr()
            scale = qtensor.q_per_channel_scales().numpy().astype(np.float32)  # (weight.q_scale() for per_tensor)
            zero_point = qtensor.q_per_channel_zero_points().numpy().astype(np.float32)  # (weight.q_zero_point() for per_tensor)
            axis = np.int32(qtensor.q_per_channel_axis())

            new_shape = np.ones(len(int8_tensor.shape), dtype=np.int32)
            new_shape[axis] = -1
            zero_point_bc = np.reshape(zero_point, new_shape)
            scale_bc = np.reshape(scale, new_shape)

            int8_const = op.Constant(int8_tensor.numpy())
            convert = ops.convert(int8_const, np.float32)
            sub = ops.subtract(convert, zero_point_bc)
            return ops.multiply(sub, scale_bc).outputs()
        elif qscheme == torch.per_tensor_affine:
            int8_tensor = qtensor.int_repr()
            scale = np.float32(qtensor.q_scale())
            zero_point = np.float32(qtensor.q_zero_point())

            int8_const = op.Constant(int8_tensor.numpy())
            convert = ops.convert(int8_const, np.float32)
            sub = ops.subtract(convert, zero_point)
            return ops.multiply(sub, scale).outputs()
        assert False, "Unsupported qscheme"

    def try_decode_get_attr(self):
        pt_value = get_value_from_getattr(self.graph_element, self.pt_module)
        assert pt_value is not None, "Couldn't retrieve value from prim::GetAttr"
        if isinstance(pt_value, torch.ScriptObject):
            # We assume this is __torch__.torch.classes.quantized.Conv2dPackedParamsBase or __torch__.torch.classes.quantized.LinearPackedParamsBase
            # TODO: but can be anything. Figure a better way to distinguish
            weight, bias = pt_value.unpack()
            res = self.convert_quantized_tensor(weight)
            if isinstance(bias, torch.Tensor):
                res += ivalue_to_constant(bias)
            else:
                res += ops.convert_like(ivalue_to_constant(torch.zeros(1))[0], res[0]).outputs()
            try:
                # these params exist only for conv params
                stride = pt_value.stride()
                padding = pt_value.padding()
                dilation = pt_value.dilation()
                groups = pt_value.groups()
                res += ivalue_to_constant(stride) + ivalue_to_constant(padding) + ivalue_to_constant(dilation) + ivalue_to_constant(groups)
            except:
                pass
            return res
        elif not isinstance(pt_value, (torch.jit.ScriptModule, torch.jit.TracedModule)):
            return ivalue_to_constant(pt_value)
        else:
            return []

    def as_constant(self):
        if not isinstance(self.graph_element, torch.Node):
            return None
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)
        pt_type = pt_value.type()
        if isinstance(pt_type, torch.TensorType):
            return ivalue_to_constant(pt_value.toIValue())
        if isinstance(pt_type, torch.ListType):
            return self._as_constant_list(pt_value)
        return ivalue_to_constant(pt_value.toIValue())

    def as_string(self):
        if self.get_op_type() == "prim::Constant":
            pt_value = self._raw_output(0)
            if str(pt_value.type()) in ["torch.StringType", "str"]:
                return pt_value.toIValue()
            elif str(pt_value.type()) == "Device":
                return pt_value.toIValue().type
        elif self.get_op_type() == "prim::device":
            return self._get_device_string()
        return None

    @staticmethod
    def _as_constant_list(pt_value: torch.Value):
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively
        # rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def _get_device_string(self) -> str:
        assert self.graph_element.kind() == "prim::device", "This function can be called for prim::device node."
        value = self.raw_inputs[0]
        if value.type().isSubtypeOf(torch.TensorType.get()):
            tensor = typing.cast(torch.TensorType, value.type())
            device = tensor.device()
            if device:
                return str(device)
        # Device cannot be statically determined.
        return "cpu"

    def input_is_none(self, index: int) -> bool:
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            if str(r_input.type()) in ["torch.NoneType", "NoneType"]:
                return True
            else:
                in_node = r_input.node()
                if in_node.kind() == "prim::GetAttr":
                    pt_value = get_value_from_getattr(in_node, self.pt_module)
                    return pt_value is None
        return False

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        if self.get_op_type() in ["aten::conv1d", "aten::conv2d", "aten::conv3d"]:
            # AliasDB::may_contain_alias sometimes return True for tensors produced by convnd, we have to workaround that
            return False
        try:
            return self.alias_db.may_contain_alias(self._raw_input(in_index), self._raw_output(out_index))
        except:
            # Sometimes pytorch fails to get result with IndexError exception while these indexes exist in node
            return False

    @staticmethod
    def _transform_tensor_list_constants_to_listconstruct(graph: torch.Graph):
        # Function replaces prim::Constant containing List of Tensors with
        # prim::ListConstruct containing prim::Constant Tensors.
        assert isinstance(graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            allowed_types = [
                output_type.isSubtypeOf(torch.ListType.ofTensors()),
                output_type.isSubtypeOf(torch.ListType(torch.OptionalType.ofTensor())),
            ]
            if not any(allowed_types):
                continue
            const_inputs = []
            for val in node.output().toIValue():
                const_input = graph.insertConstant(val)
                const_input.node().moveBefore(node)
                const_input.node().copyMetadata(node)
                const_inputs.append(const_input)

            replacement = graph.create("prim::ListConstruct", const_inputs)
            replacement.insertBefore(node)
            replacement.output().setType(torch.ListType.ofTensors())
            replacement.copyMetadata(node)
            node.output().replaceAllUsesWith(replacement.output())

    @staticmethod
    def _transform_optional_constants(graph: torch.Graph):
        # Function replaces prim::Constant containing torch.OptionalType with
        # prim::Constant containing torch.NoneType or type of IValue.
        assert isinstance(graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            if not isinstance(output_type, torch.OptionalType):
                continue
            value = node.output().toIValue()
            const_input = graph.insertConstant(value)
            const_input.node().moveBefore(node)
            const_input.node().copyMetadata(node)
            node.output().replaceAllUsesWith(const_input)
