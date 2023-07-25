# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging as log
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm

from openvino.tools.ovc.moc_frontend.check_config import new_extensions_used
from openvino.tools.ovc.moc_frontend.pipeline import moc_pipeline
from openvino.tools.ovc.moc_frontend.moc_emit_ir import moc_emit_ir
from openvino.tools.ovc.convert_data_type import destination_type_to_np_data_type
from openvino.tools.ovc.cli_parser import get_available_front_ends, \
    get_common_cli_options, get_model_name_from_args, depersonalize, get_mo_convert_params, \
    input_to_input_cut_info, freeze_placeholder_to_input_cut_info

from openvino.tools.ovc.error import Error, FrameworkError
from openvino.tools.ovc.get_ov_update_message import get_ov_update_message, get_compression_message
from openvino.tools.ovc.version import VersionChecker
from openvino.tools.ovc.utils import check_values_equal
from openvino.tools.ovc.logger import init_logger
from openvino.tools.ovc.telemetry_utils import send_params_info, send_conversion_result, \
    init_mo_telemetry
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import get_pytorch_decoder, extract_input_info_from_example
from openvino.tools.ovc.moc_frontend.paddle_frontend_utils import paddle_frontend_converter

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager, OpConversionFailure, TelemetryExtension
from openvino.runtime import get_version as get_rt_version
from openvino.runtime import Type, PartialShape

try:
    from openvino.frontend.tensorflow.utils import create_tf_graph_iterator, type_supported_by_tf_fe, \
        extract_model_graph  # pylint: disable=no-name-in-module,import-error

    tf_frontend_with_python_bindings_installed = True
except (ModuleNotFoundError, ImportError):
    tf_frontend_with_python_bindings_installed = False


def replace_ext(name: str, old: str, new: str):
    base, ext = os.path.splitext(name)
    log.debug("base: {}, ext: {}".format(base, ext))
    if ext == old:
        return base + new


def print_argv(argv: argparse.Namespace, model_name: str):
    print('Model Conversion arguments:')
    props = OrderedDict()
    props['common_args'] = get_common_cli_options(model_name)

    framework_specifics_map = {
        'common_args': 'Common parameters:'
    }

    lines = []
    for key in props:
        lines.append(framework_specifics_map[key])
        for (op, desc) in props[key].items():
            if isinstance(desc, list):
                lines.append('\t{}: \t{}'.format(desc[0], desc[1](getattr(argv, op, 'NONE'))))
            else:
                lines.append('\t{}: \t{}'.format(desc, getattr(argv, op, 'NONE')))
    print('\n'.join(lines), flush=True)


def arguments_post_parsing(argv: argparse.Namespace):
    # TODO: This function looks similar to another one. Check for code duplicates.
    log.debug("Model Conversion API started")
    log.debug('Output model name would be {}{{.xml, .bin}}'.format(argv.output_model))

    if argv.verbose:
        print_argv(argv, argv.output_model)

    params_parsing(argv)
    argv.output = argv.output.split(',') if argv.output else None

    log.debug("Placeholder shapes : {}".format(argv.placeholder_shapes))

    return argv


def get_moc_frontends(argv: argparse.Namespace):
    fem = argv.feManager

    if not fem:
        return None, []

    available_moc_front_ends = get_available_front_ends(fem)
    if argv.framework:
        moc_front_end = fem.load_by_framework(argv.framework) # WA to prevent process hanging. Need to remove when 115994 fixed.
        moc_front_end = fem.load_by_framework(argv.framework)
        return moc_front_end, available_moc_front_ends
    if argv.input_model:
        if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 2:
            moc_front_end = fem.load_by_model([argv.input_model[0], argv.input_model[1]]) # WA to prevent process hanging. Need to remove when 115994 fixed.
            moc_front_end = fem.load_by_model([argv.input_model[0], argv.input_model[1]])  # TODO: Pass all input model parts
        else:
            moc_front_end = fem.load_by_model(argv.input_model) # WA to prevent process hanging. Need to remove when 115994 fixed.
            moc_front_end = fem.load_by_model(argv.input_model)
        if not moc_front_end:
            return None, available_moc_front_ends
        argv.framework = moc_front_end.get_name()
    else:
        return None, []

    # This check as a workaround to skip IR frontend
    if not moc_front_end.get_name() in available_moc_front_ends:
        return None, available_moc_front_ends

    return moc_front_end, available_moc_front_ends


def prepare_ir(argv: argparse.Namespace):
    argv = arguments_post_parsing(argv)
    t = tm.Telemetry()

    if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 1:
        argv.input_model = argv.input_model[0]

    moc_front_end, available_moc_front_ends = get_moc_frontends(argv)
    if moc_front_end:
        # TODO: Should be moved to the same place where paddle and pytorch handle their objects
        if argv.framework == 'tf' and argv.is_python_object and type_supported_by_tf_fe(argv.input_model):
            argv.input_model = create_tf_graph_iterator(argv.input_model,
                                                        argv.placeholder_shapes,
                                                        argv.placeholder_data_types,
                                                        getattr(argv, "example_input", None))
        t.send_event("mo", "conversion_method", moc_front_end.get_name() + "_frontend")
        moc_front_end.add_extension(TelemetryExtension("mo", t.send_event, t.send_error, t.send_stack_trace))
        if new_extensions_used(argv):
            for extension in argv.extensions:
                moc_front_end.add_extension(extension)
        ov_model = moc_pipeline(argv, moc_front_end)
        return ov_model

    if not argv.input_model:
        raise Error('No input model is provided')

    raise Error('Cannot recognize input model.')


def check_model_object(argv):
    model = argv['input_model']
    if 'tensorflow' in sys.modules:
        if tf_frontend_with_python_bindings_installed and extract_model_graph(argv):
            return "tf"
    if 'torch' in sys.modules:
        import torch
        if isinstance(model, (torch.nn.Module, torch.jit.ScriptFunction)):
            return "pytorch"
        try:
            from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

            if isinstance(model, TorchScriptPythonDecoder):
                return "pytorch"
        except Exception as e:
            pass

    import io
    # FIXME: Consuming any io.BytesIO object as an ONNX model is too dengerous and
    # can conflict with others in the future (not future proof).
    # TODO: Refer to https://onnx.ai/onnx/intro/python.html to find examples with
    # real ONNX python objects. ONNX model has onnx.onnx_ml_pb2.ModelProto type.
    if isinstance(model, io.BytesIO):
        return 'onnx'

    if 'paddle' in sys.modules:
        import paddle
        if isinstance(model, paddle.hapi.model.Model) or isinstance(model,
                                                                    paddle.fluid.dygraph.layers.Layer) or isinstance(
                model, paddle.fluid.executor.Executor):
            return "paddle"

    raise Error('Unknown model type: {}'.format(type(model)))


def driver(argv: argparse.Namespace, non_default_params: dict):
    if not hasattr(argv, 'log_level'):
        argv.log_level = 'ERROR'
    init_logger(argv.log_level.upper(), argv.verbose)

    # Log dictionary with non-default cli parameters where complex classes are excluded.
    log.debug(str(non_default_params))

    start_time = datetime.datetime.now()

    ov_model = moc_emit_ir(prepare_ir(argv), argv)

    if argv.verbose:
        elapsed_time = datetime.datetime.now() - start_time
        print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))
        try:
            import resource
            mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            if sys.platform == 'darwin':
                mem_usage = round(mem_usage / 1024)
            print('[ SUCCESS ] Memory consumed: {} MB. '.format(mem_usage))
        except ImportError:
            pass

    return ov_model


def args_dict_to_list(cli_parser, **kwargs):
    # This method is needed to prepare args from convert_model() for args_parse().
    # The method will not be needed when cli_parser checks are moved from cli_parser to a separate pass.
    import inspect
    from openvino.tools.ovc import convert_model
    signature = inspect.signature(convert_model)
    result = []
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in signature.parameters and check_values_equal(signature.parameters[key].default, value):
            continue
        if check_values_equal(cli_parser.get_default(key), value):
            continue
        # skip parser checking for non str objects
        if not isinstance(value, (str, bool)):
            continue
        result.append('--{}'.format(key))
        if not isinstance(value, bool):
            result.append(value)

    return result


def get_non_default_params(argv, cli_parser):
    import numbers
    import inspect
    from openvino.tools.ovc import convert_model

    signature = inspect.signature(convert_model)
    # make dictionary with parameters which have non-default values to be serialized in IR in rt_info
    non_default_params = {}
    for arg, arg_value in vars(argv).items():
        if arg in signature.parameters and check_values_equal(arg_value, signature.parameters[arg].default):
            continue
        if check_values_equal(arg_value, cli_parser.get_default(arg)):
            continue
        value = depersonalize(arg_value, arg)
        # Skip complex classes in params to prevent
        # serializing it to rt_info
        if isinstance(value, (str, bool, numbers.Number)):
            non_default_params[arg] = value
    return non_default_params


def params_to_string(**kwargs):
    all_params = {}
    for key, value in get_mo_convert_params().items():
        all_params.update(value)

    for key, value in kwargs.items():
        if key in all_params:
            param_data = all_params[key]
            if param_data.to_string is not None:
                kwargs[key] = param_data.to_string(value)
    return kwargs


def add_line_breaks(text: str, char_num: int, line_break: str):
    words = text.replace('\n', "\n ").split(" ")
    cnt = 0
    for i, w in enumerate(words):
        cnt += len(w)
        if '\n' in w:
            cnt = len(w) - w.find('\n') - 1
        if cnt > char_num:
            if words[i][-1] not in ['\n', '\t']:
                words[i] = w + '\n'
            cnt = 0
    text = ' '.join(words).replace("\n ", "\n")
    return line_break + text.replace("\n", line_break)


def show_mo_convert_help():
    mo_convert_params = get_mo_convert_params()
    for group_name, group in mo_convert_params.items():
        print(group_name)
        for param_name in group:
            param_data = group[param_name]
            text = param_data.description.replace("    ", '')
            text = add_line_breaks(text, 56, "\n\t\t\t")
            print("  :param {} {}".format(param_name, text))
        print()


def input_model_is_object(input_model):
    if input_model == ():
        return False
    if isinstance(input_model, (str, Path)):
        return False
    if isinstance(input_model, (tuple, list)):
        return all(input_model_is_object(part) for part in input_model)
    return True


def params_parsing(argv: argparse.Namespace):
    """
    Parses params passed to convert_model and wraps resulting values into dictionaries or lists.
    After working of this method following values are set in argv:

    argv.input, argv.inputs_list - list of input names. Both values are used in some parts of MO.
    Could be good to refactor it and use only one of these values.

    argv.placeholder_shapes - dictionary where key is node name, value is PartialShape,
    or list of PartialShape if node names were not set.

    argv.placeholder_data_types - dictionary where key is node name, value is node np.type,
    or list of np.types if node names were not set.

    argv.freeze_placeholder_with_value - dictionary where key is node name, value is np.ndarray

    argv.unnamed_freeze_placeholder_with_value - list with np.ndarray

    :param argv: MO arguments
    """
    # Parse input to list of InputCutInfo
    inputs = input_to_input_cut_info(argv.input)

    # Make list of input names
    input_names_list = []
    for inp in inputs:
        if inp.name is not None:
            input_names_list.append(inp.name)
    if len(input_names_list) > 0:
        assert len(input_names_list) == len(inputs), "\"input\" parameter has unnamed inputs and named inputs. " \
                                                     "Please either set names for all inputs, " \
                                                     "or do not set names for all inputs."
    argv.inputs_list = input_names_list
    argv.input = ','.join(input_names_list)

    # Parse freeze_placeholder_with_value.
    # values for freezing can be set both by named and unnamed approach if
    # 'input' was used without names and 'freeze_placeholder_with_value' was used with names.
    # So named and unnamed values are stored separately.
    argv.freeze_placeholder_with_value, argv.unnamed_freeze_placeholder_with_value = \
        freeze_placeholder_to_input_cut_info(inputs)

    if len(input_names_list) > 0:
        # Named inputs case
        shape_dict = {}
        data_type_dict = {}
        for inp in inputs:
            if inp.shape is not None:
                # Wrap shape to PartialShape for uniformity of stored values
                shape_dict[inp.name] = PartialShape(inp.shape)
            else:
                shape_dict[inp.name] = None
            if inp.type is not None:
                # Convert type to numpy type for uniformity of stored values
                if isinstance(inp.type, str):
                    data_type_dict[inp.name] = destination_type_to_np_data_type(inp.type)
                elif isinstance(inp.type, Type):
                    data_type_dict[inp.name] = inp.type.to_dtype().type
                else:
                    data_type_dict[inp.name] = inp.type
        argv.placeholder_shapes = shape_dict if shape_dict else None
        argv.placeholder_data_types = data_type_dict if data_type_dict else {}
    else:
        # Unnamed inputs case
        shape_list = []
        data_type_list = []
        for inp in inputs:
            if inp.shape is not None:
                # Wrap shape to PartialShape for uniformity of stored values
                shape_list.append(PartialShape(inp.shape))
            if inp.type is not None:
                # Convert type to numpy type for uniformity of stored values
                if isinstance(inp.type, str):
                    data_type_list.append(destination_type_to_np_data_type(inp.type))
                elif isinstance(inp.type, Type):
                    data_type_list.append(inp.type.to_dtype().type)
                else:
                    data_type_list.append(inp.type)
        argv.placeholder_shapes = shape_list if shape_list else None
        argv.placeholder_data_types = data_type_list if data_type_list else {}
    if argv.framework == "pytorch" and getattr(argv, "example_input", None) is not None:
        extract_input_info_from_example(argv, inputs)


def pack_params_to_args_namespace(args: dict, cli_parser: argparse.ArgumentParser):
    if len(args) > 0:
        args_string = params_to_string(**args)
        argv, _ = cli_parser.parse_known_args(args_dict_to_list(cli_parser, **args_string))  # FIXME: input_model_args can be a list

        # get list of all available params for convert_model()
        all_params = {}
        for key, value in get_mo_convert_params().items():
            all_params.update(value)

        # check that there are no unknown params provided
        for key, value in args_string.items():
            if key not in argv and key not in all_params.keys():
                raise Error("Unrecognized argument: {}".format(key))

            # Non string params like input_model or extensions are ignored by parse_args()
            # so we need to set them in argv separately
            if value is not None and not check_values_equal(getattr(argv, key, None), value):
                setattr(argv, key, value)
    else:
        argv = cli_parser.parse_args()
    return argv


def is_verbose(argv: argparse.Namespace):
    return argv is not None and hasattr(argv, 'verbose') and argv.verbose


def _convert(cli_parser: argparse.ArgumentParser, args, python_api_used):
    # FIXME: It doesn't work when -h is passed
    if 'help' in args and args['help']:
        show_mo_convert_help()
        return None, None
    simplified_ie_version = VersionChecker().get_ie_simplified_version()
    telemetry = init_mo_telemetry()
    telemetry.start_session('mo')
    telemetry.send_event('mo', 'version', simplified_ie_version)
    # Initialize logger with 'ERROR' as default level to be able to form nice messages
    # before arg parser deliver log_level requested by user
    init_logger('ERROR', False)
    argv = None
    # Minimize modifications among other places in case if multiple pieces are passed as input_model
    if python_api_used:
        if 'input_model' not in args:
            args['input_model'] = ()
        if isinstance(args['input_model'], (tuple, list)) and len(args['input_model']) == 1:
            args['input_model'] = args['input_model'][0]
    try:
        model_framework = None
        inp_model_is_object = input_model_is_object(args['input_model']) if python_api_used else False

        if inp_model_is_object:
            model_framework = check_model_object(args)
            if model_framework == "pytorch":
                example_inputs = None
                if 'example_input' in args and args['example_input'] is not None:
                    example_inputs = args['example_input']
                elif 'example_inputs' in args:
                    raise AssertionError(
                        "'example_inputs' argument is not recognized, maybe you meant to provide 'example_input'?")

                get_pytorch_decoder(args['input_model'], example_inputs, args)
            if model_framework == "paddle":
                example_inputs = None
                if 'example_input' in args and args['example_input'] is not None:
                    example_inputs = args['example_input']

                #TODO: Check what example_outputs is and remove if not needed
                example_outputs = None
                if 'example_output' in args and args['example_output'] is not None:
                    example_outputs = args['example_output']
                paddle_runtime_converter = paddle_frontend_converter(args['input_model'], example_inputs,
                                                                     example_outputs)
                pdmodel = paddle_runtime_converter.convert_paddle_to_pdmodel()
                args['input_model'] = pdmodel

        argv = pack_params_to_args_namespace(args, cli_parser)
        argv.framework = model_framework
        argv.is_python_object = inp_model_is_object

        argv.feManager = FrontEndManager()

        # send telemetry with params info
        send_params_info(argv, cli_parser)

        non_default_params = get_non_default_params(argv, cli_parser)
        argv.is_python_api_used = python_api_used

        if inp_model_is_object:
            argv.output_model = "model"   # TODO: Consider removing
        if not hasattr(argv, "output_model") or argv.output_model is None:
            argv.output_model = get_model_name_from_args(argv)

        argv.framework = model_framework

        ov_model = driver(argv, {"conversion_parameters": non_default_params})

        if inp_model_is_object and model_framework == "paddle":
            if paddle_runtime_converter:
                paddle_runtime_converter.destroy()

        # add MO meta data to model
        ov_model.set_rt_info(get_rt_version(), "Runtime_version")
        for key, value in non_default_params.items():
            ov_model.set_rt_info(str(value), ["conversion_parameters", str(key)])

        if is_verbose(argv) or not python_api_used:
            if 'compress_to_fp16' in argv and argv.compress_to_fp16:
                print(get_compression_message())

            ov_update_message = get_ov_update_message()
            if ov_update_message is not None:
                print(ov_update_message)

        send_conversion_result('success')
        return ov_model, argv

    except Exception as e:
        if is_verbose(argv) or not python_api_used:
            if isinstance(e, (FileNotFoundError, NotADirectoryError)):
                log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
                log.debug(traceback.format_exc())
            elif isinstance(e, Error):
                log.error(e)
                log.debug(traceback.format_exc())
            elif isinstance(e, FrameworkError):
                log.error(e, extra={'framework_error': True})
                log.debug(traceback.format_exc())
            else:
                log.error("-------------------------------------------------")
                log.error("----------------- INTERNAL ERROR ----------------")
                log.error("Unexpected exception happened.")
                log.error("Please verify parameters and environment.")
                log.error("If you think this is a bug, please create new ticket here: ")
                log.error("https://github.com/openvinotoolkit/openvino/issues.")
                log.error("-------------- DETAILED INFORMATION -------------")
                log.error(str(e))
                log.error(traceback.format_exc())
                log.error("----------------- END OF REPORT -----------------")
                log.error("-------------------------------------------------")

        send_conversion_result('fail')
        if python_api_used:
            raise e
        else:
            return None, argv
