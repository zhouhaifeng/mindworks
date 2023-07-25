// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/compiled_model.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/properties.hpp"

#define OV_COMPILED_MODEL_CALL_STATEMENT(...)                                \
    OPENVINO_ASSERT(_impl != nullptr, "CompiledModel was not initialized."); \
    try {                                                                    \
        __VA_ARGS__;                                                         \
    } catch (const std::exception& ex) {                                     \
        OPENVINO_THROW(ex.what());                                           \
    } catch (...) {                                                          \
        OPENVINO_THROW("Unexpected exception");                              \
    }

namespace ov {

CompiledModel::~CompiledModel() {
    _impl = {};
}

CompiledModel::CompiledModel(const std::shared_ptr<ov::ICompiledModel>& impl, const std::shared_ptr<void>& so)
    : _impl{impl},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "CompiledModel was not initialized.");
}

std::shared_ptr<const Model> CompiledModel::get_runtime_model() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->get_runtime_model());
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::inputs() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->inputs());
}

const ov::Output<const ov::Node>& CompiledModel::input() const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        const auto& inputs = _impl->inputs();
        OPENVINO_ASSERT(inputs.size() == 1,
                        "CompiledModel::input() must be called on a compiled model with exactly one parameter.");
        return inputs.at(0);
    });
}

const ov::Output<const ov::Node>& CompiledModel::input(size_t i) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        OPENVINO_ASSERT(i < _impl->inputs().size(),
                        "Cannot get input for index: ",
                        i,
                        " outputs size is ",
                        _impl->inputs().size());
        return _impl->inputs().at(i);
    });
}

const ov::Output<const ov::Node>& CompiledModel::input(const std::string& tensor_name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        for (const auto& input : _impl->inputs()) {
            if (input.get_names().count(tensor_name)) {
                return input;
            }
        }
        OPENVINO_THROW("Input for tensor name '", tensor_name, "' is not found.");
    });
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::outputs() const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return _impl->outputs());
}

const ov::Output<const ov::Node>& CompiledModel::output() const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        const auto& outputs = _impl->outputs();
        OPENVINO_ASSERT(outputs.size() == 1,
                        "CompiledModel::output() must be called on a compiled model with exactly one result.");
        return outputs.at(0);
    });
}
const ov::Output<const ov::Node>& CompiledModel::output(size_t i) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        OPENVINO_ASSERT(i < _impl->outputs().size(),
                        "Cannot get output for index: ",
                        i,
                        " outputs size is ",
                        _impl->outputs().size());
        return _impl->outputs().at(i);
    });
}
const ov::Output<const ov::Node>& CompiledModel::output(const std::string& tensor_name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        for (const auto& output : _impl->outputs()) {
            if (output.get_names().count(tensor_name)) {
                return output;
            }
        }
        OPENVINO_THROW("Output for tensor name '", tensor_name, "' is not found.");
    });
}

InferRequest CompiledModel::create_infer_request() {
    OV_COMPILED_MODEL_CALL_STATEMENT(return {_impl->create_infer_request(), _so});
}

void CompiledModel::export_model(std::ostream& networkModel) {
    OV_COMPILED_MODEL_CALL_STATEMENT(_impl->export_model(networkModel));
}

void CompiledModel::set_property(const AnyMap& config) {
    OV_COMPILED_MODEL_CALL_STATEMENT(_impl->set_property(config));
}

Any CompiledModel::get_property(const std::string& name) const {
    OV_COMPILED_MODEL_CALL_STATEMENT(return {_impl->get_property(name), {_so}});
}

RemoteContext CompiledModel::get_context() const {
    OV_COMPILED_MODEL_CALL_STATEMENT({
        auto ctx = _impl->get_context();
        if (!ctx._so)
            ctx._so = _so;
        return {ctx._ptr, ctx._so};
    });
}

bool CompiledModel::operator!() const noexcept {
    return !_impl;
}

CompiledModel::operator bool() const noexcept {
    return !!_impl;
}

}  // namespace ov
