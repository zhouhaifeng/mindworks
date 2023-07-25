# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import torch

calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
validation_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [validation]
import numpy as np
import torch
import openvino
from sklearn.metrics import accuracy_score

def validate(model: openvino.runtime.CompiledModel, 
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)  
    return accuracy_score(predictions, references)
#! [validation]

#! [quantization]
model = ... # openvino.runtime.Model object

quantized_model = nncf.quantize_with_accuracy_control(model,
                        calibration_dataset=calibration_dataset,
                        validation_dataset=validation_dataset,
                        validation_fn=validate,
                        max_drop=0.01,
                        drop_type=nncf.DropType.ABSOLUTE)
#! [quantization]

#! [inference]
import openvino.runtime as ov

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(quantized_model)

input_fp32 = ... # FP32 model input
res = model_int8(input_fp32)

# save the model
ov.serialize(quantized_model, "quantized_model.xml")
#! [inference]
