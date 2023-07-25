Convert a PyTorch Model to OpenVINO™ IR
=======================================

This tutorial demonstrates step-by-step instructions on how to do
inference on a PyTorch classification model using OpenVINO Runtime.
Starting from OpenVINO 2023.0 release, OpenVINO supports direct PyTorch
model conversion without an intermediate step to convert them into ONNX
format. In order, if you try to use the lower OpenVINO version or prefer
to use ONNX, please check this
`tutorial <102-pytorch-to-openvino-with-output.html>`__.

In this tutorial, we will use the
`RegNetY_800MF <https://arxiv.org/abs/2003.13678>`__ model from
`torchvision <https://pytorch.org/vision/stable/index.html>`__ to
demonstrate how to convert PyTorch models to OpenVINO Intermediate
Representation.

The RegNet model was proposed in `Designing Network Design
Spaces <https://arxiv.org/abs/2003.13678>`__ by Ilija Radosavovic, Raj
Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár. The authors
design search spaces to perform Neural Architecture Search (NAS). They
first start from a high dimensional search space and iteratively reduce
the search space by empirically applying constraints based on the
best-performing models sampled by the current search space. Instead of
focusing on designing individual network instances, authors design
network design spaces that parametrize populations of networks. The
overall process is analogous to the classic manual design of networks
but elevated to the design space level. The RegNet design space provides
simple and fast networks that work well across a wide range of flop
regimes.

Prerequisites
-------------

Install notebook dependecies

.. code:: ipython3

    !pip install -q "openvino-dev>=2023.0.0"

Download input data and label map

.. code:: ipython3

    import requests
    from pathlib import Path
    from PIL import Image
    
    MODEL_DIR = Path("model")
    DATA_DIR = Path("data")
    
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_NAME = "regnet_y_800mf"
    
    image = Image.open(requests.get("https://farm9.staticflickr.com/8225/8511402100_fea15da1c5_z.jpg", stream=True).raw)
    
    labels_file = DATA_DIR / "imagenet_2012.txt"
    
    if not labels_file.exists():
        resp = requests.get("https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/imagenet_2012.txt")
        with labels_file.open("wb") as f:
            f.write(resp.content)
    
    imagenet_classes = labels_file.open("r").read().splitlines()

Load PyTorch Model
------------------

Generally, PyTorch models represent an instance of the
``torch.nn.Module`` class, initialized by a state dictionary with model
weights. Typical steps for getting a pre-trained model:

1. Create an instance of a model class
2. Load checkpoint state dict, which contains pre-trained model weights
3. Turn the model to evaluation for switching some operations to inference mode

The ``torchvision`` module provides a ready-to-use set of functions for
model class initialization. We will use
``torchvision.models.regnet_y_800mf``. You can directly pass pre-trained
model weights to the model initialization function using the weights
enum ``RegNet_Y_800MF_Weights.DEFAULT``.

.. code:: ipython3

    import torchvision
    
    # get default weights using available weights Enum for model
    weights = torchvision.models.RegNet_Y_800MF_Weights.DEFAULT
    
    # create model topology and load weights
    model = torchvision.models.regnet_y_800mf(weights=weights)
    
    # switch model to inference mode 
    model.eval();

Prepare Input Data
~~~~~~~~~~~~~~~~~~

The code below demonstrates how to preprocess input data using a
model-specific transforms module from ``torchvision``. After
transformation, we should concatenate images into batched tensor, in our
case, we will run the model with batch 1, so we just unsqueeze input on
the first dimension.

.. code:: ipython3

    import torch
    
    # Initialize the Weight Transforms
    preprocess = weights.transforms()
    
    # Apply it to the input image
    img_transformed = preprocess(image)
    
    # Add batch dimension to image tensor
    input_tensor = img_transformed.unsqueeze(0)

Run PyTorch Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model returns a vector of probabilities in raw logits format,
softmax can be applied to get normalized values in the [0, 1] range. For
a demonstration that the output of the original model and OpenVINO
converted is the same, we defined a common postprocessing function which
can be reused later.

.. code:: ipython3

    import numpy as np
    from scipy.special import softmax
    
    # Perform model inference on input tensor
    result = model(input_tensor)
    
    # Postprocessing function for getting results in the same way for both PyTorch model inference and OpenVINO
    def postprocess_result(output_tensor:np.ndarray, top_k:int = 5):
        """
        Posprocess model results. This function applied sofrmax on output tensor and returns specified top_k number of labels with highest probability
        Parameters:
          output_tensor (np.ndarray): model output tensor with probabilities
          top_k (int, *optional*, default 5): number of labels with highest probability for return
        Returns:
          topk_labels: label ids for selected top_k scores
          topk_scores: selected top_k highest scores predicted by model
        """
        softmaxed_scores = softmax(output_tensor, -1)[0]
        topk_labels = np.argsort(softmaxed_scores)[-top_k:][::-1]
        topk_scores = softmaxed_scores[topk_labels]
        return topk_labels, topk_scores
    
    # Postprocess results
    top_labels, top_scores = postprocess_result(result.detach().numpy())
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_11_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark PyTorch Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    # Run model inference
    model(input_tensor)


.. parsed-literal::

    13.5 ms ± 3.76 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Convert PyTorch Model to OpenVINO Intermediate Representation
-------------------------------------------------------------

Starting from the 2023.0 release OpenVINO supports direct PyTorch models
conversion to OpenVINO Intermediate Representation (IR) format. Model
Optimizer Python API should be used for these purposes. More details
regarding PyTorch model conversion can be found in OpenVINO
`documentation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch.html>`__

   **Note**: Please, take into account that direct support PyTorch
   models conversion is an experimental feature. Model coverage will be
   increased in the next releases. For cases, when PyTorch model
   conversion failed, you still can try to export the model to ONNX
   format. Please refer to this
   `tutorial <102-pytorch-to-openvino-with-output.html>`__
   which explains how to convert PyTorch model to ONNX, then to OpenVINO

The ``convert_model`` function accepts the PyTorch model object and
returns the ``openvino.runtime.Model`` instance ready to load on a
device using ``core.compile_model`` or save on disk for next usage using
``openvino.runtime.serialize``. Optionally, we can provide additional
parameters, such as:

* ``compress_to_fp16`` - flag to perform model weights compression into FP16 data format. It may reduce the required space for model storage on disk and give speedup for inference devices, where FP16 calculation is supported. 
* ``example_input`` - input data sample which can be used for model tracing.
* ``input_shape`` - the shape of input tensor for conversion

and any other advanced options supported by Model Optimizer Python API.
More details can be found on this
`page <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Python_API.html>`__

.. code:: ipython3

    from openvino.tools import mo
    from openvino.runtime import Core, serialize
    
    # Create OpenVINO Core object instance
    core = Core()
    
    # Convert model to openvino.runtime.Model object
    ov_model = mo.convert_model(model)
    
    # Save openvino.runtime.Model object on disk
    serialize(ov_model, MODEL_DIR / f"{MODEL_NAME}_dynamic.xml")
    
    ov_model




.. parsed-literal::

    <Model: 'Model30'
    inputs[
    <ConstOutput: names[x, x.1, 1] shape[?,3,?,?] type: f32>
    ]
    outputs[
    <ConstOutput: names[x.21, 401] shape[?,1000] type: f32>
    ]>



Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Load OpenVINO model on device
    compiled_model = core.compile_model(ov_model, device.value)
    compiled_model




.. parsed-literal::

    <CompiledModel:
    inputs[
    <ConstOutput: names[x, x.1, 1] shape[?,3,?,?] type: f32>
    ]
    outputs[
    <ConstOutput: names[x.21, 401] shape[?,1000] type: f32>
    ]>



Run OpenVINO Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Run model inference
    result = compiled_model(input_tensor)[0]
    
    # Posptorcess results
    top_labels, top_scores = postprocess_result(result)
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_20_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark OpenVINO Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    compiled_model(input_tensor)


.. parsed-literal::

    3.03 ms ± 5.57 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Convert PyTorch Model with Static Input Shape
---------------------------------------------

The default conversion path preserves dynamic input shapes, in order if
you want to convert the model with static shapes, you can explicitly
specify it during conversion using the ``input_shape`` parameter or
reshape the model into the desired shape after conversion. For the model
reshaping example please check the following
`tutorial <002-openvino-api-with-output.html>`__.

.. code:: ipython3

    # Convert model to openvino.runtime.Model object
    ov_model = mo.convert_model(model, input_shape=[[1,3,224,224]])
    # Save openvino.runtime.Model object on disk
    serialize(ov_model, MODEL_DIR / f"{MODEL_NAME}_static.xml")
    ov_model




.. parsed-literal::

    <Model: 'Model38'
    inputs[
    <ConstOutput: names[x, x.1, 1] shape[1,3,224,224] type: f32>
    ]
    outputs[
    <ConstOutput: names[355] shape[1,1000] type: f32>
    ]>



Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Load OpenVINO model on device
    compiled_model = core.compile_model(ov_model, device.value)
    compiled_model




.. parsed-literal::

    <CompiledModel:
    inputs[
    <ConstOutput: names[x, x.1, 1] shape[1,3,224,224] type: f32>
    ]
    outputs[
    <ConstOutput: names[355] shape[1,1000] type: f32>
    ]>



Now, we can see that input of our converted model is tensor of shape [1,
3, 224, 224] instead of [?, 3, ?, ?] reported by previously converted
model.

Run OpenVINO Model Inference with Static Input Shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # Run model inference
    result = compiled_model(input_tensor)[0]
    
    # Posptorcess results
    top_labels, top_scores = postprocess_result(result)
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_31_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark OpenVINO Model Inference with Static Input Shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    compiled_model(input_tensor)


.. parsed-literal::

    2.79 ms ± 26.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Convert TorchScript Model to OpenVINO Intermediate Representation
-----------------------------------------------------------------

TorchScript is a way to create serializable and optimizable models from
PyTorch code. Any TorchScript program can be saved from a Python process
and loaded in a process where there is no Python dependency. More
details about TorchScript can be found in `PyTorch
documentation <https://pytorch.org/docs/stable/jit.html>`__.

There are 2 possible ways to convert the PyTorch model to TorchScript:

-  ``torch.jit.script`` - Scripting a function or ``nn.Module`` will
   inspect the source code, compile it as TorchScript code using the
   TorchScript compiler, and return a ``ScriptModule`` or
   ``ScriptFunction``.
-  ``torch.jit.trace`` - Trace a function and return an executable or
   ``ScriptFunction`` that will be optimized using just-in-time
   compilation.

Let’s consider both approaches and their conversion into OpenVINO IR.

Scriped Model
~~~~~~~~~~~~~

``torch.jit.script`` inspects model source code and compiles it to
``ScriptModule``. After compilation model can be used for inference or
saved on disk using the ``torch.jit.save`` function and after that
restored with ``torch.jit.load`` in any other environment without the
original PyTorch model code definitions.

TorchScript itself is a subset of the Python language, so not all
features in Python work, but TorchScript provides enough functionality
to compute on tensors and do control-dependent operations. For a
complete guide, see the `TorchScript Language
Reference <https://pytorch.org/docs/stable/jit_language_reference.html#language-reference>`__.

.. code:: ipython3

    # Get model path
    scripted_model_path = MODEL_DIR / f"{MODEL_NAME}_scripted.pth"
    
    # Compile and save model if it has not been compiled before or load compiled model
    if not scripted_model_path.exists():
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, scripted_model_path)
    else:
        scripted_model = torch.jit.load(scripted_model_path)
    
    # Run scripted model inference
    result = scripted_model(input_tensor)
    
    # Postprocess results
    top_labels, top_scores = postprocess_result(result.detach().numpy())
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_35_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark Scripted Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    scripted_model(input_tensor)


.. parsed-literal::

    12.7 ms ± 13.4 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Convert PyTorch Scripted Model to OpenVINO Intermediate Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conversion step for the scripted model to OpenVINO IR is similar to
the original PyTorch model.

.. code:: ipython3

    # Convert model to openvino.runtime.Model object
    ov_model = mo.convert_model(scripted_model)
    
    # Load OpenVINO model on device
    compiled_model = core.compile_model(ov_model, device.value)
    
    # Run OpenVINO model inference
    result = compiled_model(input_tensor, device.value)[0]
    
    # Postprocess results
    top_labels, top_scores = postprocess_result(result)
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_39_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark OpenVINO Model Inference Converted From Scripted Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    compiled_model(input_tensor)


.. parsed-literal::

    3.1 ms ± 4.02 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


Traced Model
~~~~~~~~~~~~

Using ``torch.jit.trace``, you can turn an existing module or Python
function into a TorchScript ``ScriptFunction`` or ``ScriptModule``. You
must provide example inputs, and model will be executed, recording the
operations performed on all the tensors.

-  The resulting recording of a standalone function produces
   ScriptFunction.

-  The resulting recording of nn.Module.forward or nn.Module produces
   ScriptModule.

In the same way like scripted model, traced model can be used for
inference or saved on disk using ``torch.jit.save`` function and after
that restored with ``torch.jit.load`` in any other environment without
original PyTorch model code definitions.

.. code:: ipython3

    # Get model path
    traced_model_path = MODEL_DIR / f"{MODEL_NAME}_traced.pth"
    
    # Trace and save model if it has not been traced before or load traced model
    if not traced_model_path.exists():
        traced_model = torch.jit.trace(model, example_inputs=input_tensor)
        torch.jit.save(traced_model, traced_model_path)
    else:
        traced_model = torch.jit.load(traced_model_path)
    
    # Run traced model inference
    result = traced_model(input_tensor)
    
    # Postprocess results
    top_labels, top_scores = postprocess_result(result.detach().numpy())
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_43_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark Traced Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    traced_model(input_tensor)


.. parsed-literal::

    12.7 ms ± 39.6 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


Convert PyTorch Traced Model to OpenVINO Intermediate Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conversion step for a traced model to OpenVINO IR is similar to the
original PyTorch model.

.. code:: ipython3

    # Convert model to openvino.runtime.Model object
    ov_model = mo.convert_model(traced_model)
    
    # Load OpenVINO model on device
    compiled_model = core.compile_model(ov_model, device.value)
    
    # Run OpenVINO model inference
    result = compiled_model(input_tensor)[0]
    
    # Postprocess results
    top_labels, top_scores = postprocess_result(result)
    
    # Show results
    display(image)
    for idx, (label, score) in enumerate(zip(top_labels, top_scores)):
        _, predicted_label = imagenet_classes[label].split(" ", 1)
        print(f"{idx + 1}: {predicted_label} - {score * 100 :.2f}%")



.. image:: 102-pytorch-to-openvino-with-output_files/102-pytorch-to-openvino-with-output_47_0.png


.. parsed-literal::

    1: tiger cat - 25.91%
    2: Egyptian cat - 10.26%
    3: computer keyboard, keypad - 9.22%
    4: tabby, tabby cat - 9.09%
    5: hamper - 2.35%


Benchmark OpenVINO Model Inference Converted From Traced Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%timeit
    
    compiled_model(input_tensor)[0]


.. parsed-literal::

    3.08 ms ± 9.07 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

