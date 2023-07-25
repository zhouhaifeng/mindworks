Optimize Preprocessing
======================

When input data does not fit the model input tensor perfectly,
additional operations/steps are needed to transform the data to the
format expected by the model. This tutorial demonstrates how it could be
performed with Preprocessing API. Preprocessing API is an easy-to-use
instrument, that enables integration of preprocessing steps into an
execution graph and performing it on a selected device, which can
improve device utilization. For more information about Preprocessing
API, see this
`overview <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Overview.html#>`__
and
`details <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Preprocessing_Details.html>`__

This tutorial include following steps:

- Downloading the model.
- Setup preprocessing with ModelOptimizer, loading the model and inference with original image.
- Setup preprocessing with Preprocessing API, loading the model and inference with original image.
- Fitting image to the model input type and inference with prepared image.
- Comparing results on one picture.
- Comparing performance.

Settings
--------

Imports
-------

.. code:: ipython3

    import cv2
    import time
    
    import numpy as np
    import tensorflow as tf
    from pathlib import Path
    from openvino.tools import mo
    import matplotlib.pyplot as plt
    from openvino.runtime import Core, serialize


.. parsed-literal::

    2023-07-11 22:53:33.947684: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-11 22:53:33.981920: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-11 22:53:34.528705: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Setup image and device
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    image_path = "../data/image/coco.jpg"

.. code:: ipython3

    import ipywidgets as widgets
    
    core = Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Downloading the model
~~~~~~~~~~~~~~~~~~~~~

This tutorial uses the
`InceptionResNetV2 <https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__.
The InceptionResNetV2 model is the second of the
`Inception <https://github.com/tensorflow/tpu/tree/master/models/experimental/inception>`__
family of models designed to perform image classification. Like other
Inception models, InceptionResNetV2 has been pre-trained on the
`ImageNet <https://image-net.org/>`__ data set. For more details about
this family of models, see the `research
paper <https://arxiv.org/abs/1602.07261>`__.

Load the model by using `tf.keras.applications
api <https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2>`__
and save it to the disk.

.. code:: ipython3

    model_name = "InceptionResNetV2"
    
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / model_name
    
    model = tf.keras.applications.InceptionV3()
    model.save(model_path)


.. parsed-literal::

    2023-07-11 22:53:35.902963: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...


.. parsed-literal::

    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.


.. parsed-literal::

    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 94). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/InceptionResNetV2/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/InceptionResNetV2/assets


Create core
~~~~~~~~~~~

.. code:: ipython3

    core = Core()

Check the original parameters of image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));
    print(f"The original shape of the image is {image.shape}")
    print(f"The original data type of the image is {image.dtype}")


.. parsed-literal::

    The original shape of the image is (577, 800, 3)
    The original data type of the image is uint8



.. image:: 118-optimize-preprocessing-with-output_files/118-optimize-preprocessing-with-output_12_1.png


Convert model to OpenVINO IR and setup preprocessing steps with Model Optimizer
-------------------------------------------------------------------------------

Use Model Optimizer to convert a TensorFlow model to OpenVINO IR.
``mo.convert_model`` python function will be used for converting model
using `OpenVINO Model
Optimizer <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Python_API.html>`__.
The function returns instance of OpenVINO Model class, which is ready to
use in Python interface but can also be serialized to OpenVINO IR format
for future execution using ``openvino.runtime.serialize``. The models
will be saved to the ``./model/ir_model/`` directory.

In this step, some conversions can be setup, which will enable reduction
of work on processing the input data before propagating it through the
network. These conversions will be inserted as additional input
pre-processing sub-graphs into the converted model.

Setup the following conversions:

- mean normalization with ``mean_values`` parameter
- scale with ``scale_values``
- color conversion, the color format of example image will be ``BGR``, but the model required ``RGB`` format, so add ``reverse_input_channels=True`` to process the image into the desired format

Also converting of layout could be specified with ``layout`` option.
More information and parameters described in the `Embedding
Preprocessing Computation
article <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Additional_Optimization_Use_Cases.html#embedding-preprocessing-computation>`__.

.. code:: ipython3

    ir_path_mo_preprocess = model_dir / "ir_model" / f"{model_name}_mo_preproc.xml"
    
    ov_model_mo_preprocess = None
    
    if ir_path_mo_preprocess.exists():
        ov_model_mo_preprocess = core.read_model(model=ir_path_mo_preprocess)
        print(f"Model in OpenVINO format already exists: {ir_path_mo_preprocess}")
    else: 
        ov_model_mo_preprocess = mo.convert_model(saved_model_dir=model_path,
                                                  model_name=model_path.name,
                                                  mean_values=[127.5,127.5,127.5],
                                                  scale_values=[127.5,127.5,127.5],
                                                  reverse_input_channels=True,
                                                  input_shape=[1,299,299,3])
        serialize(ov_model_mo_preprocess, str(ir_path_mo_preprocess))

Prepare image
~~~~~~~~~~~~~

.. code:: ipython3

    def prepare_image_mo_preprocess(image_path, model):
        img = cv2.imread(filename=image_path)
    
        input_layer_ir = next(iter(model.inputs))
    
        # N, H, W, C = batch size, height, width, number of channels
        N, H, W, C = input_layer_ir.shape
        # Resize image to the input size expected by the model.
        img = cv2.resize(img, (H, W))
    
        # Fit image data type to expected by the model value
        img = np.float32(img)
    
        # Reshape to match the input shape expected by the model.
        input_tensor = np.expand_dims(img, axis=0)
    
        return input_tensor
    
    
    mo_pp_input_tensor = prepare_image_mo_preprocess(image_path, ov_model_mo_preprocess)
    
    print(f"The shape of the image is {mo_pp_input_tensor.shape}")
    print(f"The data type of the image is {mo_pp_input_tensor.dtype}")


.. parsed-literal::

    The shape of the image is (1, 299, 299, 3)
    The data type of the image is float32


Compile model and perform inerence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    compiled_model_mo_pp = core.compile_model(model=ov_model_mo_preprocess, device_name=device.value)
    
    output_layer = compiled_model_mo_pp.output(0)
    
    result = compiled_model_mo_pp(mo_pp_input_tensor)[output_layer]

Setup preprocessing steps with Preprocessing API and perform inference
----------------------------------------------------------------------

Intuitively, preprocessing API consists of the following parts:

- Tensor - declares user data format, like shape, layout, precision, color format from actual user’s data.
- Steps - describes sequence of preprocessing steps which need to be applied to user data.
- Model - specifies model data format. Usually, precision and shape are already known for model, only additional information, like layout can be specified.

Graph modifications of a model shall be performed after the model is
read from a drive and before it is loaded on the actual device.

Pre-processing support following operations (please, see more details
`here <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1aeacaf406d72a238e31a359798ebdb3b7>`__)
- Mean/Scale Normalization - Converting Precision - Converting layout
(transposing) - Resizing Image - Color Conversion - Custom Operations

Convert model to OpenVINO IR with Model Optimizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The options for preprocessing are not required.

.. code:: ipython3

    ir_path = model_dir / "ir_model" / f"{model_name}.xml"
    
    ppp_model = None
    
    if ir_path.exists():
        ppp_model = core.read_model(model=ir_path)
        print(f"Model in OpenVINO format already exists: {ir_path}")
    else: 
        ppp_model = mo.convert_model(saved_model_dir=model_path,
                                     input_shape=[1,299,299,3])
        serialize(ppp_model, str(ir_path))

Create PrePostProcessor Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`PrePostProcessor() <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PrePostProcessor.html#doxid-classov-1-1preprocess-1-1-pre-post-processor>`__
class enables specifying the preprocessing and postprocessing steps for
a model.

.. code:: ipython3

    from openvino.preprocess import PrePostProcessor
    
    ppp = PrePostProcessor(ppp_model)

Declare User’s Data Format
~~~~~~~~~~~~~~~~~~~~~~~~~~

To address particular input of a model/preprocessor, use the
``PrePostProcessor.input(input_name)`` method. If the model has only one
input, then simple ``PrePostProcessor.input()`` will get a reference to
pre-processing builder for this input (a tensor, the steps, a model). In
general, when a model has multiple inputs/outputs, each one can be
addressed by a tensor name or by its index. By default, information
about user’s input tensor will be initialized to same data
(type/shape/etc) as model’s input parameter. User application can
override particular parameters according to application’s data. Refer to
the following
`page <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1InputTensorInfo.html#doxid-classov-1-1preprocess-1-1-input-tensor-info-1a98fb73ff9178c8c71d809ddf8927faf5>`__
for more information about parameters for overriding.

Below is all the specified input information:

- Precision is ``U8``(unsigned 8-bit integer).
- Size is non-fixed, setup of one determined shape size can be done with ``.set_shape([1, 577, 800, 3])``.
- Layout is ``“NHWC”``. It means, for example: height=577, width=800, channels=3.

The height and width are necessary for resizing, and channels are needed
for mean/scale normalization.

.. code:: ipython3

    from openvino.runtime import Type, Layout
    
    # setup formant of data
    ppp.input().tensor().set_element_type(Type.u8)\
                        .set_spatial_dynamic_shape()\
                        .set_layout(Layout('NHWC'))




.. parsed-literal::

    <openvino._pyopenvino.preprocess.InputTensorInfo at 0x7efb4037f5b0>



Declaring Model Layout
~~~~~~~~~~~~~~~~~~~~~~

Model input already has information about precision and shape.
Preprocessing API is not intended to modify this. The only thing that
may be specified is input data
`layout <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Layout_Overview.html#doxid-openvino-docs-o-v-u-g-layout-overview>`__.

.. code:: ipython3

    input_layer_ir = next(iter(ppp_model.inputs))
    print(f"The input shape of the model is {input_layer_ir.shape}")
    
    ppp.input().model().set_layout(Layout('NHWC'))


.. parsed-literal::

    The input shape of the model is [1,299,299,3]




.. parsed-literal::

    <openvino._pyopenvino.preprocess.InputModelInfo at 0x7efc3610ac70>



Preprocessing Steps
~~~~~~~~~~~~~~~~~~~

Now, the sequence of preprocessing steps can be defined. For more
information about preprocessing steps, see
`here <https://docs.openvino.ai/2023.0/api/ie_python_api/_autosummary/openvino.preprocess.PreProcessSteps.html>`__.

Perform the following:

- Convert ``U8`` to ``FP32`` precision.
- Resize to height/width of a model. Be aware that if a model accepts dynamic size, for example, ``{?, 3, ?, ?}`` resize will not know how to resize the picture. Therefore, in this case, target height/ width should be specified. For more details, see also the `PreProcessSteps.resize() <https://docs.openvino.ai/2023.0/classov_1_1preprocess_1_1PreProcessSteps.html#doxid-classov-1-1preprocess-1-1-pre-process-steps-1a40dab78be1222fee505ed6a13400efe6>`__.
- Subtract mean from each channel.
- Divide each pixel data to appropriate scale value.

There is no need to specify conversion layout. If layouts are different,
then such conversion will be added explicitly.

.. code:: ipython3

    from openvino.preprocess import ResizeAlgorithm
    
    ppp.input().preprocess().convert_element_type(Type.f32) \
                            .resize(ResizeAlgorithm.RESIZE_LINEAR)\
                            .mean([127.5,127.5,127.5])\
                            .scale([127.5,127.5,127.5])




.. parsed-literal::

    <openvino._pyopenvino.preprocess.PreProcessSteps at 0x7efd2c4155b0>



Integrating Steps into a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the preprocessing steps have been finished, the model can be
finally built. It is possible to display PrePostProcessor configuration
for debugging purposes.

.. code:: ipython3

    print(f'Dump preprocessor: {ppp}')
    model_with_preprocess = ppp.build()


.. parsed-literal::

    Dump preprocessor: Input "input_1":
        User's input tensor: [1,?,?,3], [N,H,W,C], u8
        Model's expected tensor: [1,299,299,3], [N,H,W,C], f32
        Pre-processing steps (4):
          convert type (f32): ([1,?,?,3], [N,H,W,C], u8) -> ([1,?,?,3], [N,H,W,C], f32)
          resize to model width/height: ([1,?,?,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
          mean (127.5,127.5,127.5): ([1,299,299,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
          scale (127.5,127.5,127.5): ([1,299,299,3], [N,H,W,C], f32) -> ([1,299,299,3], [N,H,W,C], f32)
    


Load model and perform inference
--------------------------------

.. code:: ipython3

    def prepare_image_api_preprocess(image_path, model=None):
        image = cv2.imread(image_path)
        input_tensor = np.expand_dims(image, 0)
        return input_tensor
    
    
    compiled_model_with_preprocess_api = core.compile_model(model=ppp_model, device_name=device.value)
    
    ppp_output_layer = compiled_model_with_preprocess_api.output(0)
    
    ppp_input_tensor = prepare_image_api_preprocess(image_path)
    results = compiled_model_with_preprocess_api(ppp_input_tensor)[ppp_output_layer][0]

Fit image manually and perform inference
----------------------------------------

Load the model
~~~~~~~~~~~~~~

.. code:: ipython3

    model = core.read_model(model=ir_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)

Load image and fit it to model input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def manual_image_preprocessing(path_to_image, compiled_model):
        input_layer_ir = next(iter(compiled_model.inputs))
    
        # N, H, W, C = batch size, height, width, number of channels
        N, H, W, C = input_layer_ir.shape
        
        # load  image, image will be resized to model input size and converted to RGB
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(H, W), color_mode='rgb')
    
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
    
        # will scale input pixels between -1 and 1
        input_tensor = tf.keras.applications.inception_resnet_v2.preprocess_input(x)
    
        return input_tensor
    
    
    input_tensor = manual_image_preprocessing(image_path, compiled_model)
    print(f"The shape of the image is {input_tensor.shape}")
    print(f"The data type of the image is {input_tensor.dtype}")


.. parsed-literal::

    The shape of the image is (1, 299, 299, 3)
    The data type of the image is float32


Perform inference
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    output_layer = compiled_model.output(0)
    
    result = compiled_model(input_tensor)[output_layer]

Compare results
---------------

Compare results on one image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def check_results(input_tensor, compiled_model, imagenet_classes):
        output_layer = compiled_model.output(0)
    
        results = compiled_model(input_tensor)[output_layer][0]
    
        top_indices = np.argsort(results)[-5:][::-1]
        top_softmax = results[top_indices]
    
        for index, softmax_probability in zip(top_indices, top_softmax):
            print(f"{imagenet_classes[index]}, {softmax_probability:.5f}")
    
        return top_indices, top_softmax
    
    
    # Convert the inference result to a class name.
    imagenet_classes = open("../data/datasets/imagenet/imagenet_2012.txt").read().splitlines()
    imagenet_classes = ['background'] + imagenet_classes
    
    # get result for inference with preprocessing api
    print("Result of inference for preprocessing with ModelOptimizer:")
    res = check_results(mo_pp_input_tensor, compiled_model_mo_pp, imagenet_classes)
    
    print("\n")
    
    # get result for inference with preprocessing api
    print("Result of inference with Preprocessing API:")
    res = check_results(ppp_input_tensor, compiled_model_with_preprocess_api, imagenet_classes)
    
    print("\n")
    
    # get result for inference with the manual preparing of the image
    print("Result of inference with manual image setup:")
    res = check_results(input_tensor, compiled_model, imagenet_classes)


.. parsed-literal::

    Result of inference for preprocessing with ModelOptimizer:
    n02099601 golden retriever, 0.56439
    n02098413 Lhasa, Lhasa apso, 0.35731
    n02108915 French bulldog, 0.00730
    n02111129 Leonberg, 0.00687
    n04404412 television, television system, 0.00317
    
    
    Result of inference with Preprocessing API:
    n02099601 golden retriever, 0.80560
    n02098413 Lhasa, Lhasa apso, 0.10039
    n02108915 French bulldog, 0.01915
    n02111129 Leonberg, 0.00825
    n02097047 miniature schnauzer, 0.00294
    
    
    Result of inference with manual image setup:
    n02098413 Lhasa, Lhasa apso, 0.76848
    n02099601 golden retriever, 0.19304
    n02111129 Leonberg, 0.00725
    n02097047 miniature schnauzer, 0.00290
    n02100877 Irish setter, red setter, 0.00116


Compare performance
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def check_performance(compiled_model, preprocessing_function=None):
        num_images = 1000
    
        start = time.perf_counter()
    
        for _ in range(num_images):
            input_tensor = preprocessing_function(image_path, compiled_model)
            compiled_model(input_tensor)
    
        end = time.perf_counter()
        time_ir = end - start
    
        return time_ir, num_images
    
    
    time_ir, num_images = check_performance(compiled_model_mo_pp, prepare_image_mo_preprocess)
    print(
        f"IR model in OpenVINO Runtime/CPU with preprocessing API: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )
    
    time_ir, num_images = check_performance(compiled_model, manual_image_preprocessing)
    print(
        f"IR model in OpenVINO Runtime/CPU with preprocessing API: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )
    
    time_ir, num_images = check_performance(compiled_model_with_preprocess_api, prepare_image_api_preprocess)
    print(
        f"IR model in OpenVINO Runtime/CPU with preprocessing API: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )


.. parsed-literal::

    IR model in OpenVINO Runtime/CPU with preprocessing API: 0.0200 seconds per image, FPS: 49.90
    IR model in OpenVINO Runtime/CPU with preprocessing API: 0.0153 seconds per image, FPS: 65.52
    IR model in OpenVINO Runtime/CPU with preprocessing API: 0.0187 seconds per image, FPS: 53.59

