Convert a PaddlePaddle Model to OpenVINO™ IR
============================================

This notebook shows how to convert a MobileNetV3 model from
`PaddleHub <https://github.com/PaddlePaddle/PaddleHub>`__, pre-trained
on the `ImageNet <https://www.image-net.org>`__ dataset, to OpenVINO IR.
It also shows how to perform classification inference on a sample image,
using `OpenVINO
Runtime <https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and compares the results of the
`PaddlePaddle <https://github.com/PaddlePaddle/Paddle>`__ model with the
IR model.

Source of the
`model <https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v3_large_imagenet_ssld&en_category=ImageClassification>`__.

Preparation
-----------

Imports
~~~~~~~

.. code:: ipython3

    import sys
    
    if sys.version_info.minor > 7:
        !pip install -q "paddlepaddle>=2.5.0"
    else:
        !pip install -q "paddlepaddle==2.4.2"

.. code:: ipython3

    !pip install -q paddleclas --no-deps
    !pip install -q "prettytable" "ujson" "visualdl>=2.2.0" "faiss-cpu>=1.7.1"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    paddleclas 2.5.1 requires easydict, which is not installed.
    paddleclas 2.5.1 requires faiss-cpu==1.7.1.post2, but you have faiss-cpu 1.7.4 which is incompatible.
    paddleclas 2.5.1 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    

.. code:: ipython3

    import time
    import tarfile
    from pathlib import Path
    import sys
    
    import matplotlib.pyplot as plt
    import numpy as np
    from paddleclas import PaddleClas
    from PIL import Image
    from openvino.runtime import Core
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-07-11 22:26:05 INFO: Loading faiss with AVX2 support.
    2023-07-11 22:26:05 INFO: Successfully loaded faiss with AVX2 support.


Settings
~~~~~~~~

Set ``IMAGE_FILENAME`` to the filename of an image to use. Set
``MODEL_NAME`` to the PaddlePaddle model to download from PaddleHub.
``MODEL_NAME`` will also be the base name for the IR model. The notebook
is tested with the
`mobilenet_v3_large_x1_0 <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/models/Mobile_en.md>`__
model. Other models may use different preprocessing methods and
therefore require some modification to get the same results on the
original and converted model.

First of all, we need to download and unpack model files. The first time
you run this notebook, the PaddlePaddle model is downloaded from
PaddleHub. This may take a while.

.. code:: ipython3

    IMAGE_FILENAME = "../data/image/coco_close.png"
    
    MODEL_NAME = "MobileNetV3_large_x1_0"
    MODEL_DIR = Path("model")
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    MODEL_URL = 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar'.format(MODEL_NAME)
    download_file(MODEL_URL, directory=MODEL_DIR)
    file = tarfile.open(MODEL_DIR / '{}_infer.tar'.format(MODEL_NAME))
    res = file.extractall(MODEL_DIR)
    if not res:
        print(f"Model Extracted to \"./{MODEL_DIR}\".")
    else:
        print("Error Extracting the model. Please check the network.")



.. parsed-literal::

    model/MobileNetV3_large_x1_0_infer.tar:   0%|          | 0.00/19.5M [00:00<?, ?B/s]


.. parsed-literal::

    Model Extracted to "./model".


Show Inference on PaddlePaddle Model
------------------------------------

In the next cell, we load the model, load and display an image, do
inference on that image, and then show the top three prediction results.

.. code:: ipython3

    classifier = PaddleClas(inference_model_dir=MODEL_DIR / '{}_infer'.format(MODEL_NAME))
    result = next(classifier.predict(IMAGE_FILENAME))
    class_names = result[0]['label_names']
    scores = result[0]['scores']
    image = Image.open(IMAGE_FILENAME)
    plt.imshow(image)
    for class_name, softmax_probability in zip(class_names, scores):
        print(f"{class_name}, {softmax_probability:.5f}")


.. parsed-literal::

    [2023/07/11 22:26:25] ppcls WARNING: The current running environment does not support the use of GPU. CPU has been used instead.
    Labrador retriever, 0.75138
    German short-haired pointer, 0.02373
    Great Dane, 0.01848
    Rottweiler, 0.01435
    flat-coated retriever, 0.01144



.. image:: 103-paddle-to-openvino-classification-with-output_files/103-paddle-to-openvino-classification-with-output_8_1.png


``classifier.predict()`` takes an image file name, reads the image,
preprocesses the input, then returns the class labels and scores of the
image. Preprocessing the image is done behind the scenes. The
classification model returns an array with floating point values for
each of the 1000 ImageNet classes. The higher the value, the more
confident the network is that the class number corresponding to that
value (the index of that value in the network output array) is the class
number for the image.

To see PaddlePaddle’s implementation for the classification function and
for loading and preprocessing data, uncomment the next two cells.

.. code:: ipython3

    # classifier??

.. code:: ipython3

    # classifier.get_config()

The ``classifier.get_config()`` module shows the preprocessing
configuration for the model. It should show that images are normalized,
resized and cropped, and that the BGR image is converted to RGB before
propagating it through the network. In the next cell, we get the
``classifier.predictror.preprocess_ops`` property that returns list of
preprocessing operations to do inference on the OpenVINO IR model using
the same method.

.. code:: ipython3

    preprocess_ops = classifier.predictor.preprocess_ops
    
    
    def process_image(image):
        for op in preprocess_ops:
            image = op(image)
        return image

It is useful to show the output of the ``process_image()`` function, to
see the effect of cropping and resizing. Because of the normalization,
the colors will look strange, and matplotlib will warn about clipping
values.

.. code:: ipython3

    pil_image = Image.open(IMAGE_FILENAME)
    processed_image = process_image(np.array(pil_image))
    print(f"Processed image shape: {processed_image.shape}")
    # Processed image is in (C,H,W) format, convert to (H,W,C) to show the image
    plt.imshow(np.transpose(processed_image, (1, 2, 0)))


.. parsed-literal::

    2023-07-11 22:26:25 WARNING: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


.. parsed-literal::

    Processed image shape: (3, 224, 224)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fd87c6ee3a0>




.. image:: 103-paddle-to-openvino-classification-with-output_files/103-paddle-to-openvino-classification-with-output_15_3.png


To decode the labels predicted by the model to names of classes, we need
to have a mapping between them. The model config contains information
about ``class_id_map_file``, which stores such mapping. The code below
shows how to parse the mapping into a dictionary to use with the
OpenVINO model.

.. code:: ipython3

    class_id_map_file = classifier.get_config()['PostProcess']['Topk']['class_id_map_file']
    class_id_map = {}
    with open(class_id_map_file, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            partition = line.split("\n")[0].partition(" ")
            class_id_map[int(partition[0])] = str(partition[-1])

Convert the Model to OpenVINO IR Format
---------------------------------------

Call the OpenVINO Model Optimizer Python API to convert the PaddlePaddle
model to OpenVINO IR, with FP32 precision. ``mo.convert_model`` function
accept path to PaddlePaddle model and returns OpenVINO Model class
instance which represents this model. Obtained model is ready to use and
loading on device using ``compile_model`` or can be saved on disk using
``serialize`` function. See the `Model Optimizer Developer
Guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__
for more information about Model Optimizer.

.. code:: ipython3

    from openvino.tools import mo
    from openvino.runtime import serialize
    
    model_xml = Path(MODEL_NAME).with_suffix('.xml')
    if not model_xml.exists():
        ov_model = mo.convert_model("model/MobileNetV3_large_x1_0_infer/inference.pdmodel")
        serialize(ov_model, str(model_xml))
    else:
        print(f"{model_xml} already exists.")

Select inference device
-----------------------

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    
    ie = Core()
    device = widgets.Dropdown(
        options=ie.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Show Inference on OpenVINO Model
--------------------------------

Load the IR model, get model information, load the image, do inference,
convert the inference to a meaningful result, and show the output. See
the `OpenVINO Runtime API
Notebook <002-openvino-api-with-output.html>`__ for more
information.

.. code:: ipython3

    # Load OpenVINO Runtime and OpenVINO IR model
    core = Core()
    model = core.read_model(model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    
    # Get model output
    output_layer = compiled_model.output(0)
    
    # Read, show, and preprocess input image
    # See the "Show Inference on PaddlePaddle Model" section for source of process_image
    image = Image.open(IMAGE_FILENAME)
    plt.imshow(image)
    input_image = process_image(np.array(image))[None,]
    
    # Do inference
    ov_result = compiled_model([input_image])[output_layer][0]
    
    # find the top three values
    top_indices = np.argsort(ov_result)[-3:][::-1]
    top_scores = ov_result[top_indices]
    
    # Convert the inference results to class names, using the same labels as the PaddlePaddle classifier
    for index, softmax_probability in zip(top_indices, top_scores):
        print(f"{class_id_map[index]}, {softmax_probability:.5f}")


.. parsed-literal::

    Labrador retriever, 0.75138
    German short-haired pointer, 0.02373
    Great Dane, 0.01848



.. image:: 103-paddle-to-openvino-classification-with-output_files/103-paddle-to-openvino-classification-with-output_23_1.png


Timing and Comparison
---------------------

Measure the time it takes to do inference on fifty images and compare
the result. The timing information gives an indication of performance.
For a fair comparison, we include the time it takes to process the
image. For more accurate benchmarking, use the `OpenVINO benchmark
tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__.
Note that many optimizations are possible to improve the performance.

.. code:: ipython3

    num_images = 50
    
    image = Image.open(fp=IMAGE_FILENAME)

.. code:: ipython3

    # Show device information
    core = Core()
    devices = core.available_devices
    
    for device_name in devices:
        device_full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
        print(f"{device_name}: {device_full_name}")


.. parsed-literal::

    CPU: Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


.. code:: ipython3

    # Show inference speed on PaddlePaddle model
    start = time.perf_counter()
    for _ in range(num_images):
        result = next(classifier.predict(np.array(image)))
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"PaddlePaddle model on CPU: {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}\n"
    )
    print("PaddlePaddle result:")
    class_names = result[0]['label_names']
    scores = result[0]['scores']
    for class_name, softmax_probability in zip(class_names, scores):
        print(f"{class_name}, {softmax_probability:.5f}")
    plt.imshow(image);


.. parsed-literal::

    PaddlePaddle model on CPU: 0.0074 seconds per image, FPS: 135.48
    
    PaddlePaddle result:
    Labrador retriever, 0.75138
    German short-haired pointer, 0.02373
    Great Dane, 0.01848
    Rottweiler, 0.01435
    flat-coated retriever, 0.01144



.. image:: 103-paddle-to-openvino-classification-with-output_files/103-paddle-to-openvino-classification-with-output_27_1.png


Select inference device
-----------------------

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # Show inference speed on OpenVINO IR model
    compiled_model = ie.compile_model(model=model, device_name=device.value)
    output_layer = compiled_model.output(0)
    
    
    start = time.perf_counter()
    input_image = process_image(np.array(image))[None,]
    for _ in range(num_images):
        ie_result = compiled_model([input_image])[output_layer][0]
        top_indices = np.argsort(ie_result)[-5:][::-1]
        top_softmax = ie_result[top_indices]
    
    end = time.perf_counter()
    time_ir = end - start
    
    print(
        f"OpenVINO IR model in OpenVINO Runtime ({device.value}): {time_ir/num_images:.4f} "
        f"seconds per image, FPS: {num_images/time_ir:.2f}"
    )
    print()
    print("OpenVINO result:")
    for index, softmax_probability in zip(top_indices, top_softmax):
        print(f"{class_id_map[index]}, {softmax_probability:.5f}")
    plt.imshow(image);


.. parsed-literal::

    OpenVINO IR model in OpenVINO Runtime (AUTO): 0.0031 seconds per image, FPS: 326.50
    
    OpenVINO result:
    Labrador retriever, 0.75138
    German short-haired pointer, 0.02373
    Great Dane, 0.01848
    Rottweiler, 0.01435
    flat-coated retriever, 0.01144



.. image:: 103-paddle-to-openvino-classification-with-output_files/103-paddle-to-openvino-classification-with-output_30_1.png


References
----------

-  `PaddleClas <https://github.com/PaddlePaddle/PaddleClas>`__
-  `OpenVINO PaddlePaddle
   support <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle.html>`__
-  `OpenVINO Model Optimizer
   Documentation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html>`__
