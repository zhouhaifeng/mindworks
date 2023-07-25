Convert a Tensorflow Lite Model to OpenVINO™
============================================

`TensorFlow Lite <https://www.tensorflow.org/lite/guide>`__, often
referred to as TFLite, is an open source library developed for deploying
machine learning models to edge devices.

This short tutorial shows how to convert a TensorFlow Lite
`efficientnet-lite-b0 <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
image classification model to OpenVINO `Intermediate
Representation <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_IR_and_opsets.html>`__
(OpenVINO IR) format, using `Model
Optimizer <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.
After creating the OpenVINO IR, load the model in `OpenVINO
Runtime <https://docs.openvino.ai/nightly/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`__
and do inference with a sample image.

Preparation
-----------

Install requirements
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !pip install -q "openvino-dev>=2023.0.0"
    !pip install -q opencv-python requests tqdm
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    );

Imports
~~~~~~~

.. code:: ipython3

    from pathlib import Path
    import numpy as np
    from PIL import Image
    from openvino.runtime import Core, serialize
    from openvino.tools import mo
    
    from notebook_utils import download_file, load_image

Download TFLite model
---------------------

.. code:: ipython3

    model_dir = Path("model")
    tflite_model_path = model_dir / "efficientnet_lite0_fp32_2.tflite"
    
    ov_model_path = tflite_model_path.with_suffix(".xml")
    model_url = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite"
    
    download_file(model_url, tflite_model_path.name, model_dir)



.. parsed-literal::

    model/efficientnet_lite0_fp32_2.tflite:   0%|          | 0.00/17.7M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-448/.workspace/scm/ov-notebook/notebooks/119-tflite-to-openvino/model/efficientnet_lite0_fp32_2.tflite')



Convert a Model to OpenVINO IR Format
-------------------------------------

To convert the TFLite model to OpenVINO IR, OpenVINO Model Optimizer
Python API can be used. ``mo.convert_model`` function accept path to
TFLite model and returns OpenVINO Model class instance which represents
this model. Obtained model is ready to use and loading on device using
``compile_model`` or can be saved on disk using ``serialize`` function
reducing loading time for next running. Optionally, we can apply
compression to FP16 model weigths using ``compress_to_fp16=True`` option
and integrate preprocessing using this approach. See the `Model
Optimizer Developer
Guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__
for more information about Model Optimizer and TensorFlow Lite `models
suport <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite.html>`__.

.. code:: ipython3

    ov_model = mo.convert_model(tflite_model_path, compress_to_fp16=True)
    serialize(ov_model, ov_model_path)
    print(f"Model {tflite_model_path} successfully converted and saved to {ov_model_path}")


.. parsed-literal::

    Model model/efficientnet_lite0_fp32_2.tflite successfully converted and saved to model/efficientnet_lite0_fp32_2.xml


Load model using OpenVINO TensorFlow Lite Frontend
--------------------------------------------------

TensorFlow Lite models are supported via FrontEnd API. You may skip
conversion to IR and read models directly by OpenVINO runtime API. For
more examples supported formats reading via Frontend API, please look
this `tutorial <../002-openvino-api>`__.

.. code:: ipython3

    core = Core()
    
    ov_model = core.read_model(tflite_model_path)

Run OpenVINO model inference
----------------------------

We can find information about model input preprocessing in its
`description <https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2>`__
on `TensorFlow Hub <https://tfhub.dev/>`__.

.. code:: ipython3

    image = load_image("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bricks.png")
    # load_image reads the image in BGR format, [:,:,::-1] reshape transfroms it to RGB
    image = Image.fromarray(image[:,:,::-1])
    resized_image = image.resize((224, 224))
    input_tensor = np.expand_dims((np.array(resized_image).astype(np.float32) - 127) / 128, 0)

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

    compiled_model = core.compile_model(ov_model)
    predicted_scores = compiled_model(input_tensor)[0]

.. code:: ipython3

    imagenet_classes_file_path = download_file("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt")
    imagenet_classes = open(imagenet_classes_file_path).read().splitlines()
    
    top1_predicted_cls_id = np.argmax(predicted_scores)
    top1_predicted_score = predicted_scores[0][top1_predicted_cls_id]
    predicted_label = imagenet_classes[top1_predicted_cls_id]
    
    display(image.resize((640, 512)))
    print(f"Predicted label: {predicted_label} with probability {top1_predicted_score :2f}")



.. parsed-literal::

    imagenet_2012.txt:   0%|          | 0.00/30.9k [00:00<?, ?B/s]



.. image:: 119-tflite-to-openvino-with-output_files/119-tflite-to-openvino-with-output_16_1.png


.. parsed-literal::

    Predicted label: n02109047 Great Dane with probability 0.715318


Estimate Model Performance
--------------------------

`Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
is used to measure the inference performance of the model on CPU and
GPU.

   **NOTE**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    print("Benchmark model inference on CPU")
    !benchmark_app -m $ov_model_path -d CPU -t 15
    if "GPU" in core.available_devices:
        print("\n\nBenchmark model inference on GPU")
        !benchmark_app -m $ov_model_path -d GPU -t 15


.. parsed-literal::

    Benchmark model inference on CPU
    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.0.0-10926-b4452d56304-releases/2023/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 21.86 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,224,224,3]
    [ INFO ] Model outputs:
    [ INFO ]     Softmax (node: 61) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,H,W,C] / [1,224,224,3]
    [ INFO ] Model outputs:
    [ INFO ]     Softmax (node: 61) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 183.81 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Lite_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.41 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            17364 iterations
    [ INFO ] Duration:         15009.20 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        5.04 ms
    [ INFO ]    Average:       5.05 ms
    [ INFO ]    Min:           3.22 ms
    [ INFO ]    Max:           15.36 ms
    [ INFO ] Throughput:   1156.89 FPS

