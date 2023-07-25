Post-Training Quantization with TensorFlow Classification Model
===============================================================

This example demonstrates how to quantize the OpenVINO model that was
created in
`301-tensorflow-training-openvino.ipynb <301-tensorflow-training-openvino.ipynb>`__,
to improve inference speed. Quantization is performed with
`Post-training Quantization with
NNCF <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__.
A custom dataloader and metric will be defined, and accuracy and
performance will be computed for the original IR model and the quantized
model.

Preparation
-----------

The notebook requires that the training notebook has been run and that
the Intermediate Representation (IR) models are created. If the IR
models do not exist, running the next cell will run the training
notebook. This will take a while.

.. code:: ipython3

    from pathlib import Path
    
    import tensorflow as tf
    
    model_xml = Path("model/flower/flower_ir.xml")
    dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    )
    data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))
    
    if not model_xml.exists():
        print("Executing training notebook. This will take a while...")
        %run 301-tensorflow-training-openvino.ipynb


.. parsed-literal::

    2023-07-11 23:45:46.901183: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-11 23:45:46.936281: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-11 23:45:47.523785: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    Executing training notebook. This will take a while...
    3670
    Found 3670 files belonging to 5 classes.
    Using 2936 files for training.


.. parsed-literal::

    2023-07-11 23:45:49.130715: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...


.. parsed-literal::

    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


.. parsed-literal::

    2023-07-11 23:45:49.466286: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:49.466548: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_2_5.png


.. parsed-literal::

    2023-07-11 23:45:49.938600: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:49.938843: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:50.062635: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:50.062905: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)
    0.0 0.9967369


.. parsed-literal::

    2023-07-11 23:45:50.838129: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:50.838521: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_2_9.png


.. parsed-literal::

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     sequential_1 (Sequential)   (None, 180, 180, 3)       0         
                                                                     
     rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         
                                                                     
     conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         
     2D)                                                             
                                                                     
     conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                     
     max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         
     2D)                                                             
                                                                     
     conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                     
     max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         
     2D)                                                             
                                                                     
     dropout (Dropout)           (None, 22, 22, 64)        0         
                                                                     
     flatten_1 (Flatten)         (None, 30976)             0         
                                                                     
     dense_2 (Dense)             (None, 128)               3965056   
                                                                     
     outputs (Dense)             (None, 5)                 645       
                                                                     
    =================================================================
    Total params: 3,989,285
    Trainable params: 3,989,285
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/15


.. parsed-literal::

    2023-07-11 23:45:51.781311: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:51.781738: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.3257 - accuracy: 0.4353

.. parsed-literal::

    2023-07-11 23:45:58.082635: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:45:58.082922: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    92/92 [==============================] - 7s 66ms/step - loss: 1.3257 - accuracy: 0.4353 - val_loss: 1.1364 - val_accuracy: 0.5341
    Epoch 2/15
    92/92 [==============================] - 6s 63ms/step - loss: 1.0419 - accuracy: 0.5872 - val_loss: 1.0635 - val_accuracy: 0.5886
    Epoch 3/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.9311 - accuracy: 0.6352 - val_loss: 0.9998 - val_accuracy: 0.6131
    Epoch 4/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.8423 - accuracy: 0.6788 - val_loss: 0.9286 - val_accuracy: 0.6703
    Epoch 5/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.8117 - accuracy: 0.6832 - val_loss: 0.8297 - val_accuracy: 0.6567
    Epoch 6/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.7639 - accuracy: 0.7078 - val_loss: 0.7671 - val_accuracy: 0.7112
    Epoch 7/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.7212 - accuracy: 0.7204 - val_loss: 0.7973 - val_accuracy: 0.6962
    Epoch 8/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.6783 - accuracy: 0.7435 - val_loss: 0.8085 - val_accuracy: 0.6717
    Epoch 9/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.6554 - accuracy: 0.7500 - val_loss: 0.7403 - val_accuracy: 0.7193
    Epoch 10/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.6283 - accuracy: 0.7629 - val_loss: 0.6977 - val_accuracy: 0.7153
    Epoch 11/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.6220 - accuracy: 0.7592 - val_loss: 0.7095 - val_accuracy: 0.7343
    Epoch 12/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5818 - accuracy: 0.7813 - val_loss: 0.7068 - val_accuracy: 0.7234
    Epoch 13/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5503 - accuracy: 0.7922 - val_loss: 0.6985 - val_accuracy: 0.7207
    Epoch 14/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5497 - accuracy: 0.7953 - val_loss: 0.6983 - val_accuracy: 0.7439
    Epoch 15/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5285 - accuracy: 0.8028 - val_loss: 0.7165 - val_accuracy: 0.7425



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_2_15.png


.. parsed-literal::

    1/1 [==============================] - 0s 76ms/step
    This image most likely belongs to sunflowers with a 92.57 percent confidence.


.. parsed-literal::

    2023-07-11 23:47:21.447950: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2023-07-11 23:47:21.533588: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.543460: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2023-07-11 23:47:21.555445: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.562382: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.569165: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.579854: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.618942: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2023-07-11 23:47:21.686887: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.707365: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2023-07-11 23:47:21.746351: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.769890: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.843307: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:21.985172: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:22.122564: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:22.156382: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:22.184461: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:47:22.230863: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets



.. parsed-literal::

    output/A_Close_Up_Photo_of_a_Dandelion.jpg:   0%|          | 0.00/21.7k [00:00<?, ?B/s]


.. parsed-literal::

    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 95.29 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_2_22.png


Imports
~~~~~~~

The Post Training Quantization API is implemented in the ``nncf``
library.

.. code:: ipython3

    import sys
    
    import matplotlib.pyplot as plt
    import numpy as np
    import nncf
    from openvino.runtime import Core
    from openvino.runtime import serialize
    from PIL import Image
    from sklearn.metrics import accuracy_score
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Post-training Quantization with NNCF
------------------------------------

`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop.

Create a quantized model from the pre-trained FP32 model and the
calibration dataset. The optimization process contains the following
steps:

1. Create a Dataset for quantization.
2. Run nncf.quantize for getting an optimized model.

The validation dataset already defined in the training notebook.

.. code:: ipython3

    img_height = 180
    img_width = 180
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=1
    )
    
    for a, b in val_dataset:
        print(type(a), type(b))
        break


.. parsed-literal::

    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.
    <class 'tensorflow.python.framework.ops.EagerTensor'> <class 'tensorflow.python.framework.ops.EagerTensor'>


.. parsed-literal::

    2023-07-11 23:47:25.157142: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:47:25.157516: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [734]
    	 [[{{node Placeholder/_4}}]]


The validation dataset can be reused in quantization process. But it
returns a tuple (images, labels), whereas calibration_dataset should
only return images. The transformation function helps to transform a
user validation dataset to the calibration dataset.

.. code:: ipython3

    def transform_fn(data_item):
        """
        The transformation function transforms a data item into model input data.
        This function should be passed when the data item cannot be used as model's input.
        """
        images, _ = data_item
        return images.numpy()
    
    
    calibration_dataset = nncf.Dataset(val_dataset, transform_fn)

Download Intermediate Representation (IR) model.

.. code:: ipython3

    ie = Core()
    ir_model = ie.read_model(model_xml)

Use `Basic Quantization
Flow <https://docs.openvino.ai/2023.0/basic_qauntization_flow.html#doxid-basic-qauntization-flow>`__.
To use the most advanced quantization flow that allows to apply 8-bit
quantization to the model with accuracy control see `Quantizing with
accuracy
control <https://docs.openvino.ai/2023.0/quantization_w_accuracy_control.html#>`__.

.. code:: ipython3

    quantized_model = nncf.quantize(
        ir_model,
        calibration_dataset,
        subset_size=1000
    )


.. parsed-literal::

    Statistics collection:  73%|███████▎  | 734/1000 [00:04<00:01, 163.35it/s]
    Biases correction: 100%|██████████| 5/5 [00:01<00:00,  3.98it/s]


Save quantized model to benchmark.

.. code:: ipython3

    compressed_model_dir = Path("model/optimized")
    compressed_model_dir.mkdir(parents=True, exist_ok=True)
    compressed_model_xml = compressed_model_dir / "flower_ir.xml"
    serialize(quantized_model, str(compressed_model_xml))

Compare Metrics
---------------

Define a metric to determine the performance of the model.

For this demo we define validate function to compute accuracy metrics.

.. code:: ipython3

    def validate(model, validation_loader):
        """
        Evaluate model and compute accuracy metrics.
    
        :param model: Model to validate
        :param validation_loader: Validation dataset
        :returns: Accuracy scores
        """
        predictions = []
        references = []
    
        output = model.outputs[0]
    
        for images, target in validation_loader:
            pred = model(images.numpy())[output]
    
            predictions.append(np.argmax(pred, axis=1))
            references.append(target)
    
        predictions = np.concatenate(predictions, axis=0)
        references = np.concatenate(references, axis=0)
    
        scores = accuracy_score(references, predictions)
    
        return scores

Calculate accuracy for the original model and the quantized model.

.. code:: ipython3

    original_compiled_model = ie.compile_model(model=ir_model, device_name="CPU")
    quantized_compiled_model = ie.compile_model(model=quantized_model, device_name="CPU")
    
    original_accuracy = validate(original_compiled_model, val_dataset)
    quantized_accuracy = validate(quantized_compiled_model, val_dataset)
    
    print(f"Accuracy of the original model: {original_accuracy:.3f}")
    print(f"Accuracy of the quantized model: {quantized_accuracy:.3f}")


.. parsed-literal::

    Accuracy of the original model: 0.743
    Accuracy of the quantized model: 0.741


Compare file size of the models.

.. code:: ipython3

    original_model_size = model_xml.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = compressed_model_xml.with_suffix(".bin").stat().st_size / 1024
    
    print(f"Original model size: {original_model_size:.2f} KB")
    print(f"Quantized model size: {quantized_model_size:.2f} KB")


.. parsed-literal::

    Original model size: 7791.65 KB
    Quantized model size: 3897.08 KB


So, we can see that the original and quantized models have similar
accuracy with a much smaller size of the quantized model.

Run Inference on Quantized Model
--------------------------------

Copy the preprocess function from the training notebook and run
inference on the quantized model with Inference Engine. See the
`OpenVINO API tutorial <002-openvino-api-with-output.html>`__
for more information about running inference with Inference Engine
Python API.

.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, c, h, w = [1, 3, img_height, img_height]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)
    
        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
    
        input_image = image.reshape((n, h, w, c))
    
        return input_image

.. code:: ipython3

    # Get the names of the input and output layer
    # model_pot = ie.read_model(model="model/optimized/flower_ir.xml")
    input_layer = quantized_compiled_model.input(0)
    output_layer = quantized_compiled_model.output(0)
    
    # Get the class names: a list of directory names in alphabetical order
    class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])
    
    # Run inference on an input image...
    inp_img_url = (
        "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    )
    directory = "output"
    inp_file_name = "A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(directory)/Path(inp_file_name)
    # Download the image if it does not exist yet
    if not Path(inp_file_name).exists():
        download_file(inp_img_url, inp_file_name, directory=directory)
    
    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(imagePath=file_path)
    print(f'input image shape: {input_image.shape}')
    print(f'input layer shape: {input_layer.shape}')
    
    res = quantized_compiled_model([input_image])[output_layer]
    
    score = tf.nn.softmax(res[0])
    
    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    input image shape: (1, 180, 180, 3)
    input layer shape: [1,180,180,3]
    This image most likely belongs to dandelion with a 95.55 percent confidence.



.. image:: 301-tensorflow-training-openvino-nncf-with-output_files/301-tensorflow-training-openvino-nncf-with-output_24_1.png


Compare Inference Speed
-----------------------

Measure inference speed with the `OpenVINO Benchmark
App <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__.

Benchmark App is a command line tool that measures raw inference
performance for a specified OpenVINO IR model. Run
``benchmark_app --help`` to see a list of available parameters. By
default, Benchmark App tests the performance of the model specified with
the ``-m`` parameter with asynchronous inference on CPU, for one minute.
Use the ``-d`` parameter to test performance on a different device, for
example an Intel integrated Graphics (iGPU), and ``-t`` to set the
number of seconds to run inference. See the
`documentation <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
for more information.

This tutorial uses a wrapper function from `Notebook
Utils <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/utils/notebook_utils.ipynb>`__.
It prints the ``benchmark_app`` command with the chosen parameters.

In the next cells, inference speed will be measured for the original and
quantized model on CPU. If an iGPU is available, inference speed will be
measured for CPU+GPU as well. The number of seconds is set to 15.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    # print the available devices on this system
    print("Device information:")
    print(ie.get_property("CPU", "FULL_DEVICE_NAME"))
    if "GPU" in ie.available_devices:
        print(ie.get_property("GPU", "FULL_DEVICE_NAME"))


.. parsed-literal::

    Device information:
    Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


.. code:: ipython3

    # Original model - CPU
    ! benchmark_app -m $model_xml -d CPU -t 15 -api async


.. parsed-literal::

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
    [ INFO ] Read model took 11.63 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 68.32 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
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
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.07 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            57672 iterations
    [ INFO ] Duration:         15004.09 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.92 ms
    [ INFO ]    Average:       2.94 ms
    [ INFO ]    Min:           1.83 ms
    [ INFO ]    Max:           10.49 ms
    [ INFO ] Throughput:   3843.75 FPS


.. code:: ipython3

    # Quantized model - CPU
    ! benchmark_app -m $compressed_model_xml -d CPU -t 15 -api async


.. parsed-literal::

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
    [ INFO ] Read model took 13.24 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : f32 / [...] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     sequential_1_input (node: sequential_1_input) : u8 / [N,H,W,C] / [1,180,180,3]
    [ INFO ] Model outputs:
    [ INFO ]     outputs (node: sequential_2/outputs/BiasAdd) : f32 / [...] / [1,5]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 56.10 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: TensorFlow_Frontend_IR
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   NUM_STREAMS: 12
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
    [ WARNING ] No input files were given for input 'sequential_1_input'!. This input will be filled with random values!
    [ INFO ] Fill input 'sequential_1_input' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 1.97 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            179064 iterations
    [ INFO ] Duration:         15001.51 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.92 ms
    [ INFO ]    Average:       0.92 ms
    [ INFO ]    Min:           0.55 ms
    [ INFO ]    Max:           4.95 ms
    [ INFO ] Throughput:   11936.40 FPS


**Benchmark on MULTI:CPU,GPU**

With a recent Intel CPU, the best performance can often be achieved by
doing inference on both the CPU and the iGPU, with OpenVINO’s `Multi
Device
Plugin <https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_MULTI.html>`__.
It takes a bit longer to load a model on GPU than on CPU, so this
benchmark will take a bit longer to complete than the CPU benchmark,
when run for the first time. Benchmark App supports caching, by
specifying the ``--cdir`` parameter. In the cells below, the model will
cached to the ``model_cache`` directory.

.. code:: ipython3

    # Original model - MULTI:CPU,GPU
    if "GPU" in ie.available_devices:
        ! benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
    else:
        print("A supported integrated GPU is not available on this system.")


.. parsed-literal::

    A supported integrated GPU is not available on this system.


.. code:: ipython3

    # Quantized model - MULTI:CPU,GPU
    if "GPU" in ie.available_devices:
        ! benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
    else:
        print("A supported integrated GPU is not available on this system.")


.. parsed-literal::

    A supported integrated GPU is not available on this system.


.. code:: ipython3

    # print the available devices on this system
    print("Device information:")
    print(ie.get_property("CPU", "FULL_DEVICE_NAME"))
    if "GPU" in ie.available_devices:
        print(ie.get_property("GPU", "FULL_DEVICE_NAME"))


.. parsed-literal::

    Device information:
    Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz


**Original IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            58656 iterations
    [ INFO ] Duration:         15003.30 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        2.88 ms
    [ INFO ]    Average:       2.88 ms
    [ INFO ]    Min:           1.26 ms
    [ INFO ]    Max:           11.15 ms
    [ INFO ] Throughput:   3909.54 FPS


**Quantized IR model - CPU**

.. code:: ipython3

    benchmark_output = %sx benchmark_app -m $compressed_model_xml -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))


.. parsed-literal::

    [ INFO ] Count:            179484 iterations
    [ INFO ] Duration:         15001.08 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        0.92 ms
    [ INFO ]    Average:       0.92 ms
    [ INFO ]    Min:           0.56 ms
    [ INFO ]    Max:           4.81 ms
    [ INFO ] Throughput:   11964.74 FPS


**Original IR model - MULTI:CPU,GPU**

With a recent Intel CPU, the best performance can often be achieved by
doing inference on both the CPU and the iGPU, with OpenVINO’s `Multi
Device
Plugin <https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_Running_on_multiple_devices.html>`__.
It takes a bit longer to load a model on GPU than on CPU, so this
benchmark will take a bit longer to complete than the CPU benchmark.

.. code:: ipython3

    if "GPU" in ie.available_devices:
        benchmark_output = %sx benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
        # Remove logging info from benchmark_app output and show only the results
        benchmark_result = benchmark_output[-8:]
        print("\n".join(benchmark_result))
    else:
        print("An GPU is not available on this system.")


.. parsed-literal::

    An GPU is not available on this system.


**Quantized IR model - MULTI:CPU,GPU**

.. code:: ipython3

    if "GPU" in ie.available_devices:
        benchmark_output = %sx benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
        # Remove logging info from benchmark_app output and show only the results
        benchmark_result = benchmark_output[-8:]
        print("\n".join(benchmark_result))
    else:
        print("An GPU is not available on this system.")


.. parsed-literal::

    An GPU is not available on this system.

