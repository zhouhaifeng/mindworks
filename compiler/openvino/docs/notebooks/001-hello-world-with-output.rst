Hello Image Classification
==========================

This basic introduction to OpenVINO™ shows how to do inference with an
image classification model.

A pre-trained `MobileNetV3
model <https://docs.openvino.ai/2023.0/omz_models_model_mobilenet_v3_small_1_0_224_tf.html>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__ is used in
this tutorial. For more information about how OpenVINO IR models are
created, refer to the `TensorFlow to
OpenVINO <101-tensorflow-classification-to-openvino-with-output.html>`__
tutorial.

Imports
-------

.. code:: ipython3

    from pathlib import Path
    import sys
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from openvino.runtime import Core
    
    sys.path.append("../utils")
    from notebook_utils import download_file

Download the Model and data samples
-----------------------------------

.. code:: ipython3

    base_artifacts_dir = Path('./artifacts').expanduser()
    
    model_name = "v3-small_224_1.0_float"
    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'
    
    model_xml_path = base_artifacts_dir / model_xml_name
    
    base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'
    
    if not model_xml_path.exists():
        download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
        download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
    else:
        print(f'{model_name} already downloaded to {base_artifacts_dir}')



.. parsed-literal::

    artifacts/v3-small_224_1.0_float.xml:   0%|          | 0.00/294k [00:00<?, ?B/s]



.. parsed-literal::

    artifacts/v3-small_224_1.0_float.bin:   0%|          | 0.00/4.84M [00:00<?, ?B/s]


Select inference device
-----------------------

select device from dropdown list for running inference using OpenVINO

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



Load the Model
--------------

.. code:: ipython3

    core = Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    
    output_layer = compiled_model.output(0)

Load an Image
-------------

.. code:: ipython3

    # The MobileNet model expects images in RGB format.
    image = cv2.cvtColor(cv2.imread(filename="../data/image/coco.jpg"), code=cv2.COLOR_BGR2RGB)
    
    # Resize to MobileNet image shape.
    input_image = cv2.resize(src=image, dsize=(224, 224))
    
    # Reshape to model input shape.
    input_image = np.expand_dims(input_image, 0)
    plt.imshow(image);



.. image:: 001-hello-world-with-output_files/001-hello-world-with-output_10_0.png


Do Inference
------------

.. code:: ipython3

    result_infer = compiled_model([input_image])[output_layer]
    result_index = np.argmax(result_infer)

.. code:: ipython3

    # Convert the inference result to a class name.
    imagenet_classes = open("../data/datasets/imagenet/imagenet_2012.txt").read().splitlines()
    
    # The model description states that for this model, class 0 is a background.
    # Therefore, a background must be added at the beginning of imagenet_classes.
    imagenet_classes = ['background'] + imagenet_classes
    
    imagenet_classes[result_index]




.. parsed-literal::

    'n02099267 flat-coated retriever'


