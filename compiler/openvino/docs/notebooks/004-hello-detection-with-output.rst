Hello Object Detection
======================

A very basic introduction to using object detection models with
OpenVINO™.

The
`horizontal-text-detection-0001 <https://docs.openvino.ai/2023.0/omz_models_model_horizontal_text_detection_0001.html>`__
model from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__ is used. It
detects horizontal text in images and returns a blob of data in the
shape of ``[100, 5]``. Each detected text box is stored in the
``[x_min, y_min, x_max, y_max, conf]`` format, where the
``(x_min, y_min)`` are the coordinates of the top left bounding box
corner, ``(x_max, y_max)`` are the coordinates of the bottom right
bounding box corner and ``conf`` is the confidence for the predicted
class.

Imports
-------

.. code:: ipython3

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from openvino.runtime import Core
    from pathlib import Path
    import sys
    
    sys.path.append("../utils")
    from notebook_utils import download_file

Download model weights
----------------------

.. code:: ipython3

    base_model_dir = Path("./model").expanduser()
    
    model_name = "horizontal-text-detection-0001"
    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'
    
    model_xml_path = base_model_dir / model_xml_name
    model_bin_path = base_model_dir / model_bin_name
    
    if not model_xml_path.exists():
        model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
        model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"
    
        download_file(model_xml_url, model_xml_name, base_model_dir)
        download_file(model_bin_url, model_bin_name, base_model_dir)
    else:
        print(f'{model_name} already downloaded to {base_model_dir}')



.. parsed-literal::

    model/horizontal-text-detection-0001.xml:   0%|          | 0.00/680k [00:00<?, ?B/s]



.. parsed-literal::

    model/horizontal-text-detection-0001.bin:   0%|          | 0.00/7.39M [00:00<?, ?B/s]


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



Load the Model
--------------

.. code:: ipython3

    ie = Core()
    
    model = ie.read_model(model=model_xml_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output("boxes")

Load an Image
-------------

.. code:: ipython3

    # Text detection models expect an image in BGR format.
    image = cv2.imread("../data/image/intel_rnb.jpg")
    
    # N,C,H,W = batch size, number of channels, height, width.
    N, C, H, W = input_layer_ir.shape
    
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))
    
    # Reshape to the network input shape.
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));



.. image:: 004-hello-detection-with-output_files/004-hello-detection-with-output_10_0.png


Do Inference
------------

.. code:: ipython3

    # Create an inference request.
    boxes = compiled_model([input_image])[output_layer_ir]
    
    # Remove zero only boxes.
    boxes = boxes[~np.all(boxes == 0, axis=1)]

Visualize Results
-----------------

.. code:: ipython3

    # For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
    # The image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib, use cvtColor function
    def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
        # Define colors for boxes and descriptions.
        colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    
        # Fetch the image shapes to calculate a ratio.
        (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    
        # Convert the base image from BGR to RGB format.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
        # Iterate through non-zero boxes.
        for box in boxes:
            # Pick a confidence factor from the last place in an array.
            conf = box[-1]
            if conf > threshold:
                # Convert float to int and multiply corner position of each box by x and y ratio.
                # If the bounding box is found at the top of the image, 
                # position the upper box bar little lower to make it visible on the image. 
                (x_min, y_min, x_max, y_max) = [
                    int(max(corner_position * ratio_y, 10)) if idx % 2 
                    else int(corner_position * ratio_x)
                    for idx, corner_position in enumerate(box[:-1])
                ]
    
                # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
                rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
    
                # Add text to the image based on position and confidence.
                # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
                if conf_labels:
                    rgb_image = cv2.putText(
                        rgb_image,
                        f"{conf:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        colors["red"],
                        1,
                        cv2.LINE_AA,
                    )
    
        return rgb_image

.. code:: ipython3

    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(convert_result_to_image(image, resized_image, boxes, conf_labels=False));



.. image:: 004-hello-detection-with-output_files/004-hello-detection-with-output_15_0.png

