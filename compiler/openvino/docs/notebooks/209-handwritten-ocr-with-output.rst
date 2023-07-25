Handwritten Chinese and Japanese OCR with OpenVINO™
===================================================

In this tutorial, we perform optical character recognition (OCR) for
handwritten Chinese (simplified) and Japanese. An OCR tutorial using the
Latin alphabet is available in `notebook
208 <208-optical-character-recognition-with-output.html>`__.
This model is capable of processing only one line of symbols at a time.

The models used in this notebook are
`handwritten-japanese-recognition <https://docs.openvino.ai/2023.0/omz_models_model_handwritten_japanese_recognition_0001.html>`__
and
`handwritten-simplified-chinese <https://docs.openvino.ai/2023.0/omz_models_model_handwritten_simplified_chinese_recognition_0001.html>`__.
To decode model outputs as readable text
`kondate_nakayosi <https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/kondate_nakayosi.txt>`__
and
`scut_ept <https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/dataset_classes/scut_ept.txt>`__
charlists are used. Both models are available on `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__.

Imports
-------

.. code:: ipython3

    from collections import namedtuple
    from itertools import groupby
    from pathlib import Path
    
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from openvino.runtime import Core

Settings
--------

Set up all constants and folders used in this notebook

.. code:: ipython3

    # Directories where data will be placed.
    model_folder = "model"
    data_folder = "../data"
    charlist_folder = f"{data_folder}/text"
    
    # Precision used by the model.
    precision = "FP16"

To group files, you have to define the collection. In this case, use
``namedtuple``.

.. code:: ipython3

    Language = namedtuple(
        typename="Language", field_names=["model_name", "charlist_name", "demo_image_name"]
    )
    chinese_files = Language(
        model_name="handwritten-simplified-chinese-recognition-0001",
        charlist_name="chinese_charlist.txt",
        demo_image_name="handwritten_chinese_test.jpg",
    )
    japanese_files = Language(
        model_name="handwritten-japanese-recognition-0001",
        charlist_name="japanese_charlist.txt",
        demo_image_name="handwritten_japanese_test.png",
    )

Select a Language
-----------------

Depending on your choice you will need to change a line of code in the
cell below.

If you want to perform OCR on a text in Japanese, set
``language = "japanese"``. For Chinese, set ``language = "chinese"``.

.. code:: ipython3

    # Select the language by using either language="chinese" or language="japanese".
    language = "chinese"
    
    languages = {"chinese": chinese_files, "japanese": japanese_files}
    
    selected_language = languages.get(language)

Download the Model
------------------

In addition to images and charlists, you need to download the model
file. In the sections below, there are cells for downloading either the
Chinese or Japanese model.

If it is your first time running the notebook, the model will be
downloaded. It may take a few minutes.

Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. It automatically creates a directory structure
and downloads the selected model.

.. code:: ipython3

    path_to_model_weights = Path(f'{model_folder}/intel/{selected_language.model_name}/{precision}/{selected_language.model_name}.bin')
    if not path_to_model_weights.is_file():
        download_command = f'omz_downloader --name {selected_language.model_name} --output_dir {model_folder} --precision {precision}'
        print(download_command)
        ! $download_command


.. parsed-literal::

    omz_downloader --name handwritten-simplified-chinese-recognition-0001 --output_dir model --precision FP16
    ################|| Downloading handwritten-simplified-chinese-recognition-0001 ||################
    
    ========== Downloading model/intel/handwritten-simplified-chinese-recognition-0001/FP16/handwritten-simplified-chinese-recognition-0001.xml
    
    
    ========== Downloading model/intel/handwritten-simplified-chinese-recognition-0001/FP16/handwritten-simplified-chinese-recognition-0001.bin
    
    


Load the Network and Execute
----------------------------

When all files are downloaded and language is selected, read and compile
the network to run inference. The path to the model is defined based on
the selected language.

.. code:: ipython3

    ie = Core()
    path_to_model = path_to_model_weights.with_suffix(".xml")
    model = ie.read_model(model=path_to_model)

Select a Device Name
~~~~~~~~~~~~~~~~~~~~

You may choose to run the network on multiple devices. By default, it
will load the model on CPU (CPU, GPU, etc. can be set manually) or let
the engine choose the best available device (AUTO).

To list all available devices, uncomment and run the line
``print(ie.available_devices)``.

.. code:: ipython3

    # To check available device names run the line below
    # print(ie.available_devices)
    
    compiled_model = ie.compile_model(model=model, device_name="CPU")

Fetch Information About Input and Output Layers
-----------------------------------------------

Now that the model is loaded, fetch information about the input and
output layers (shape).

.. code:: ipython3

    recognition_output_layer = compiled_model.output(0)
    recognition_input_layer = compiled_model.input(0)

Load an Image
-------------

Next, load an image. The model expects a single-channel image as input,
so the image is read in grayscale.

After loading the input image, get information to use for calculating
the scale ratio between required input layer height and the current
image height. In the cell below, the image will be resized and padded to
keep letters proportional and meet input shape.

.. code:: ipython3

    # Read a filename of a demo file based on the selected model.
    
    file_name = selected_language.demo_image_name
    
    # Text detection models expect an image in grayscale format.
    # IMPORTANT! This model enables reading only one line at time.
    
    # Read the image.
    image = cv2.imread(filename=f"{data_folder}/image/{file_name}", flags=cv2.IMREAD_GRAYSCALE)
    
    # Fetch the shape.
    image_height, _ = image.shape
    
    # B,C,H,W = batch size, number of channels, height, width.
    _, _, H, W = recognition_input_layer.shape
    
    # Calculate scale ratio between the input shape height and image height to resize the image.
    scale_ratio = H / image_height
    
    # Resize the image to expected input sizes.
    resized_image = cv2.resize(
        image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA
    )
    
    # Pad the image to match input size, without changing aspect ratio.
    resized_image = np.pad(
        resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge"
    )
    
    # Reshape to network input shape.
    input_image = resized_image[None, None, :, :]

Visualise Input Image
---------------------

After preprocessing, you can display the image.

.. code:: ipython3

    plt.figure(figsize=(20, 1))
    plt.axis("off")
    plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255);



.. image:: 209-handwritten-ocr-with-output_files/209-handwritten-ocr-with-output_20_0.png


Prepare Charlist
----------------

The model is loaded and the image is ready. The only element left is the
charlist, which is downloaded. You must add a blank symbol at the
beginning of the charlist before using it. This is expected for both the
Chinese and Japanese models.

.. code:: ipython3

    # Get a dictionary to encode the output, based on model documentation.
    used_charlist = selected_language.charlist_name
    
    # With both models, there should be blank symbol added at index 0 of each charlist.
    blank_char = "~"
    
    with open(f"{charlist_folder}/{used_charlist}", "r", encoding="utf-8") as charlist:
        letters = blank_char + "".join(line.strip() for line in charlist)

Run Inference
-------------

Now, run inference. The ``compiled_model()`` function takes a list with
input(s) in the same order as model input(s). Then, fetch the output
from output tensors.

.. code:: ipython3

    # Run inference on the model
    predictions = compiled_model([input_image])[recognition_output_layer]

Process the Output Data
-----------------------

The output of a model is in the ``W x B x L`` format, where:

-  W - output sequence length
-  B - batch size
-  L - confidence distribution across the supported symbols in Kondate
   and Nakayosi.

To get a more human-readable format, select a symbol with the highest
probability. When you hold a list of indexes that are predicted to have
the highest probability, due to limitations in `CTC
Decoding <https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7>`__,
you will remove concurrent symbols and then remove the blanks.

Finally, get the symbols from corresponding indexes in the charlist.

.. code:: ipython3

    # Remove a batch dimension.
    predictions = np.squeeze(predictions)
    
    # Run the `argmax` function to pick the symbols with the highest probability.
    predictions_indexes = np.argmax(predictions, axis=1)

.. code:: ipython3

    # Use the `groupby` function to remove concurrent letters, as required by CTC greedy decoding.
    output_text_indexes = list(groupby(predictions_indexes))
    
    # Remove grouper objects.
    output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))
    
    # Remove blank symbols.
    output_text_indexes = output_text_indexes[output_text_indexes != 0]
    
    # Assign letters to indexes from the output array.
    output_text = [letters[letter_index] for letter_index in output_text_indexes]

Print the Output
----------------

Now, having a list of letters predicted by the model, you can display
the image with predicted text printed below.

.. code:: ipython3

    plt.figure(figsize=(20, 1))
    plt.axis("off")
    plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
    
    print("".join(output_text))


.. parsed-literal::

    人有悲欢离合，月有阴睛圆缺，此事古难全。



.. image:: 209-handwritten-ocr-with-output_files/209-handwritten-ocr-with-output_29_1.png

