From Training to Deployment with TensorFlow and OpenVINO™
=========================================================

.. code:: ipython3

    # @title Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    # https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # Copyright 2018 The TensorFlow Authors
    #
    # Modified for OpenVINO Notebooks

This tutorial demonstrates how to train, convert, and deploy an image
classification model with TensorFlow and OpenVINO. This particular
notebook shows the process where we perform the inference step on the
freshly trained model that is converted to OpenVINO IR with Model
Optimizer. For faster inference speed on the model created in this
notebook, check out the `Post-Training Quantization with TensorFlow
Classification Model <./301-tensorflow-training-openvino-nncf.ipynb>`__
notebook.

This training code comprises the official `TensorFlow Image
Classification
Tutorial <https://www.tensorflow.org/tutorials/images/classification>`__
in its entirety.

The **flower_ir.bin** and **flower_ir.xml** (pre-trained models) can be
obtained by executing the code with ‘Runtime->Run All’ or the Ctrl+F9
command.

TensorFlow Image Classification Training
----------------------------------------

The first part of the tutorial shows how to classify images of flowers
(based on the TensorFlow’s official tutorial). It creates an image
classifier using a ``keras.Sequential`` model, and loads data using
``preprocessing.image_dataset_from_directory``. You will gain practical
experience with the following concepts:

-  Efficiently loading a dataset off disk.
-  Identifying overfitting and applying techniques to mitigate it,
   including data augmentation and Dropout.

This tutorial follows a basic machine learning workflow:

1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model

Import TensorFlow and Other Libraries
-------------------------------------

.. code:: ipython3

    import os
    import sys
    from pathlib import Path
    
    import PIL
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from openvino.runtime import Core
    from openvino.tools import mo
    from openvino.runtime import serialize
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-07-11 23:48:43.746139: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-11 23:48:43.781112: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-11 23:48:44.293875: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Download and Explore the Dataset
--------------------------------

This tutorial uses a dataset of about 3,700 photos of flowers. The
dataset contains 5 sub-directories, one per class:

::

   flower_photo/
     daisy/
     dandelion/
     roses/
     sunflowers/
     tulips/

.. code:: ipython3

    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

After downloading, you should now have a copy of the dataset available.
There are 3,670 total images:

.. code:: ipython3

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)


.. parsed-literal::

    3670


Here are some roses:

.. code:: ipython3

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_13_0.png



.. code:: ipython3

    PIL.Image.open(str(roses[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_14_0.png



And some tulips:

.. code:: ipython3

    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_16_0.png



.. code:: ipython3

    PIL.Image.open(str(tulips[1]))




.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_17_0.png



Load Using keras.preprocessing
------------------------------

Let’s load these images off disk using the helpful
`image_dataset_from_directory <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`__
utility. This will take you from a directory of images on disk to a
``tf.data.Dataset`` in just a couple lines of code. If you like, you can
also write your own data loading code from scratch by visiting the `load
images <https://www.tensorflow.org/tutorials/load_data/images>`__
tutorial.

Create a Dataset
----------------

Define some parameters for the loader:

.. code:: ipython3

    batch_size = 32
    img_height = 180
    img_width = 180

It’s good practice to use a validation split when developing your model.
Let’s use 80% of the images for training, and 20% for validation.

.. code:: ipython3

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.
    Using 2936 files for training.


.. parsed-literal::

    2023-07-11 23:48:45.680125: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1956] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
    Skipping registering GPU devices...


.. code:: ipython3

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


.. parsed-literal::

    Found 3670 files belonging to 5 classes.
    Using 734 files for validation.


You can find the class names in the ``class_names`` attribute on these
datasets. These correspond to the directory names in alphabetical order.

.. code:: ipython3

    class_names = train_ds.class_names
    print(class_names)


.. parsed-literal::

    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


Visualize the Data
------------------

Here are the first 9 images from the training dataset.

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


.. parsed-literal::

    2023-07-11 23:48:46.046287: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:48:46.046887: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_28_1.png


You will train a model using these datasets by passing them to
``model.fit`` in a moment. If you like, you can also manually iterate
over the dataset and retrieve batches of images:

.. code:: ipython3

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


.. parsed-literal::

    (32, 180, 180, 3)
    (32,)


.. parsed-literal::

    2023-07-11 23:48:46.535781: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]
    2023-07-11 23:48:46.536007: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


The ``image_batch`` is a tensor of the shape ``(32, 180, 180, 3)``. This
is a batch of 32 images of shape ``180x180x3`` (the last dimension
refers to color channels RGB). The ``label_batch`` is a tensor of the
shape ``(32,)``, these are corresponding labels to the 32 images.

You can call ``.numpy()`` on the ``image_batch`` and ``labels_batch``
tensors to convert them to a ``numpy.ndarray``.

Configure the Dataset for Performance
-------------------------------------

Let’s make sure to use buffered prefetching so you can yield data from
disk without having I/O become blocking. These are two important methods
you should use when loading data.

``Dataset.cache()`` keeps the images in memory after they’re loaded off
disk during the first epoch. This will ensure the dataset does not
become a bottleneck while training your model. If your dataset is too
large to fit into memory, you can also use this method to create a
performant on-disk cache.

``Dataset.prefetch()`` overlaps data preprocessing and model execution
while training.

Interested readers can learn more about both methods, as well as how to
cache data to disk in the `data performance
guide <https://www.tensorflow.org/guide/data_performance#prefetching>`__.

.. code:: ipython3

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

Standardize the Data
--------------------

The RGB channel values are in the ``[0, 255]`` range. This is not ideal
for a neural network; in general you should seek to make your input
values small. Here, you will standardize values to be in the ``[0, 1]``
range by using a Rescaling layer.

.. code:: ipython3

    normalization_layer = layers.Rescaling(1./255)

Note: The Keras Preprocessing utilities and layers introduced in this
section are currently experimental and may change.

There are two ways to use this layer. You can apply it to the dataset by
calling map:

.. code:: ipython3

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image)) 


.. parsed-literal::

    2023-07-11 23:48:46.736316: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:48:46.736650: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    0.0 1.0


Or, you can include the layer inside your model definition, which can
simplify deployment. Let’s use the second approach here.

Note: you previously resized images using the ``image_size`` argument of
``image_dataset_from_directory``. If you want to include the resizing
logic in your model as well, you can use the
`Resizing <https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing>`__
layer.

Create the Model
----------------

The model consists of three convolution blocks with a max pool layer in
each of them. There’s a fully connected layer with 128 units on top of
it that is activated by a ``relu`` activation function. This model has
not been tuned for high accuracy, the goal of this tutorial is to show a
standard approach.

.. code:: ipython3

    num_classes = 5
    
    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

Compile the Model
-----------------

For this tutorial, choose the ``optimizers.Adam`` optimizer and
``losses.SparseCategoricalCrossentropy`` loss function. To view training
and validation accuracy for each training epoch, pass the ``metrics``
argument.

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

Model Summary
-------------

View all the layers of the network using the model’s ``summary`` method.

   **NOTE:** This section is commented out for performance reasons.
   Please feel free to uncomment these to compare the results.

.. code:: ipython3

    # model.summary()

Train the Model
---------------

.. code:: ipython3

    # epochs=10
    # history = model.fit(
    #   train_ds,
    #   validation_data=val_ds,
    #   epochs=epochs
    # )

Visualize Training Results
--------------------------

Create plots of loss and accuracy on the training and validation sets.

.. code:: ipython3

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    
    # epochs_range = range(epochs)
    
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

As you can see from the plots, training accuracy and validation accuracy
are off by large margin and the model has achieved only around 60%
accuracy on the validation set.

Let’s look at what went wrong and try to increase the overall
performance of the model.

Overfitting
-----------

In the plots above, the training accuracy is increasing linearly over
time, whereas validation accuracy stalls around 60% in the training
process. Also, the difference in accuracy between training and
validation accuracy is noticeable — a sign of
`overfitting <https://www.tensorflow.org/tutorials/keras/overfit_and_underfit>`__.

When there are a small number of training examples, the model sometimes
learns from noises or unwanted details from training examples—to an
extent that it negatively impacts the performance of the model on new
examples. This phenomenon is known as overfitting. It means that the
model will have a difficult time generalizing on a new dataset.

There are multiple ways to fight overfitting in the training process. In
this tutorial, you’ll use *data augmentation* and add *Dropout* to your
model.

Data Augmentation
-----------------

Overfitting generally occurs when there are a small number of training
examples. `Data
augmentation <https://www.tensorflow.org/tutorials/images/data_augmentation>`__
takes the approach of generating additional training data from your
existing examples by augmenting them using random transformations that
yield believable-looking images. This helps expose the model to more
aspects of the data and generalize better.

You will implement data augmentation using the layers from
``tf.keras.layers.experimental.preprocessing``. These can be included
inside your model like other layers, and run on the GPU.

.. code:: ipython3

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

Let’s visualize what a few augmented examples look like by applying data
augmentation to the same image several times:

.. code:: ipython3

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


.. parsed-literal::

    2023-07-11 23:48:47.665043: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:48:47.666032: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [2936]
    	 [[{{node Placeholder/_0}}]]



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_56_1.png


You will use data augmentation to train a model in a moment.

Dropout
-------

Another technique to reduce overfitting is to introduce
`Dropout <https://developers.google.com/machine-learning/glossary#dropout_regularization>`__
to the network, a form of *regularization*.

When you apply Dropout to a layer it randomly drops out (by setting the
activation to zero) a number of output units from the layer during the
training process. Dropout takes a fractional number as its input value,
in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20%
or 40% of the output units randomly from the applied layer.

Let’s create a new neural network using ``layers.Dropout``, then train
it using augmented images.

.. code:: ipython3

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

Compile and Train the Model
---------------------------

.. code:: ipython3

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

.. code:: ipython3

    model.summary()


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


.. code:: ipython3

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


.. parsed-literal::

    Epoch 1/15


.. parsed-literal::

    2023-07-11 23:48:48.689479: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]
    2023-07-11 23:48:48.689842: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_4' with dtype int32 and shape [2936]
    	 [[{{node Placeholder/_4}}]]


.. parsed-literal::

    92/92 [==============================] - ETA: 0s - loss: 1.2325 - accuracy: 0.4789

.. parsed-literal::

    2023-07-11 23:48:55.021722: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]
    2023-07-11 23:48:55.022051: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype string and shape [734]
    	 [[{{node Placeholder/_0}}]]


.. parsed-literal::

    92/92 [==============================] - 7s 67ms/step - loss: 1.2325 - accuracy: 0.4789 - val_loss: 1.1056 - val_accuracy: 0.5681
    Epoch 2/15
    92/92 [==============================] - 6s 64ms/step - loss: 1.0170 - accuracy: 0.5943 - val_loss: 0.9563 - val_accuracy: 0.6281
    Epoch 3/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.9168 - accuracy: 0.6468 - val_loss: 0.8525 - val_accuracy: 0.6553
    Epoch 4/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.8412 - accuracy: 0.6754 - val_loss: 0.9478 - val_accuracy: 0.6417
    Epoch 5/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.7881 - accuracy: 0.6856 - val_loss: 0.8132 - val_accuracy: 0.6839
    Epoch 6/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.7470 - accuracy: 0.7163 - val_loss: 0.8087 - val_accuracy: 0.6907
    Epoch 7/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.7111 - accuracy: 0.7302 - val_loss: 0.7582 - val_accuracy: 0.7234
    Epoch 8/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.6980 - accuracy: 0.7398 - val_loss: 0.7545 - val_accuracy: 0.7180
    Epoch 9/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.6437 - accuracy: 0.7578 - val_loss: 0.7517 - val_accuracy: 0.7003
    Epoch 10/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.6150 - accuracy: 0.7711 - val_loss: 0.7419 - val_accuracy: 0.7139
    Epoch 11/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5926 - accuracy: 0.7715 - val_loss: 0.7543 - val_accuracy: 0.7248
    Epoch 12/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5702 - accuracy: 0.7841 - val_loss: 0.6891 - val_accuracy: 0.7466
    Epoch 13/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.5467 - accuracy: 0.7977 - val_loss: 0.7306 - val_accuracy: 0.7234
    Epoch 14/15
    92/92 [==============================] - 6s 64ms/step - loss: 0.5182 - accuracy: 0.8052 - val_loss: 0.7316 - val_accuracy: 0.7371
    Epoch 15/15
    92/92 [==============================] - 6s 63ms/step - loss: 0.4997 - accuracy: 0.8048 - val_loss: 0.6754 - val_accuracy: 0.7411


Visualize Training Results
--------------------------

After applying data augmentation and Dropout, there is less overfitting
than before, and training and validation accuracy are closer aligned.

.. code:: ipython3

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_65_0.png


Predict on New Data
-------------------

Finally, let us use the model to classify an image that was not included
in the training or validation sets.

   **Note**: Data augmentation and Dropout layers are inactive at
   inference time.

.. code:: ipython3

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    
    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    1/1 [==============================] - 0s 71ms/step
    This image most likely belongs to sunflowers with a 97.75 percent confidence.


Save the TensorFlow Model
-------------------------

.. code:: ipython3

    #save the trained model - a new folder flower will be created
    #and the file "saved_model.pb" is the pre-trained model
    model_dir = "model"
    saved_model_dir = f"{model_dir}/flower/saved_model"
    model.save(saved_model_dir)


.. parsed-literal::

    2023-07-11 23:50:18.571028: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2023-07-11 23:50:18.656829: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.666782: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'random_flip_input' with dtype float and shape [?,180,180,3]
    	 [[{{node random_flip_input}}]]
    2023-07-11 23:50:18.677553: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.684368: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.691156: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.701873: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.740670: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2023-07-11 23:50:18.807441: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.827744: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'sequential_1_input' with dtype float and shape [?,180,180,3]
    	 [[{{node sequential_1_input}}]]
    2023-07-11 23:50:18.866385: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.889771: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:18.963275: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:19.105456: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:19.242700: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,22,22,64]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:19.276405: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:19.304418: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    2023-07-11 23:50:19.350826: I tensorflow/core/common_runtime/executor.cc:1197] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'inputs' with dtype float and shape [?,180,180,3]
    	 [[{{node inputs}}]]
    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _update_step_xla while saving (showing 4 of 4). These functions will not be directly callable after loading.


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


.. parsed-literal::

    INFO:tensorflow:Assets written to: model/flower/saved_model/assets


Convert the TensorFlow model with OpenVINO Model Optimizer
----------------------------------------------------------

Use Model Optimizer Python API to convert the model to OpenVINO IR with
``FP16`` precision. For more information about Model Optimizer Python
API, see the `Model Optimizer Developer
Guide <https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_Python_API.html>`__.

.. code:: ipython3

    # Convert the model to ir model format and save it.
    ir_model_path = Path("model/flower")
    ir_model_path.mkdir(parents=True, exist_ok=True)
    ir_model = mo.convert_model(saved_model_dir=saved_model_dir, input_shape=[1,180,180,3], compress_to_fp16=True)
    serialize(ir_model, str(ir_model_path / "flower_ir.xml"))

Preprocessing Image Function
----------------------------

.. code:: ipython3

    def pre_process_image(imagePath, img_height=180):
        # Model input format
        n, h, w, c = [1, img_height, img_height, 3]
        image = Image.open(imagePath)
        image = image.resize((h, w), resample=Image.BILINEAR)
    
        # Convert to array and change data layout from HWC to CHW
        image = np.array(image)
        input_image = image.reshape((n, h, w, c))
    
        return input_image

OpenVINO Inference Engine Setup
-------------------------------

.. code:: ipython3

    class_names=["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    
    # Initialize OpenVINO runtime
    ie = Core()
    
    # Neural Compute Stick
    # compile the model for the CPU (you can choose manually CPU, GPU, etc.)
    # or let the engine choose the best available device (AUTO)
    compiled_model = ie.compile_model(model=ir_model, device_name="CPU")
    
    del ir_model
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

Run the Inference Step
----------------------

.. code:: ipython3

    # Run inference on the input image...
    inp_img_url = "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
    OUTPUT_DIR = "output"
    inp_file_name = f"A_Close_Up_Photo_of_a_Dandelion.jpg"
    file_path = Path(OUTPUT_DIR)/Path(inp_file_name)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download the image
    download_file(inp_img_url, inp_file_name, directory=OUTPUT_DIR)
    
    # Pre-process the image and get it ready for inference.
    input_image = pre_process_image(file_path)
    
    print(input_image.shape)
    print(input_layer.shape)
    res = compiled_model([input_image])[output_layer]
    
    score = tf.nn.softmax(res[0])
    
    # Show the results
    image = Image.open(file_path)
    plt.imshow(image)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


.. parsed-literal::

    'output/A_Close_Up_Photo_of_a_Dandelion.jpg' already exists.
    (1, 180, 180, 3)
    [1,180,180,3]
    This image most likely belongs to dandelion with a 97.68 percent confidence.



.. image:: 301-tensorflow-training-openvino-with-output_files/301-tensorflow-training-openvino-with-output_77_1.png


The Next Steps
--------------

This tutorial showed how to train a TensorFlow model, how to convert
that model to OpenVINO’s IR format, and how to do inference on the
converted model. For faster inference speed, you can quantize the IR
model. To see how to quantize this model with OpenVINO’s `Post-training
Quantization with NNCF
Tool <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__,
check out the `Post-Training Quantization with TensorFlow Classification
Model <./301-tensorflow-training-openvino-nncf.ipynb>`__ notebook.
