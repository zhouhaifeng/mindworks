# Quantizing with Accuracy Control {#quantization_w_accuracy_control}

@sphinxdirective

Introduction
####################

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the :doc:`Basic 8-bit quantization <basic_quantization_flow>` and has the following differences:

* Besides the calibration dataset, a **validation dataset** is required to compute the accuracy metric. Both datasets can refer to the same data in the simplest case.
* **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
* Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the :doc:`Basic 8-bit quantization <basic_quantization_flow>` flow.
* The resulted model can provide smaller performance improvement than the :doc:`Basic 8-bit quantization <basic_quantization_flow>` flow because some of the operations are kept in the original precision.

.. note:: Currently, 8-bit quantization with accuracy control is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described below.

Prepare calibration and validation datasets
############################################

This step is similar to the :doc:`Basic 8-bit quantization <basic_quantization_flow>` flow. The only difference is that two datasets, calibration and validation, are required.

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [dataset]

Prepare validation function
############################

Validation function receives ``openvino.runtime.CompiledModel`` object and validation dataset and returns accuracy metric value. The following code snippet shows an example of validation function for OpenVINO model:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [validation]

Run quantization with accuracy control
#######################################

``nncf.quantize_with_accuracy_control()`` function is used to run the quantization with accuracy control. The following code snippet shows an example of quantization with accuracy control for OpenVINO model:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [quantization]

* ``max_drop`` defines the accuracy drop threshold. The quantization process stops when the degradation of accuracy metric on the validation dataset is less than the ``max_drop``. The default value is 0.01. NNCF will stop the quantization and report an error if the ``max_drop`` value can't be reached.

* ``drop_type`` defines how the accuracy drop will be calculated: ``ABSOLUTE`` (used by default) or ``RELATIVE``.

After that the model can be compiled and run with OpenVINO:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/ptq/code/ptq_aa_openvino.py
         :language: python
         :fragment: [inference]

``nncf.quantize_with_accuracy_control()`` API supports all the parameters from :doc:`Basic 8-bit quantization <basic_quantization_flow>` API, to quantize a model with accuracy control and a custom configuration.

If the accuracy or performance of the quantized model is not satisfactory, you can try :doc:`Training-time Optimization <tmo_introduction>` as the next step.

Examples of NNCF post-training quantization with control of accuracy metric:
#############################################################################

* `Post-Training Quantization of Anomaly Classification OpenVINO model with control of accuracy metric <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/quantize_with_accuracy_control>`__
* `Post-Training Quantization of YOLOv8 OpenVINO Model with control of accuracy metric <https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control>`__

See also
####################

* :doc:`Optimizing Models at Training Time <tmo_introduction>` 

@endsphinxdirective

