# Inference with OpenVINO Runtime {#openvino_docs_OV_UG_OV_Runtime_User_Guide}

@sphinxdirective

.. _deep learning openvino runtime:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Integrate_OV_with_your_application
   openvino_docs_Runtime_Inference_Modes_Overview
   openvino_docs_OV_UG_Working_with_devices
   openvino_docs_OV_UG_ShapeInference
   openvino_docs_OV_UG_DynamicShapes
   openvino_docs_OV_UG_model_state_intro
   Optimize Inference <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>

.. meta::
   :description: OpenVINO Runtime is an API comprised of a set of C++ libraries 
                 with C and Python bindings and it delivers inference solutions 
                 on different platforms.


OpenVINO Runtime is a set of C++ libraries with C and Python bindings providing a common API to deliver inference solutions on the platform of your choice. Use the OpenVINO Runtime API to read an Intermediate Representation (IR), TensorFlow, TensorFlow Lite, ONNX, or PaddlePaddle model and execute it on preferred devices.

OpenVINO Runtime uses a plugin architecture. Its plugins are software components that contain complete implementation for inference on a particular Intel® hardware device: CPU, GPU, GNA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs for configuring devices or API interoperability between OpenVINO Runtime and underlying plugin backend.

The scheme below illustrates the typical workflow for deploying a trained deep learning model:


.. image:: _static/images/BASIC_FLOW_IE_C.svg


Video
####################


.. list-table::

   * - .. raw:: html

           <iframe allowfullscreen mozallowfullscreen msallowfullscreen oallowfullscreen webkitallowfullscreen height="315" width="560"
           src="https://www.youtube.com/embed/e6R13V8nbak">
           </iframe>
   * - **OpenVINO Runtime Concept**. Duration: 3:43

@endsphinxdirective
