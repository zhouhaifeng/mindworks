# Install OpenVINO™ Runtime via Homebrew {#openvino_docs_install_guides_installing_openvino_brew}

@sphinxdirective

.. meta::
   :description: Learn how to install OpenVINO™ Runtime on Linux and macOS 
                 operating systems, using Homebrew, which is a recommended 
                 installation method for C++ developers.


.. note::

   Installing OpenVINO Runtime from Homebrew is recommended for C++ developers. 
   If you work with Python, consider :doc:`installing OpenVINO from PyPI <openvino_docs_install_guides_installing_openvino_pip>`

You can use `Homebrew <https://brew.sh/>`__ to install OpenVINO Runtime on macOS and Linux. 
OpenVINO™ Development Tools can be installed via PyPI only. 
See `Installing Additional Components <#optional-installing-additional-components>`__ for more information.


.. warning:: 

   By downloading and using this container and the included software, you agree to the terms and conditions of the 
   `software license agreements <https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf>`_.


.. tab-set::

   .. tab-item:: System Requirements
      :sync: system-requirements

      | Full requirement listing is available in:
      | `System Requirements Page <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html>`__
   
   .. tab-item:: Software Requirements
      :sync: software-requirements

      .. tab-set::

         .. tab-item:: Linux
            :sync: linux
         
            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher, 64-bit <https://cmake.org/download/>`__
            * GCC 7.5.0 (for Ubuntu 18.04), GCC 9.3.0 (for Ubuntu 20.04) or GCC 11.3.0 (for Ubuntu 22.04)
            * `Python 3.7 - 3.10, 64-bit <https://www.python.org/downloads/>`__

         .. tab-item:: macOS
            :sync: macos
         
            * `Homebrew <https://brew.sh/>`_
            * `CMake 3.13 or higher <https://cmake.org/download/>`__ (choose "macOS 10.13 or later"). Add ``/Applications/CMake.app/Contents/bin`` to path (for default installation). 
            * `Python 3.7 - 3.11 <https://www.python.org/downloads/mac-osx/>`__ . Install and add it to path.
            * Apple Xcode Command Line Tools. In the terminal, run ``xcode-select --install`` from any directory to install it.
            * (Optional) Apple Xcode IDE (not required for OpenVINO™, but useful for development)
         


Installing OpenVINO Runtime
###########################

1. Make sure that you have installed Homebrew on your system. If not, follow the instructions on `the Homebrew website <https://brew.sh/>`__ to install and configure it.

2. Open a command prompt terminal window, and run the following command to install OpenVINO Runtime:

   .. code-block:: sh

      brew install openvino

3. Check if the installation was successful by listing all Homebrew packages:

   .. code-block:: sh

      brew list


Congratulations, you've finished the installation!

(Optional) Installing Additional Components
###########################################

OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It provides tools like Model Optimizer, Benchmark Tool, Post-Training Optimization Tool, and Open Model Zoo Downloader. If you installed OpenVINO Runtime using Homebrew, OpenVINO Development Tools must be installed separately.

See the **For C++ Developers** section on the :doc:`Install OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>` page for instructions.

OpenCV is necessary to run demos from Open Model Zoo (OMZ). Some OpenVINO samples can also extend their capabilities when compiled with OpenCV as a dependency. To install OpenCV for OpenVINO, see the `instructions on GitHub <https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO>`__.

Uninstalling OpenVINO
#####################

To uninstall OpenVINO via Homebrew, use the following command:

.. code-block:: sh

   brew uninstall openvino


What's Next?
####################

Now that you've installed OpenVINO Runtime, you can try the following things:

* Learn more about :doc:`OpenVINO Workflow <openvino_workflow>`.
* To prepare your models for working with OpenVINO, see :doc:`Model Preparation <openvino_docs_model_processing_introduction>`.
* See pre-trained deep learning models in our :doc:`Open Model Zoo <model_zoo>`.
* Learn more about :doc:`Inference with OpenVINO Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.
* See sample applications in :doc:`OpenVINO toolkit Samples Overview <openvino_docs_OV_UG_Samples_Overview>`.
* Check out the OpenVINO product home page: https://software.intel.com/en-us/openvino-toolkit.



@endsphinxdirective
