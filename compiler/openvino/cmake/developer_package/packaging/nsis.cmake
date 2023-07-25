# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(ov_nsis_specific_settings)
    # installation directory
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "Intel")

    # TODO: provide icons
    # set(CPACK_NSIS_MUI_ICON "")
    # set(CPACK_NSIS_MUI_UNIICON "${CPACK_NSIS_MUI_ICON}")
    # set(CPACK_NSIS_MUI_WELCOMEFINISHPAGE_BITMAP "")
    # set(CPACK_NSIS_MUI_UNWELCOMEFINISHPAGE_BITMAP "")
    # set(CPACK_NSIS_MUI_HEADERIMAGE "")

    # we allow to install several packages at once
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL OFF)
    set(CPACK_NSIS_MODIFY_PATH OFF)

    set(CPACK_NSIS_DISPLAY_NAME "Intel(R) OpenVINO(TM) ${OpenVINO_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "Intel(R) OpenVINO(TM) ToolKit, v. ${OpenVINO_VERSION}.${OpenVINO_PATCH_VERSION}")

    # contact
    set(CPACK_NSIS_CONTACT "CPACK_NSIS_CONTACT")

    # links in menu
    set(CPACK_NSIS_MENU_LINKS "https://docs.openvinoo.ai" "OpenVINO Documentation")

    # welcome and finish titles
    set(CPACK_NSIS_WELCOME_TITLE "Welcome to Intel(R) Distribution of OpenVINO(TM) Toolkit installation")
    set(CPACK_NSIS_FINISH_TITLE "")

    # autoresize?
    set(CPACK_NSIS_MANIFEST_DPI_AWARE ON)

    # branding text
    set(CPACK_NSIS_BRANDING_TEXT "Intel(R) Corp.")
    set(CPACK_NSIS_BRANDING_TEXT_TRIM_POSITION RIGHT)

    # don't set this variable since we need a user to agree with a lincense
    # set(CPACK_NSIS_IGNORE_LICENSE_PAGE OFF)
endmacro()

ov_nsis_specific_settings()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. NSIS packages don't require wheels to be packacged)
#

macro(ov_define_component_include_rules)
    # core components
    unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL)
    # licensing
    unset(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL)
    # samples
    unset(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL)
    # python
    unset(OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL)
    set(OV_CPACK_COMP_BENCHMARK_APP_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    set(OV_CPACK_COMP_OVC_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # tools
    unset(OV_CPACK_COMP_OPENVINO_DEV_REQ_FILES_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_DEPLOYMENT_MANAGER_EXCLUDE_ALL)
    # scripts
    unset(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL)
endmacro()

ov_define_component_include_rules()
