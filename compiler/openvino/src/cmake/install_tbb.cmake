# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(cmake/ie_parallel.cmake)

# pre-find TBB: need to provide TBB_IMPORTED_TARGETS used for installation
ov_find_package_tbb()

# check whether TBB has TBBBind 2.5+ with hwloc 2.5+ or higher which is required
# to detect hybrid cores
function(_ov_detect_dynamic_tbbbind_2_5 var)
    if(NOT TBB_FOUND)
        return()
    endif()

    # try to select proper library directory
    _ov_get_tbb_location(TBB::tbb _tbb_lib_location)
    get_filename_component(_tbb_libs_dir "${_tbb_lib_location}" DIRECTORY)
    # unset for cases if user specified different TBB_DIR / TBBROOT
    unset(_ov_tbbbind_2_5 CACHE)

    find_file(_ov_tbbbind_2_5
              NAMES "${CMAKE_SHARED_LIBRARY_PREFIX}tbbbind_2_5${CMAKE_SHARED_LIBRARY_SUFFIX}"
              HINTS "${_tbb_libs_dir}"
              DOC "Path to TBBBind 2.5+ library"
              NO_DEFAULT_PATH
              NO_CMAKE_FIND_ROOT_PATH)

    if(_ov_tbbbind_2_5)
        set(${var} ON PARENT_SCOPE)
    endif()
endfunction()

_ov_detect_dynamic_tbbbind_2_5(_ov_dynamic_tbbbind_2_5_found)

if(_ov_dynamic_tbbbind_2_5_found)
    message(STATUS "Static tbbbind_2_5 package usage is disabled, since oneTBB (ver. ${TBB_VERSION}) provides dynamic TBBBind 2.5+")
    set(ENABLE_TBBBIND_2_5 OFF)
elseif(ENABLE_TBBBIND_2_5)
    # TMP: for Apple Silicon TBB does not provide TBBBind
    if(TBB_VERSION VERSION_GREATER_EQUAL 2021 AND NOT (APPLE AND AARCH64))
        message(STATUS "oneTBB (ver. ${TBB_VERSION}) is used, but dynamic TBBBind 2.5+ is not found. Use custom static TBBBind 2.5")
    endif()

    # download and find a prebuilt version of TBBBind_2_5
    ov_download_tbbbind_2_5()
    find_package(TBBBIND_2_5 QUIET)

    if(TBBBIND_2_5_FOUND)
        message(STATUS "Static tbbbind_2_5 package is found")
        set_target_properties(${TBBBIND_2_5_IMPORTED_TARGETS} PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS TBBBIND_2_5_AVAILABLE)
        if(NOT BUILD_SHARED_LIBS)
            set(install_tbbbind ON)
        endif()
    else()
        message(STATUS "Prebuilt static tbbbind_2_5 package is not available for current platform (${CMAKE_SYSTEM_NAME})")
    endif()
endif()

unset(_ov_dynamic_tbbbind_2_5_found)

# install TBB

# define variables for OpenVINOConfig.cmake
if(THREADING MATCHES "^(TBB|TBB_AUTO)$")
    set(IE_TBB_DIR "${TBB_DIR}")
    list(APPEND PATH_VARS "IE_TBB_DIR")
endif()

if(install_tbbbind)
    set(IE_TBBBIND_DIR "${TBBBIND_2_5_DIR}")
    list(APPEND PATH_VARS "IE_TBBBIND_DIR")
endif()

# install only downloaded | custom TBB, system one is not installed
# - downloaded TBB should be a part of all packages
# - custom TBB provided by users, needs to be a part of wheel packages
# - system TBB also needs to be a part of wheel packages
if(THREADING MATCHES "^(TBB|TBB_AUTO)$" AND
       ( (DEFINED TBBROOT AND TBBROOT MATCHES ${TEMP}) OR
         (DEFINED TBBROOT OR DEFINED TBB_DIR OR DEFINED ENV{TBBROOT} OR
          DEFINED ENV{TBB_DIR}) OR ENABLE_SYSTEM_TBB ) )
    ov_cpack_add_component(tbb HIDDEN)
    list(APPEND core_components tbb)

    if(TBBROOT MATCHES ${TEMP})
        set(tbb_downloaded ON)
    elseif(DEFINED ENV{TBBROOT} OR DEFINED ENV{TBB_DIR} OR
           DEFINED TBBROOT OR DEFINED TBB_DIR)
        set(tbb_custom ON)
    endif()

    if(OV_GLIBC_VERSION VERSION_LESS_EQUAL 2.26)
        set(_ov_system_tbb_is_obsolete ON)
    endif()

    if(CPACK_GENERATOR MATCHES "^(DEB|RPM|CONDA-FORGE|BREW|CONAN|VCPKG)$" AND
        NOT ENABLE_SYSTEM_TBB AND
        NOT _ov_system_tbb_is_obsolete)
        message(FATAL_ERROR "Debian | RPM | Conda-forge | brew | vcpkg packages can be built only with system TBB. Use -DENABLE_SYSTEM_TBB=ON")
    endif()

    if(ENABLE_SYSTEM_TBB)
        # for system libraries we still need to install TBB libraries
        # so, need to take locations of actual libraries and install them
        foreach(tbb_target IN LISTS TBB_IMPORTED_TARGETS)
            _ov_get_tbb_location(${tbb_target} tbb_lib_location)
            # depending on the TBB, tbb_lib_location can be in form:
            # - libtbb.so.x.y
            # - libtbb.so.x
            # We need to install such only libtbb.so.x files
            get_filename_component(name_we "${tbb_lib_location}" NAME_WE)
            get_filename_component(dir "${tbb_lib_location}" DIRECTORY)
            # grab all tbb files matching pattern
            file(GLOB tbb_files "${dir}/${name_we}.*")

            # since the setup.py for pip installs tbb component
            # explicitly, it's OK to put EXCLUDE_FROM_ALL to such component
            # to ignore from IRC / apt / yum distribution;
            # but they will be present in .wheel
            foreach(tbb_file IN LISTS tbb_files)
                # TODO: check by what name TBB loads the libraries
                ov_install_with_name("${tbb_file}" tbb)
            endforeach()
        endforeach()

        set(pkg_config_tbb_lib_dir "runtime/3rdparty/tbb/lib")
    elseif(tbb_custom)
        # for custom TBB we need to install it to our package
        # to simplify life for our customers
        set(IE_TBBROOT_INSTALL "runtime/3rdparty/tbb")

        # TBBROOT is not defined if ENV{TBBROOT} is not found
        # so, we have to deduce this value ourselves
        if(NOT DEFINED TBBROOT AND DEFINED ENV{TBBROOT})
            file(TO_CMAKE_PATH $ENV{TBBROOT} TBBROOT)
        endif()
        if(NOT DEFINED TBBROOT)
            get_target_property(_tbb_include_dir TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
            get_filename_component(TBBROOT ${_tbb_include_dir} PATH)
        endif()
        if(DEFINED TBBROOT)
            set(TBBROOT "${TBBROOT}" CACHE PATH "TBBROOT path" FORCE)
        else()
            message(FATAL_ERROR "Failed to deduce TBBROOT, please define env var TBBROOT")
        endif()

        if(TBB_DIR MATCHES "^${TBBROOT}.*")
            file(RELATIVE_PATH IE_TBB_DIR_INSTALL "${TBBROOT}" "${TBB_DIR}")
            set(IE_TBB_DIR_INSTALL "${IE_TBBROOT_INSTALL}/${IE_TBB_DIR_INSTALL}")
        else()
            # TBB_DIR is not a subdirectory of TBBROOT
            # example: old TBB 2017 with no cmake support at all
            # - TBBROOT point to actual root of TBB
            # - TBB_DIR points to cmake/developer_package/tbb/<lnx|mac|win>
            set(IE_TBB_DIR_INSTALL "${TBB_DIR}")
        endif()

        # try to select proper library directory
        _ov_get_tbb_location(TBB::tbb _tbb_lib_location)
        get_filename_component(_tbb_libs_dir "${_tbb_lib_location}" DIRECTORY)
        file(RELATIVE_PATH tbb_libs_dir "${TBBROOT}" "${_tbb_libs_dir}")

        # install only meaningful directories
        foreach(dir include ${tbb_libs_dir} cmake lib/cmake lib/pkgconfig lib/intel64/vc14)
            if(EXISTS "${TBBROOT}/${dir}")
                if(dir STREQUAL "include" OR dir MATCHES ".*(cmake|pkgconfig)$" OR dir STREQUAL "lib/intel64/vc14")
                    set(tbb_component tbb_dev)
                    set(core_dev_components tbb_dev)
                    unset(exclude_pattern)
                else()
                    set(tbb_component tbb)
                    set(exclude_pattern REGEX ".*(cmake|pkgconfig)$" EXCLUDE)
                endif()
                install(DIRECTORY "${TBBROOT}/${dir}/"
                        DESTINATION "${IE_TBBROOT_INSTALL}/${dir}"
                        COMPONENT ${tbb_component}
                        ${exclude_pattern})
            endif()
        endforeach()

        set(pkg_config_tbb_lib_dir "${IE_TBBROOT_INSTALL}/${tbb_libs_dir}")
    elseif(tbb_downloaded)
        set(IE_TBB_DIR_INSTALL "runtime/3rdparty/tbb/")

        if(WIN32)
            install(DIRECTORY "${TBBROOT}/bin"
                    DESTINATION "${IE_TBB_DIR_INSTALL}"
                    COMPONENT tbb)
        else()
            install(DIRECTORY "${TBBROOT}/lib"
                    DESTINATION "${IE_TBB_DIR_INSTALL}"
                    COMPONENT tbb
                    PATTERN "cmake" EXCLUDE)
        endif()

        install(FILES "${TBBROOT}/LICENSE"
                DESTINATION "${IE_TBB_DIR_INSTALL}"
                COMPONENT tbb)

        # install development files

        ov_cpack_add_component(tbb_dev
                               HIDDEN
                               DEPENDS tbb)
        list(APPEND core_dev_components tbb_dev)

        if(EXISTS "${TBBROOT}/lib/cmake")
            # oneTBB case
            install(DIRECTORY "${TBBROOT}/lib/cmake"
                    DESTINATION "${IE_TBB_DIR_INSTALL}/lib"
                    COMPONENT tbb_dev)
        else()
            # tbb2020 case
            install(FILES "${TBBROOT}/cmake/TBBConfig.cmake"
                          "${TBBROOT}/cmake/TBBConfigVersion.cmake"
                    DESTINATION "${IE_TBB_DIR_INSTALL}/cmake"
                    COMPONENT tbb_dev)
        endif()

        install(DIRECTORY "${TBBROOT}/include"
                DESTINATION "${IE_TBB_DIR_INSTALL}"
                COMPONENT tbb_dev)

        if(WIN32)
            # .lib files are needed only for Windows
            install(DIRECTORY "${TBBROOT}/lib"
                    DESTINATION "${IE_TBB_DIR_INSTALL}"
                    COMPONENT tbb_dev
                    PATTERN "cmake" EXCLUDE)
        endif()

        set(pkg_config_tbb_lib_dir "${IE_TBB_DIR_INSTALL}/lib")
    else()
        message(WARNING "TBB of unknown origin. TBB files are not installed")
    endif()

    unset(tbb_downloaded)
    unset(tbb_custom)
endif()

# install tbbbind for static OpenVINO case
if(install_tbbbind)
    set(IE_TBBBIND_DIR_INSTALL "runtime/3rdparty/tbb_bind_2_5")

    install(DIRECTORY "${TBBBIND_2_5_ROOT}/lib"
            DESTINATION "${IE_TBBBIND_DIR_INSTALL}"
            COMPONENT tbb)
    install(FILES "${TBBBIND_2_5_ROOT}/LICENSE"
            DESTINATION "${IE_TBBBIND_DIR_INSTALL}"
            COMPONENT tbb)

    install(FILES "${TBBBIND_2_5_ROOT}/cmake/TBBBIND_2_5Config.cmake"
            DESTINATION "${IE_TBBBIND_DIR_INSTALL}/cmake"
            COMPONENT tbb_dev)
endif()
