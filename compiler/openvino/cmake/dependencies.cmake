# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_policy(SET CMP0054 NEW)

# TODO: fix it, outside of source dir MO cannot find TBB dependency
set_temp_directory(TEMP "${CMAKE_SOURCE_DIR}")

if(ENABLE_SAME_BRANCH_FOR_MODELS)
    branchName(MODELS_BRANCH)
else()
    set(MODELS_BRANCH "master")
endif()

if(ENABLE_DATA)
    add_models_repo(${ENABLE_DATA} "data:https://github.com/openvinotoolkit/testdata.git")
    set(MODELS_PATH "${TEMP}/models/src/data")
    set(DATA_PATH "${MODELS_PATH}")
endif()

message(STATUS "MODELS_PATH=" ${MODELS_PATH})

fetch_models_and_validation_set()

## Intel OMP package
if(THREADING STREQUAL "OMP")
    reset_deps_cache(OMP)
    if(WIN32 AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_WIN "iomp.zip"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "62c68646747fb10f19b53217cb04a1e10ff93606f992e6b35eb8c31187c68fbf"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_LIN "iomp.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "7832b16d82513ee880d97c27c7626f9525ebd678decf6a8fe6c38550f73227d9"
                USE_NEW_LOCATION TRUE)
    elseif(APPLE AND X86_64)
        RESOLVE_DEPENDENCY(OMP
                ARCHIVE_MAC "iomp_20190130_mac.tgz"
                TARGET_PATH "${TEMP}/omp"
                ENVIRONMENT "OMP"
                VERSION_REGEX ".*_([a-z]*_([a-z0-9]+\\.)*[0-9]+).*"
                SHA256 "591ea4a7e08bbe0062648916f42bded71d24c27f00af30a8f31a29b5878ea0cc"
                USE_NEW_LOCATION TRUE)
    else()
        message(FATAL_ERROR "Intel OMP is not available on current platform")
    endif()
    update_deps_cache(OMP "${OMP}" "Path to OMP root folder")
    debug_message(STATUS "intel_omp=" ${OMP})

    ov_cpack_add_component(omp HIDDEN)
    file(GLOB_RECURSE source_list "${OMP}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
    install(FILES ${source_list}
            DESTINATION ${OV_CPACK_RUNTIMEDIR}
            COMPONENT omp)
endif()

## TBB package
unset(_ov_download_tbb_done CACHE)

#
# The function downloads prebuilt TBB package
# NOTE: the function should be used if system TBB is not found
# or ENABLE_SYSTEM_TBB is OFF
#
function(ov_download_tbb)
    if(_ov_download_tbb_done OR NOT THREADING MATCHES "^(TBB|TBB_AUTO)$")
        return()
    endif()
    set(_ov_download_tbb_done ON CACHE INTERNAL "Whether prebuilt TBB is already downloaded")

    reset_deps_cache(TBBROOT TBB_DIR)

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    endif()

    if(NOT DEFINED ENV{TBBROOT} AND (DEFINED ENV{TBB_DIR} OR DEFINED TBB_DIR))
        if(DEFINED ENV{TBB_DIR})
            set(TEMP_ROOT $ENV{TBB_DIR})
        elseif (DEFINED TBB_DIR)
            set(TEMP_ROOT ${TBB_DIR})
        endif()
        while(NOT EXISTS "${TEMP_ROOT}/include")
            get_filename_component(TEMP_ROOT ${TEMP_ROOT} PATH)
        endwhile()
        set(TBBROOT ${TEMP_ROOT})
    endif()

    if(WIN32 AND X86_64)
        # TODO: add target_path to be platform specific as well, to avoid following if
        # build oneTBB 2021.2.1 with Visual Studio 2019 (MSVC 14.21)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_WIN "oneapi-tbb-2021.2.2-win.zip"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "103b19a8af288c6a7d83ed3f0d2239c4afd0dd189fc12aad1d34b3c9e78df94b"
                USE_NEW_LOCATION TRUE)
    elseif(ANDROID AND X86_64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_ANDROID "tbb2020_20200404_android.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "f42d084224cc2d643314bd483ad180b081774608844000f132859fca3e9bf0ce"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND X86_64 AND OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.17)
        # build oneTBB 2021.2.1 with gcc 4.8 (glibc 2.17)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "oneapi-tbb-2021.2.1-lin-canary.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "3a2c2ec79b3cce7e6a2484754ba6f029fa968db2eefc6659540792b7db8fea0c"
                USE_NEW_LOCATION TRUE)
    elseif(YOCTO_AARCH64)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "keembay/tbb2020_38404_kmb_lic.tgz"
                TARGET_PATH "${TEMP}/tbb_yocto"
                ENVIRONMENT "TBBROOT"
                SHA256 "321261ff2eda6d4568a473cb883262bce77a93dac599f7bd65d2918bdee4d75b"
                USE_NEW_LOCATION TRUE)
    elseif(APPLE AND X86_64)
        # build oneTBB 2021.2.1 with OS version 11.4
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_MAC "oneapi-tbb-2021.2.1-mac.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "c57ce4b97116cd3093c33e6dcc147fb1bbb9678d0ee6c61a506b2bfe773232cb"
                USE_NEW_LOCATION TRUE)
    elseif(WIN32 AND AARCH64)
        # build oneTBB 2021.2.1 with Visual Studio 2022 (MSVC 14.35)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_WIN "oneapi-tbb-2021.2.1-win-arm64.zip"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "09fe7f5e7be589aa34ccd20fdfd7cad9e0afa89d1e74ecdb008a75d0af71d6e1"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND AARCH64 AND OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.17)
        # build oneTBB 2021.2.1 with gcc 4.8 (glibc 2.17)
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_LIN "oneapi-tbb-2021.2.1-lin-arm64-canary.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "042fdac53be65841a970b05d892f4b20b556b06fd3b20d2d0068e49c4fd74f07"
                USE_NEW_LOCATION TRUE)
    elseif(APPLE AND AARCH64)
        # build oneTBB 2021.2.1 with export MACOSX_DEPLOYMENT_TARGET=11.0
        RESOLVE_DEPENDENCY(TBB
                ARCHIVE_MAC "oneapi-tbb-2021.2.1-mac-arm64.tgz"
                TARGET_PATH "${TEMP}/tbb"
                ENVIRONMENT "TBBROOT"
                SHA256 "15d46ef19501e4315a5498af59af873dbf8180e9a3ea55253ccf7f0c0bb6f940"
                USE_NEW_LOCATION TRUE)
    else()
        message(WARNING "Prebuilt TBB is not available on current platform")
    endif()

    update_deps_cache(TBBROOT "${TBB}" "Path to TBB root folder")
    if(EXISTS "${TBBROOT}/lib/cmake/TBB/TBBConfig.cmake")
        # oneTBB case
        update_deps_cache(TBB_DIR "${TBBROOT}/lib/cmake/TBB" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/lib/cmake/tbb/TBBConfig.cmake")
        # oneTBB release package version less than 2021.6.0
        update_deps_cache(TBB_DIR "${TBBROOT}/lib/cmake/tbb" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/lib64/cmake/TBB/TBBConfig.cmake")
        # 64-bits oneTBB case
        update_deps_cache(TBB_DIR "${TBBROOT}/lib64/cmake/TBB" "Path to TBB cmake folder")
    elseif(EXISTS "${TBBROOT}/cmake/TBBConfig.cmake")
        # custom downloaded or user provided TBB
        update_deps_cache(TBB_DIR "${TBBROOT}/cmake" "Path to TBB cmake folder")
    else()
        message(WARNING "Failed to find TBBConfig.cmake in ${TBBROOT} tree. Custom TBBConfig.cmake will be used")
    endif()

    debug_message(STATUS "tbb=" ${TBB})
    debug_message(STATUS "tbb_dir=" ${TBB_DIR})
    debug_message(STATUS "tbbroot=" ${TBBROOT})
endfunction()

## TBBBind_2_5 package
unset(_ov_download_tbbbind_2_5_done CACHE)

#
# The function downloads static prebuilt TBBBind_2_5 package
# NOTE: the function should be called only we have TBB with version less 2021
#
function(ov_download_tbbbind_2_5)
    if(_ov_download_tbbbind_2_5_done OR NOT ENABLE_TBBBIND_2_5)
        return()
    endif()
    set(_ov_download_tbbbind_2_5_done ON CACHE INTERNAL "Whether prebuilt TBBBind_2_5 is already downloaded")

    reset_deps_cache(TBBBIND_2_5_ROOT TBBBIND_2_5_DIR)

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    endif()

    if(WIN32 AND X86_64)
        RESOLVE_DEPENDENCY(TBBBIND_2_5
                ARCHIVE_WIN "tbbbind_2_5_static_win_v2.zip"
                TARGET_PATH "${TEMP}/tbbbind_2_5"
                ENVIRONMENT "TBBBIND_2_5_ROOT"
                SHA256 "49ae93b13a13953842ff9ae8d01681b269b5b0bc205daf18619ea9a828c44bee"
                USE_NEW_LOCATION TRUE)
    elseif(LINUX AND X86_64)
        RESOLVE_DEPENDENCY(TBBBIND_2_5
                ARCHIVE_LIN "tbbbind_2_5_static_lin_v3.tgz"
                TARGET_PATH "${TEMP}/tbbbind_2_5"
                ENVIRONMENT "TBBBIND_2_5_ROOT"
                SHA256 "d39deb262c06981b5e2d2e3c593e9fc9be62ce4feb91dd4e648e92753659a6b3"
                USE_NEW_LOCATION TRUE)
    else()
        # TMP: for Apple Silicon TBB does not provide TBBBind
        if(NOT (APPLE AND AARCH64))
            message(WARNING "prebuilt TBBBIND_2_5 is not available.
Build oneTBB from sources and set TBBROOT environment var before OpenVINO cmake configure")
        endif()
        return()
    endif()

    update_deps_cache(TBBBIND_2_5_ROOT "${TBBBIND_2_5}" "Path to TBBBIND_2_5 root folder")
    update_deps_cache(TBBBIND_2_5_DIR "${TBBBIND_2_5}/cmake" "Path to TBBBIND_2_5 cmake folder")
endfunction()

## OpenCV
if(ENABLE_OPENCV)
    reset_deps_cache(OpenCV_DIR)

    set(OPENCV_VERSION "4.5.2")
    set(OPENCV_BUILD "076")
    set(OPENCV_BUILD_YOCTO "772")

    if(YOCTO_AARCH64)
        if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
            set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        elseif(DEFINED THIRDPARTY_SERVER_PATH)
            set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        else()
            message(WARNING "OpenCV is not found!")
        endif()

        if(DEFINED IE_PATH_TO_DEPS)
            set(OPENCV_SUFFIX "yocto_kmb")
            set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")

            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "23c250796ad5fc9db810e1680ccdb32c45dc0e50cace5e0f02b30faf652fe343")

            unset(IE_PATH_TO_DEPS)
        endif()
    else()
        if(WIN32 AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_WIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "a14f872e6b63b6ac12c7ff47fa49e578d14c14433b57f5d85ab5dd48a079938c")
        elseif(APPLE AND X86_64)
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_MAC "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_osx.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_osx/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 "3e162f96e86cba8836618134831d9cf76df0438778b3e27e261dedad9254c514")
        elseif(LINUX)
            if(YOCTO_AARCH64)
                set(OPENCV_SUFFIX "yocto_kmb")
                set(OPENCV_BUILD "${OPENCV_BUILD_YOCTO}")
            elseif((OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.17 AND
                    CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.9") AND X86_64)
                set(OPENCV_SUFFIX "centos7")
                set(OPENCV_HASH "5fa76985c84fe7c64531682ef0b272510c51ac0d0565622514edf1c88b33404a")
            elseif(OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.31 AND X86_64)
                set(OPENCV_SUFFIX "ubuntu20")
                set(OPENCV_HASH "2fe7bbc40e1186eb8d099822038cae2821abf617ac7a16fadf98f377c723e268")
            elseif(OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.27 AND X86_64)
                set(OPENCV_SUFFIX "ubuntu18")
                set(OPENCV_HASH "db087dfd412eedb8161636ec083ada85ff278109948d1d62a06b0f52e1f04202")
            elseif(OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.24 AND ARM)
                set(OPENCV_SUFFIX "debian9arm")
                set(OPENCV_HASH "4274f8c40b17215f4049096b524e4a330519f3e76813c5a3639b69c48633d34e")
            elseif(OV_GLIBC_VERSION VERSION_GREATER_EQUAL 2.23 AND X86_64)
                set(OPENCV_SUFFIX "ubuntu16")
                set(OPENCV_HASH "cd46831b4d8d1c0891d8d22ff5b2670d0a465a8a8285243059659a50ceeae2c3")
            elseif(NOT DEFINED OpenCV_DIR AND NOT DEFINED ENV{OpenCV_DIR})
                message(FATAL_ERROR "OpenCV is not available on current platform (OS = ${CMAKE_SYSTEM_NAME}, glibc ${OV_GLIBC_VERSION})")
            endif()
            RESOLVE_DEPENDENCY(OPENCV
                    ARCHIVE_LIN "opencv/opencv_${OPENCV_VERSION}-${OPENCV_BUILD}_${OPENCV_SUFFIX}.txz"
                    TARGET_PATH "${TEMP}/opencv_${OPENCV_VERSION}_${OPENCV_SUFFIX}/opencv"
                    ENVIRONMENT "OpenCV_DIR"
                    VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+).*"
                    SHA256 ${OPENCV_HASH})
        endif()
    endif()

    if(ANDROID)
        set(ocv_cmake_path "${OPENCV}/sdk/native/jni/")
    else()
        set(ocv_cmake_path "${OPENCV}/cmake")
    endif()

    update_deps_cache(OpenCV_DIR "${ocv_cmake_path}" "Path to OpenCV package folder")
    debug_message(STATUS "opencv=" ${OPENCV})
else()
    reset_deps_cache(OpenCV_DIR)
endif()

if(ENABLE_INTEL_GNA)
    reset_deps_cache(
            GNA_EXT_DIR
            GNA_PLATFORM_DIR
            GNA_KERNEL_LIB_NAME
            GNA_LIBS_LIST
            GNA_LIB_DIR
            libGNA_INCLUDE_DIRS
            libGNA_LIBRARIES_BASE_PATH)
        set(GNA_VERSION "03.05.00.2116")
        set(GNA_HASH "960350567702bda17276ac4c060d7524fb7ce7ced785004bd861c81ff2bfe2c5")

        set(FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/include)
        if(WIN32)
            LIST(APPEND FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/win64)
        else()
            LIST(APPEND FILES_TO_EXTRACT_LIST gna_${GNA_VERSION}/linux)
        endif()

        RESOLVE_DEPENDENCY(GNA_EXT_DIR
                ARCHIVE_UNIFIED "gna/gna_${GNA_VERSION}.zip"
                TARGET_PATH "${TEMP}/gna_${GNA_VERSION}"
                VERSION_REGEX ".*_([0-9]+.[0-9]+.[0-9]+.[0-9]+).*"
                FILES_TO_EXTRACT FILES_TO_EXTRACT_LIST
                SHA256 ${GNA_HASH}
                USE_NEW_LOCATION TRUE)
    update_deps_cache(GNA_EXT_DIR "${GNA_EXT_DIR}" "Path to GNA root folder")
    debug_message(STATUS "gna=" ${GNA_EXT_DIR})

    if (WIN32)
        set(GNA_PLATFORM_DIR win64 CACHE STRING "" FORCE)
    elseif (UNIX)
        set(GNA_PLATFORM_DIR linux CACHE STRING "" FORCE)
    else ()
        message(FATAL_ERROR "GNA not supported on this platform, only linux, and windows")
    endif ()
    set(GNA_LIB_DIR x64 CACHE STRING "" FORCE)
    set(GNA_PATH ${GNA_EXT_DIR}/${GNA_PLATFORM_DIR}/${GNA_LIB_DIR} CACHE STRING "" FORCE)

    if(NOT BUILD_SHARED_LIBS)
        list(APPEND PATH_VARS "GNA_PATH")
    endif()
endif()
