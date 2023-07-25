// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/so_extension.hpp"

const ov::Extension::Ptr& ov::detail::SOExtension::extension() const {
    return m_ext;
}

const std::shared_ptr<void> ov::detail::SOExtension::shared_object() const {
    return m_so;
}
