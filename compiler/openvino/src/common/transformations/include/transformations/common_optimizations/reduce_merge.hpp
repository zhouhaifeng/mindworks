// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReduceMerge;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ReduceMerge transformation matches following graph:
 *
 *    +----------+         +----------+
 *    |    A     |         |    B     |
 *    +----------+         +----------+
 *         |                    |
 *         ---------    ---------
 *                 |    |
 *                 v    v
 *               +--------+   +--------+
 *               | Reduce |   |    C   |
 *               +--------+   +--------+
 *                   |             |
 *                   |       -------
 *                   |       |
 *                   v       v
 *                  +----------+
 *                  |  Reduce  |
 *                  +----------+
 *
 *
 * and replaces with:
 *
 *           +----------+     +----------+
 *           |    B     |     |    C     |
 *           +----------+     +----------+
 *                |                |
 *                -------    -------
 *                      |    |
 *                      v    v
 *    +----------+   +-------------------+
 *    |     A    |   |  Concat/Constant  |
 *    +----------+   +-------------------+
 *          |             |
 *          |      --------
 *          |      |
 *          v      v
 *        +----------+
 *        |  Reduce  |
 *        +----------+
 *
 */
class ov::pass::ReduceMerge : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceMerge", "0");
    ReduceMerge();
};
