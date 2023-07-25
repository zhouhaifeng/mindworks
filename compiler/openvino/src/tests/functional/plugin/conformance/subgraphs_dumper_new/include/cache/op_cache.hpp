// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/cache.hpp"

#include "matchers/single_op/single_op.hpp"
#include "matchers/single_op/convolutions.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class OpCache : public ICache {
public:
    void update_cache(const std::shared_ptr<ov::Model>& model,
                      const std::string& model_path, bool extract_body) override;
    void serialize_cache() override;

    static std::shared_ptr<OpCache> get() {
        if (m_cache_instance == nullptr) {
            m_cache_instance = std::shared_ptr<OpCache>(new OpCache);
        }
        return std::shared_ptr<OpCache>(m_cache_instance);
    }

    static void reset() {
        m_cache_instance.reset();
        m_cache_instance = nullptr;
    }

    void reset_cache() override {
        reset();
    };

protected:
    std::map<std::shared_ptr<ov::Node>, MetaInfo> m_ops_cache;
    static std::shared_ptr<OpCache> m_cache_instance;
    MatchersManager m_manager = MatchersManager();

    OpCache() {
        MatchersManager::MatchersMap matchers = {
            { "generic_single_op", SingleOpMatcher::Ptr(new SingleOpMatcher) },
            { "convolutions", ConvolutionsMatcher::Ptr(new ConvolutionsMatcher) },
        };
        m_manager.set_matchers(matchers);
    }

    void update_cache(const std::shared_ptr<ov::Node>& node, const std::string& model_path, size_t model_op_cnt = 1);
    bool serialize_op(const std::pair<std::shared_ptr<ov::Node>, MetaInfo>& op_info);
    std::string get_rel_serilization_dir(const std::shared_ptr<ov::Node>& node);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov