// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ngraph_test_utils.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pass/manager.hpp>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ::testing;
using namespace std;
using namespace ngraph;

class TestPass : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TestPass() : MatcherPass() {
        auto divide =
            std::make_shared<ngraph::pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset3::Divide>());
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto relu = std::make_shared<ngraph::opset3::Relu>(m.get_match_root()->input_value(0));
                ngraph::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

class GatherNodesPass : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherNodesPass(NodeVector& order) : MatcherPass() {
        ngraph::matcher_pass_callback callback = [&order](pattern::Matcher& m) {
            order.push_back(m.get_match_root());
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::any_input(), "GatherNodesPass");
        this->register_matcher(m, callback);
    }
};

class Anchor : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    Anchor() : GraphRewrite() {}
};

NGRAPH_RTTI_DEFINITION(TestPass, "TestPass");
NGRAPH_RTTI_DEFINITION(Anchor, "Anchor");
NGRAPH_RTTI_DEFINITION(GatherNodesPass, "GatherNodesPass");

std::shared_ptr<Function> get_function() {
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
    auto divide_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    auto divide = std::make_shared<ngraph::opset3::Divide>(data, divide_constant);
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
}

ngraph::pass::param_callback get_callback() {
    return [](const std::shared_ptr<const Node>& node) -> bool {
        if (std::dynamic_pointer_cast<const opset3::Divide>(node)) {
            return true;
        } else {
            return false;
        }
    };
}

TEST(GraphRewriteOrderTest, MatcherPass) {
    auto f = get_function();

    NodeVector order;
    ngraph::pass::Manager m;
    auto pass = m.register_pass<pass::GraphRewrite>();
    pass->add_matcher<GatherNodesPass>(order);
    m.run_passes(f);

    ASSERT_EQ(order, f->get_ordered_ops());
}

TEST(BackwardGraphRewriteOrderTest, MatcherPass) {
    auto f = get_function();

    NodeVector order;
    ngraph::pass::Manager m;
    auto pass = m.register_pass<pass::BackwardGraphRewrite>();
    pass->add_matcher<GatherNodesPass>(order);
    m.run_passes(f);

    auto ref_order = f->get_ordered_ops();
    std::reverse(ref_order.begin(), ref_order.end());
    ASSERT_EQ(order, ref_order);
}

TEST(GraphRewriteTest, MatcherPassCallback) {
    auto f = get_function();

    Anchor anchor;
    anchor.add_matcher<TestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, GraphRewriteCallback) {
    auto f = get_function();

    Anchor anchor;
    anchor.add_matcher<TestPass>();
    anchor.set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallbackDeprecated) {
    auto f = get_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<Anchor>();
    anchor->add_matcher<TestPass>();
    manager.get_pass_config()->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback) {
    auto f = get_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<Anchor>();
    anchor->add_matcher<TestPass>();
    auto pass_config = manager.get_pass_config();
    pass_config->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, ManagerCallback2) {
    auto f = get_function();

    pass::Manager manager;
    auto anchor = manager.register_pass<TestPass>();
    manager.get_pass_config()->set_callback(get_callback());
    manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

class PrivateDivide : public ngraph::opset3::Divide {
public:
    NGRAPH_RTTI_DECLARATION;
    using ngraph::opset3::Divide::Divide;
};

NGRAPH_RTTI_DEFINITION(PrivateDivide, "PrivateDivide", ngraph::opset3::Divide);

std::shared_ptr<Function> get_derived_function() {
    auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
    auto divide_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
    auto divide = std::make_shared<PrivateDivide>(data, divide_constant);
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
}

TEST(GraphRewriteTest, MatcherPassCallbackDerived) {
    auto f = get_derived_function();

    Anchor anchor;
    anchor.add_matcher<TestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

class TypeBasedTestPass : public ngraph::pass::MatcherPass {
public:
    TypeBasedTestPass() : MatcherPass() {
        auto divide = std::make_shared<ngraph::opset3::Divide>(std::make_shared<ngraph::pattern::op::Label>(),
                                                               std::make_shared<ngraph::pattern::op::Label>());
        //        element::f32, Shape{}, pattern::has_class<opset3::Divide>());
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto relu = std::make_shared<ngraph::opset3::Relu>(m.get_match_root()->input_value(0));
                ngraph::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

class TypeBasedTestPassDerived : public ngraph::pass::MatcherPass {
public:
    TypeBasedTestPassDerived() : MatcherPass() {
        auto divide = std::make_shared<PrivateDivide>(std::make_shared<ngraph::pattern::op::Label>(),
                                                      std::make_shared<ngraph::pattern::op::Label>());
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto tanh = std::make_shared<ngraph::opset3::Tanh>(m.get_match_root()->input_value(0));
                ngraph::replace_node(m.get_match_root(), tanh);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "TestMatcher");
        this->register_matcher(m, callback);
    }
};

TEST(GraphRewriteTest, TypeBasedMatcherPassCallback) {
    auto f = get_function();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassCallbackDerived) {
    auto f = get_derived_function();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassOrder1) {
    auto f = get_derived_function();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.add_matcher<TypeBasedTestPassDerived>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
}

TEST(GraphRewriteTest, TypeBasedMatcherPassOrder2) {
    auto f = get_derived_function();

    Anchor anchor;
    anchor.add_matcher<TypeBasedTestPassDerived>()->set_callback(get_callback());
    anchor.add_matcher<TypeBasedTestPass>()->set_callback(get_callback());
    anchor.run_on_model(f);

    ASSERT_EQ(count_ops_of_type<opset3::Tanh>(f), 1);
}

TEST(PassConfigTest, Test1) {
    {
        auto f = get_function();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto f = get_function();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto f = get_function();

        auto pass_config = std::make_shared<ngraph::pass::PassConfig>();
        pass::Manager manager(pass_config);

        manager.register_pass<TestPass>();

        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto f = get_function();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = anchor->get_pass_config();
        pass_config->set_callback(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto f = get_function();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = anchor->get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto pass_config = std::make_shared<pass::PassConfig>();

        pass::Manager manager1(pass_config);
        pass::Manager manager2(pass_config);
        ASSERT_EQ(pass_config.use_count(), 3);
    }

    {
        auto f = get_function();

        pass::Manager manager;
        manager.register_pass<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        pass_config->disable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 0);

        pass_config->enable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }

    {
        auto f = get_function();

        pass::Manager manager;
        auto anchor = manager.register_pass<Anchor>();
        anchor->add_matcher<TestPass>();

        auto pass_config = manager.get_pass_config();
        pass_config->set_callback<TestPass>(get_callback());

        pass_config->disable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 0);

        pass_config->enable<TestPass>();
        manager.run_passes(f);
        ASSERT_EQ(count_ops_of_type<opset3::Relu>(f), 1);
    }
}

class CheckConsumers : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    CheckConsumers() {
        ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) -> bool {
            auto node = m.get_match_root();
            auto consumers = [](Node* node) {
                int64_t cnt{0};
                for (auto output : node->outputs()) {
                    cnt += output.get_target_inputs().size();
                }
                if (ov::as_type<op::Parameter>(node) || ov::as_type<op::Result>(node)) {
                    cnt += 1;
                }
                return cnt;
            };
            /* The expected number of use_count() for Node is equal to the sum of next components:
             * 1. Each consumer holds a pointer to Output<Node> which holds a shared_ptr to Node
             * 2. pattern::Matcher object holds a shared_ptr to the matched node
             * 3. Local node variable increases use_counter
             * 4. Some GraphRewrite facilities
             */
            auto cnt = consumers(node.get());
            if (node.use_count() != cnt + 7) {
                OPENVINO_THROW("Wrong number of consumers");
            }

            NodeVector nodes;
            for (const auto& inputs : node->input_values()) {
                nodes.emplace_back(inputs.get_node_shared_ptr());
            }

            /* The expected number of use_count() for Node is equal to the sum of next components:
             * 1. Each consumer holds a pointer to Output<Node> which holds a shared_ptr to Node
             * 2. Local input_node variable increases use_counter
             */
            for (const auto& input_node : nodes) {
                if (input_node.use_count() != consumers(input_node.get()) + 1) {
                    OPENVINO_THROW("Wrong number of consumers");
                }
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::any_input(), "CheckConsumers");
        this->register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(CheckConsumers, "CheckConsumers");

TEST(GraphRewriteTest, nodes_use_count) {
    auto f = get_function();
    pass::Manager m;
    m.register_pass<CheckConsumers>();
    ASSERT_NO_THROW(m.run_passes(f));
}
