/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <algorithm>
#include <atomic>
#include <tuple>
#include <utility>
#include "driver.hpp"
#include "pass/pass.hpp"
#include <unordered_map>
#include <util/scoped_timer.hpp>

#ifdef _MSC_VER
#include <Windows.h>
#define getprocessid GetCurrentProcessId
#else
#include <unistd.h>
#define getprocessid getpid
#endif

namespace sc {

SC_MODULE(graph.driver)

basic_graph_pass_ptr create_graph_pass(const std::string &name,
        pass_func func_t, const std::vector<std::string> &requires,
        pass_type type, bool enabled) {
    return std::make_shared<basic_graph_pass_t>(
            func_t, name, requires, type, enabled);
}

static std::tuple<std::vector<basic_graph_pass_ptr>,
        std::vector<basic_graph_pass_ptr>>
create_default_graph_flow() {
    std::vector<basic_graph_pass_ptr> pre_tune_passes, post_tune_passes;
    pre_tune_passes.push_back(create_graph_pass("analysis_quantized",
            analysis_quantized, {}, pass_type::analysis, true));
    pre_tune_passes.push_back(create_graph_pass(
            "graph_inline", graph_inline, {}, pass_type::pre_tune, true));
    pre_tune_passes.push_back(create_graph_pass("constant_optimization",
            constant_optimization, {}, pass_type::pre_tune, true));
    pre_tune_passes.push_back(create_graph_pass("quantized_info_propagation",
            quantize::quantize_info_propagation, {}, pass_type::pre_tune,
            true));
    pre_tune_passes.push_back(create_graph_pass("quantized_graph_reschedule",
            quantize::graph_reschedule, {}, pass_type::pre_tune, true));
    pre_tune_passes.push_back(create_graph_pass("quantize_inline",
            quantize::quantize_inline, {}, pass_type::pre_tune, true));
    pre_tune_passes.push_back(create_graph_pass("elemtwise_bcast_swap",
            elemwise_bcast_swap, {}, pass_type::pre_tune, true));
    pre_tune_passes.push_back(create_graph_pass("permute_propagation",
            permute_propagation, {}, pass_type::pre_tune, true));

    // ------------------ post_tune -------------------------------------------
    post_tune_passes.push_back(create_graph_pass("quantize_op_compensation",
            quantize::calculate_op_compensation, {}, pass_type::post_tune,
            true));
    post_tune_passes.push_back(create_graph_pass("layout_propagation",
            layout_propagation, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("tensor_view_transform",
            tensor_view_transform, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass(
            "graph_simplify", graph_simplify, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("global_reschedule",
            global_reschedule, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("brgemm_fusion_transform",
            brgemm_fusion_transform, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("const_folding",
            graph_constant_input_folding, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass(
            "fuse_ops", fuse_ops, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("horizontal_merge",
            horizontal_merge, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("const_folding",
            graph_constant_input_folding, {}, pass_type::post_tune, true));
    post_tune_passes.push_back(create_graph_pass("inplace_transform",
            inplace_transform, {}, pass_type::post_tune, true));

    // get passes map
    std::unordered_map<std::string, basic_graph_pass_ptr> passes_map;
    std::transform(pre_tune_passes.begin(), pre_tune_passes.end(),
            std::inserter(passes_map, passes_map.end()),
            [](const basic_graph_pass_ptr &pass) {
                return std::make_pair(pass->name_, pass);
            });

    std::transform(post_tune_passes.begin(), post_tune_passes.end(),
            std::inserter(passes_map, passes_map.end()),
            [](const basic_graph_pass_ptr &pass) {
                return std::make_pair(pass->name_, pass);
            });
    // get pass's dependies and reset enabled_.
    for (auto &kv : passes_map) {
        if (kv.second->enabled_) {
            for (const std::string &require : kv.second->requires_) {
                passes_map[require]->enabled_ = true;
            }
        }
    }
    return std::make_tuple(pre_tune_passes, post_tune_passes);
}

const std::tuple<std::vector<basic_graph_pass_ptr>,
        std::vector<basic_graph_pass_ptr>> &
get_graph_passes() {
    static auto passes = create_default_graph_flow();
    return passes;
}

static void run_passes(sc_graph_t &graph, const context_ptr &ctx,
        const std::vector<basic_graph_pass_ptr> &passes) {
    bool need_time = utils::compiler_configs_t::get().print_pass_time_;
    bool need_result = utils::compiler_configs_t::get().print_pass_result_;
    for (auto &pass : passes) {
        if (pass->enabled_) {
            auto timer = utils::create_scoped_timer(
                    need_time, [&pass](utils::time_duration dur) {
                        std::string name = std::string("graph.driver.time.")
                                + pass->name_;
                        SC_MODULE_INFO2(name.c_str())
                                << "took "
                                << std::chrono::duration_cast<
                                           std::chrono::microseconds>(dur)
                                           .count()
                                << " us";
                    });
            pass->func_(graph, ctx);
            if (need_result) {
                std::string name
                        = std::string("graph.driver.debug.") + pass->name_;
                if (auto stream
                        = ::sc::utils::get_info_logging_stream(name.c_str())) {
                    *stream.stream_ << "IR after this pass:\n";
                    print_graph(graph, *stream.stream_, true, true);
                }
            }
        }
    }
}

void graph_driver(sc_graph_t &graph, const context_ptr &ctx,
        const graph_config *in_cfg, graph_config *out_cfg, int batch_size,
        int repeat, int64_t timeout, tuner_creator *tune_creator,
        std::vector<basic_graph_pass_ptr> *pre_tune_pass,
        std::vector<basic_graph_pass_ptr> *post_tune_pass) {
    auto &passes_tuple = get_graph_passes();

    const std::vector<basic_graph_pass_ptr> *prepass
            = pre_tune_pass ? pre_tune_pass : &std::get<0>(passes_tuple);
    const std::vector<basic_graph_pass_ptr> *postpass
            = post_tune_pass ? post_tune_pass : &std::get<1>(passes_tuple);
    // run pre_processing passes
    run_passes(graph, ctx, *prepass);

    // run post tune passes
    run_passes(graph, ctx, *postpass);
}

void graph_driver(
        sc_graph_t &graph, int batch_size, int repeat, const context_ptr &ctx) {
    graph_config *pincfg = nullptr;
    graph_config *poutcfg = nullptr;
    tuner_creator *ptun_creator = nullptr;
    int64_t real_timeout = 0;
    sc_graph_t orig_graph;
    if (poutcfg) { orig_graph = copy_graph(graph); }
    graph_driver(graph, ctx, pincfg, poutcfg, batch_size, repeat, real_timeout,
            ptun_creator);
}

} // namespace sc
