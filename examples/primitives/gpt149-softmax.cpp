/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

/// @example softmax.cpp
/// > Annotated version: @ref softmax_example_cpp
///
/// @page softmax_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Softmax](@ref dev_guide_softmax) primitive in forward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Softmax along axis 1 (C) for 2D tensors.
///
/// @page softmax_example_cpp Softmax Primitive Example
/// @copydetails softmax_example_cpp_short
///
/// @include softmax.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "cs149.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void softmax_example(dnnl::engine::kind engine_kind) {

  onednn_engine engine(engine_kind);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 1000; // channels

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims dims = {N, IC};

    // Allocate buffer.
    std::vector<float> src_data(product(dims));

    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Softmax axis.
    const int axis = 1;

    engine.softmax(dims, axis, src_data.data());
    engine.softmax(dims, axis, src_data.data());

    engine.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(
            softmax_example, parse_engine_kind(argc, argv));
}
