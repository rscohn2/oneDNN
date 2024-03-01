#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

struct onednn_engine {
  onednn_engine(dnnl::engine::kind engine_kind) :
    engine_(engine_kind, 0),
    stream_(engine_)
  {}

  void softmax(const dnnl::memory::dims &dims, const int axis, float *data) {
    auto src_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
    auto dst_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
    pending_mems_.emplace_back(src_md, engine_, data);
    auto &src_mem = pending_mems_.back();

    // This can be expensive, but dnn will cache it
    auto softmax_pd =
      dnnl::softmax_forward::primitive_desc(engine_, dnnl::prop_kind::forward_training,
                                            dnnl::algorithm::softmax_accurate, src_md,
                                            dst_md, axis);

    auto op = dnnl::softmax_forward(softmax_pd);

    // Set up in-place execution by assigning src as DST.
    std::unordered_map<int, dnnl::memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});

    op.execute(stream_, softmax_args);
  }

  void wait() {
    stream_.wait();
    pending_mems_.clear();
  }
  
  std::vector<dnnl::memory> pending_mems_;
  dnnl::engine engine_;
  dnnl::stream stream_;
};
