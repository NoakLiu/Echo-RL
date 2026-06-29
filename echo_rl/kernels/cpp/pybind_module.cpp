#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/echo_kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(_echo_kernels, m) {
    m.doc() = "EchoRL C++ performance kernels";

    py::class_<echo_rl::EMAPlanTracker>(m, "EMAPlanTracker")
        .def(py::init<int, double>(), py::arg("dim"), py::arg("decay") = 0.99)
        .def("update", [](echo_rl::EMAPlanTracker& self, py::array_t<float> plan) {
            auto buf = plan.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("plan must be 1-D");
            }
            self.update(static_cast<const float*>(buf.ptr), static_cast<int>(buf.size));
        })
        .def("get_ema", [](const echo_rl::EMAPlanTracker& self, int dim) {
            py::array_t<float> out(dim);
            auto buf = out.mutable_unchecked<1>();
            self.get_ema(buf.mutable_data(0), dim);
            return out;
        })
        .def_property_readonly("initialized", &echo_rl::EMAPlanTracker::initialized)
        .def_property_readonly("dim", &echo_rl::EMAPlanTracker::dim);

    m.def("compute_plan_surprise", [](py::array_t<float> plan,
                                      py::array_t<float> ema_plan,
                                      float reward,
                                      float surprise_weight,
                                      float reward_weight) {
        auto p = plan.request();
        auto e = ema_plan.request();
        if (p.ndim != 1 || e.ndim != 1 || p.size != e.size) {
            throw std::runtime_error("plan and ema_plan must be same-length 1-D arrays");
        }
        return echo_rl::compute_plan_surprise(
            static_cast<const float*>(p.ptr),
            static_cast<const float*>(e.ptr),
            static_cast<int>(p.size),
            reward,
            surprise_weight,
            reward_weight
        );
    });

    m.def("compute_plan_surprise_batch", [](py::array_t<float> plans,
                                            py::array_t<float> ema_plan,
                                            py::array_t<float> rewards,
                                            float surprise_weight,
                                            float reward_weight) {
        auto pl = plans.request();
        auto em = ema_plan.request();
        auto rw = rewards.request();
        if (pl.ndim != 2) {
            throw std::runtime_error("plans must be 2-D [batch, dim]");
        }
        const int batch = static_cast<int>(pl.shape[0]);
        const int dim = static_cast<int>(pl.shape[1]);
        if (static_cast<int>(rw.size) != batch) {
            throw std::runtime_error("rewards length must match batch size");
        }
        py::array_t<float> out(batch);
        auto out_buf = out.mutable_unchecked<1>();
        echo_rl::compute_plan_surprise_batch(
            static_cast<const float*>(pl.ptr),
            static_cast<const float*>(em.ptr),
            static_cast<const float*>(rw.ptr),
            batch,
            dim,
            surprise_weight,
            reward_weight,
            out_buf.mutable_data(0)
        );
        return out;
    });

    m.def("compute_attention_bandwidth_cost", &echo_rl::compute_attention_bandwidth_cost,
          py::arg("seq_len"), py::arg("scale") = 1.0f);

    m.def("compute_attention_bandwidth_cost_batch",
          [](py::array_t<int> seq_lengths, float scale) {
              auto buf = seq_lengths.request();
              const int batch = static_cast<int>(buf.size);
              py::array_t<float> out(batch);
              auto out_buf = out.mutable_unchecked<1>();
              echo_rl::compute_attention_bandwidth_cost_batch(
                  static_cast<const int*>(buf.ptr),
                  batch,
                  scale,
                  out_buf.mutable_data(0)
              );
              return out;
          });

    m.def("compute_schedule_priorities",
          [](py::array_t<float> rewards, py::array_t<float> queue_times, float epsilon) {
              auto r = rewards.request();
              auto q = queue_times.request();
              if (r.size != q.size) {
                  throw std::runtime_error("rewards and queue_times must have same length");
              }
              const int n = static_cast<int>(r.size);
              py::array_t<float> out(n);
              auto out_buf = out.mutable_unchecked<1>();
              echo_rl::compute_schedule_priorities(
                  static_cast<const float*>(r.ptr),
                  static_cast<const float*>(q.ptr),
                  n,
                  epsilon,
                  out_buf.mutable_data(0)
              );
              return out;
          });

    m.def("compute_effective_bandwidth_cost",
          &echo_rl::compute_effective_bandwidth_cost,
          py::arg("seq_len"),
          py::arg("reuse_len"),
          py::arg("scale") = 1.0f);

    m.def("compute_bandwidth_aware_priorities",
          [](py::array_t<float> rewards,
             py::array_t<float> bandwidth_costs,
             py::array_t<float> queue_times,
             float epsilon) {
              auto r = rewards.request();
              auto b = bandwidth_costs.request();
              auto q = queue_times.request();
              if (r.size != b.size || r.size != q.size) {
                  throw std::runtime_error(
                      "rewards, bandwidth_costs, and queue_times must have same length");
              }
              const int n = static_cast<int>(r.size);
              py::array_t<float> out(n);
              auto out_buf = out.mutable_unchecked<1>();
              echo_rl::compute_bandwidth_aware_priorities(
                  static_cast<const float*>(r.ptr),
                  static_cast<const float*>(b.ptr),
                  static_cast<const float*>(q.ptr),
                  n,
                  epsilon,
                  out_buf.mutable_data(0)
              );
              return out;
          });

    m.def("hash_state_vector", [](py::array_t<float> data) {
        auto buf = data.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("data must be 1-D");
        }
        return echo_rl::hash_state_vector(
            static_cast<const float*>(buf.ptr),
            static_cast<int>(buf.size)
        );
    });

    m.def("find_longest_prefix",
          [](py::array_t<uint64_t> sequence_hashes,
             py::list cache_entries,
             int min_prefix_len) {
              auto buf = sequence_hashes.request();
              std::vector<std::pair<uint64_t, int>> entries;
              entries.reserve(cache_entries.size());
              for (const auto& item : cache_entries) {
                  auto tup = item.cast<std::pair<uint64_t, int>>();
                  entries.emplace_back(tup);
              }
              const auto result = echo_rl::find_longest_prefix(
                  static_cast<const uint64_t*>(buf.ptr),
                  static_cast<int>(buf.size),
                  entries,
                  min_prefix_len
              );
              if (!result.has_value()) {
                  return py::make_tuple(-1, 0);
              }
              return py::make_tuple(
                  static_cast<int>(result->cache_index),
                  static_cast<int>(result->prefix_length)
              );
          });

    m.def("softmax_priority_sample",
          [](py::array_t<float> priorities,
             int sample_size,
             float temperature,
             float importance_beta,
             uint64_t seed) {
              auto buf = priorities.request();
              const int n = static_cast<int>(buf.size);
              const auto result = echo_rl::softmax_priority_sample(
                  static_cast<const float*>(buf.ptr),
                  n,
                  sample_size,
                  temperature,
                  importance_beta,
                  seed
              );
              py::array_t<int> indices(static_cast<py::ssize_t>(result.indices.size()));
              py::array_t<float> weights(static_cast<py::ssize_t>(result.importance_weights.size()));
              auto idx_buf = indices.mutable_unchecked<1>();
              auto w_buf = weights.mutable_unchecked<1>();
              for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(result.indices.size()); ++i) {
                  idx_buf(i) = result.indices[static_cast<size_t>(i)];
                  w_buf(i) = result.importance_weights[static_cast<size_t>(i)];
              }
              return py::make_tuple(indices, weights);
          });

    m.def("compute_bandwidth_efficiency", &echo_rl::compute_bandwidth_efficiency);

    m.def("compute_temporal_kl",
          [](py::array_t<float> current, py::array_t<float> previous, float sigma_squared) {
              auto c = current.request();
              auto p = previous.request();
              if (c.size != p.size) {
                  throw std::runtime_error("current and previous must have same length");
              }
              return echo_rl::compute_temporal_kl(
                  static_cast<const float*>(c.ptr),
                  static_cast<const float*>(p.ptr),
                  static_cast<int>(c.size),
                  sigma_squared
              );
          });

    m.def("compute_temporal_kl_batch",
          [](py::array_t<float> current_plans,
             py::array_t<float> previous_plans,
             float sigma_squared) {
              auto c = current_plans.request();
              auto p = previous_plans.request();
              if (c.ndim != 2 || p.ndim != 2) {
                  throw std::runtime_error("plans must be 2-D");
              }
              const int batch = static_cast<int>(c.shape[0]);
              const int dim = static_cast<int>(c.shape[1]);
              py::array_t<float> out(batch);
              auto out_buf = out.mutable_unchecked<1>();
              echo_rl::compute_temporal_kl_batch(
                  static_cast<const float*>(c.ptr),
                  static_cast<const float*>(p.ptr),
                  batch,
                  dim,
                  sigma_squared,
                  out_buf.mutable_data(0)
              );
              return out;
          });
}
