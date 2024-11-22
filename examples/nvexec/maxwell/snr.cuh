/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "common.cuh"
#include "stdexec/execution.hpp"
#include "exec/on.hpp"

#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
#include "nvexec/detail/throw_on_cuda_error.cuh"
#include <nvexec/stream_context.cuh>
#include <nvexec/multi_gpu_context.cuh>
#include <nvexec/repeat_n.cuh>
#else
#include <exec/repeat_n.hpp>

namespace nvexec {
    struct stream_receiver_base {
        using receiver_concept = stdexec::receiver_t;
    };

    struct stream_sender_base {
        using sender_concept = stdexec::sender_t;
    };

    namespace detail {
        struct stream_op_state_base { };
    } // namespace detail

    inline bool is_on_gpu() {
        return false;
    }
} // namespace nvexec

#endif

#include <optional>
#include <exec/inline_scheduler.hpp>
#include <exec/static_thread_pool.hpp>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace ex = stdexec;

template <class SchedulerT>
[[nodiscard]]
bool is_gpu_scheduler(SchedulerT&& scheduler) {
  auto snd = ex::just() | exec::on(scheduler, ex::then([] { return nvexec::is_on_gpu(); }));
  auto [on_gpu] = stdexec::sync_wait(std::move(snd)).value();
  return on_gpu;
}

auto maxwell_eqs_snr(
  float dt,
  float* time,
  bool write_results,
  std::size_t n_iterations,
  fields_accessor accessor,
  stdexec::scheduler auto&& computer) {
  return ex::just()
       | exec::on(
           computer,
#if defined(_NVHPC_CUDA) || defined(__CUDACC__)
           nvexec::repeat_n(
#else
           exec::repeat_n(
#endif
             ex::bulk(accessor.cells, update_h(accessor))
               | ex::bulk(accessor.cells, update_e(time, dt, accessor)), n_iterations))
       | ex::then(dump_vtk(write_results, accessor));
}

void run_snr(
  float dt,
  bool write_vtk,
  std::size_t n_iterations,
  grid_t& grid,
  std::string_view scheduler_name,
  stdexec::scheduler auto&& computer) {
  time_storage_t time{is_gpu_scheduler(computer)};
  fields_accessor accessor = grid.accessor();

  auto init = ex::just() | exec::on(computer, ex::bulk(grid.cells, grid_initializer(dt, accessor)));
  stdexec::sync_wait(init);

  auto snd = maxwell_eqs_snr(dt, time.get(), write_vtk, n_iterations, accessor, computer);

  report_performance(grid.cells, n_iterations, scheduler_name, [&snd] {
    stdexec::sync_wait(std::move(snd));
  });
}

STDEXEC_PRAGMA_POP()
