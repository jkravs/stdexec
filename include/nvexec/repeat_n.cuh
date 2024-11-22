#pragma once

#include "nvexec/stream/common.cuh"

namespace nvexec {
  namespace STDEXEC_STREAM_DETAIL_NS::repeat_n {
    using namespace stdexec;

    template <class OpT>
    class receiver_2_t : public stream_receiver_base {
      using Sender = typename OpT::PredSender;
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

     public:
      void set_value() noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        op_state_.i_++;

        if (op_state_.i_ == op_state_.n_) {
          op_state_.propagate_completion_signal(stdexec::set_value);
          return;
        }

        auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
        inner_op_state_t& inner_op_state =
          op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
            return stdexec::connect(stdexec::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
          }});

        stdexec::start(inner_op_state);
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        op_state_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> typename OpT::env_t {
        return op_state_.make_env();
      }

      explicit receiver_2_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class OpT>
    class receiver_1_t : public stream_receiver_base {
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

     public:
      void set_value() noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        if (op_state_.n_) {
          auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
          inner_op_state_t& inner_op_state =
            op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
              return stdexec::connect(
                stdexec::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
            }});

          stdexec::start(inner_op_state);
        } else {
          op_state_.propagate_completion_signal(set_value_t());
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        op_state_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> typename OpT::env_t {
        return op_state_.make_env();
      }

      explicit receiver_1_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class PredecessorSenderId, class Closure, class ReceiverId>
    struct operation_state_t : operation_state_base_t<ReceiverId> {
      using PredSender = stdexec::__t<PredecessorSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Scheduler =
        std::invoke_result_t<stdexec::get_scheduler_t, stdexec::env_of_t<Receiver>>;
      using InnerSender =
        std::invoke_result_t<Closure, stdexec::schedule_result_t<Scheduler>>;

      using predecessor_op_state_t =
        stdexec::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
      using inner_op_state_t = stdexec::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

      PredSender pred_sender_;
      Closure closure_;
      std::optional<predecessor_op_state_t> pred_op_state_;
      std::optional<inner_op_state_t> inner_op_state_;
      std::size_t n_{};
      std::size_t i_{};

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        if (op.stream_provider_.status_ != cudaSuccess) {
          // Couldn't allocate memory for operation state, complete with error
          op.propagate_completion_signal(
            stdexec::set_error, std::move(op.stream_provider_.status_));
        } else {
          if (op.n_) {
            stdexec::start(*op.pred_op_state_);
          } else {
            op.propagate_completion_signal(stdexec::set_value);
          }
        }
      }

      operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
        : operation_state_base_t<ReceiverId>(
          static_cast<Receiver&&>(rcvr),
          stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(pred_sender))
            .context_state_)
        , pred_sender_{static_cast<PredSender&&>(pred_sender)}
        , closure_(closure)
        , n_(n) {
        pred_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
          return stdexec::connect(static_cast<PredSender&&>(pred_sender_), receiver_1_t{*this});
        }});
      }
    };
  } // STDEXEC_STREAM_DETAIL_NS::repeat_n

  namespace repeat_n_detail {

    template <class OpT>
    class receiver_2_t {
      using Sender = typename OpT::PredSender;
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

     public:
      using receiver_concept = stdexec::receiver_t;

      void set_value() noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        op_state_.i_++;

        if (op_state_.i_ == op_state_.n_) {
          stdexec::set_value(std::move(op_state_.rcvr_));
          return;
        }

        auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
        inner_op_state_t& inner_op_state =
          op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
            return stdexec::connect(stdexec::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
          }});

        stdexec::start(inner_op_state);
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        stdexec::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(std::move(op_state_.rcvr_));
      }

      auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
        return stdexec::get_env(op_state_.rcvr_);
      }

      explicit receiver_2_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class OpT>
    class receiver_1_t {
      using Receiver = typename OpT::Receiver;

      OpT& op_state_;

     public:
      using receiver_concept = stdexec::receiver_t;

      void set_value() noexcept {
        using inner_op_state_t = typename OpT::inner_op_state_t;

        if (op_state_.n_) {
          auto sch = stdexec::get_scheduler(stdexec::get_env(op_state_.rcvr_));
          inner_op_state_t& inner_op_state =
            op_state_.inner_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
              return stdexec::connect(stdexec::schedule(sch) | op_state_.closure_, receiver_2_t<OpT>{op_state_});
            }});

          stdexec::start(inner_op_state);
        } else {
          stdexec::set_value(std::move(op_state_.rcvr_));
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept {
        stdexec::set_error(std::move(op_state_.rcvr_), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept {
        stdexec::set_stopped(std::move(op_state_.rcvr_));
      }

      auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
        return stdexec::get_env(op_state_.rcvr_);
      }

      explicit receiver_1_t(OpT& op_state)
        : op_state_(op_state) {
      }
    };

    template <class PredecessorSenderId, class Closure, class ReceiverId>
    struct operation_state_t {
      using PredSender = stdexec::__t<PredecessorSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using Scheduler =
        std::invoke_result_t<stdexec::get_scheduler_t, stdexec::env_of_t<Receiver>>;
      using InnerSender =
        std::invoke_result_t<Closure, stdexec::schedule_result_t<Scheduler>>;

      using predecessor_op_state_t =
        stdexec::connect_result_t<PredSender, receiver_1_t<operation_state_t>>;
      using inner_op_state_t = stdexec::connect_result_t<InnerSender, receiver_2_t<operation_state_t>>;

      PredSender pred_sender_;
      Closure closure_;
      Receiver rcvr_;
      std::optional<predecessor_op_state_t> pred_op_state_;
      std::optional<inner_op_state_t> inner_op_state_;
      std::size_t n_{};
      std::size_t i_{};

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        if (op.n_) {
          stdexec::start(*op.pred_op_state_);
        } else {
          stdexec::set_value(std::move(op.rcvr_));
        }
      }

      operation_state_t(PredSender&& pred_sender, Closure closure, Receiver&& rcvr, std::size_t n)
        : pred_sender_{static_cast<PredSender&&>(pred_sender)}
        , closure_(closure)
        , rcvr_(rcvr)
        , n_(n) {
        pred_op_state_.emplace(stdexec::__emplace_from{[&]() noexcept {
          return stdexec::connect(static_cast<PredSender&&>(pred_sender_), receiver_1_t{*this});
        }});
      }
    };

    template <class SenderId, class Closure>
    struct repeat_n_sender_t {
      using __t = repeat_n_sender_t;
      using __id = repeat_n_sender_t;
      using Sender = stdexec::__t<SenderId>;
      using sender_concept = stdexec::sender_t;

      using completion_signatures = //
        stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_stopped_t(),
          stdexec::set_error_t(std::exception_ptr),
          stdexec::set_error_t(cudaError_t)>;

      Sender sender_;
      Closure closure_;
      std::size_t n_{};

      template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
        requires(stdexec::sender_to<Sender, Receiver>)
             && (!nvexec::STDEXEC_STREAM_DETAIL_NS::receiver_with_stream_env<Receiver>)
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver r)
        -> repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>> {
        return repeat_n_detail::operation_state_t<SenderId, Closure, stdexec::__id<Receiver>>(
          static_cast<Sender&&>(self.sender_), self.closure_, static_cast<Receiver&&>(r), self.n_);
      }

      template <stdexec::__decays_to<repeat_n_sender_t> Self, stdexec::receiver Receiver>
        requires(stdexec::sender_to<Sender, Receiver>)
             && (nvexec::STDEXEC_STREAM_DETAIL_NS::receiver_with_stream_env<Receiver>)
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver r)
        -> nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n::
          operation_state_t<SenderId, Closure, stdexec::__id<Receiver>> {
        return nvexec::STDEXEC_STREAM_DETAIL_NS::repeat_n::
          operation_state_t<SenderId, Closure, stdexec::__id<Receiver>>(
            static_cast<Sender&&>(self.sender_), self.closure_, static_cast<Receiver&&>(r), self.n_);
      }

      auto get_env() const noexcept -> stdexec::env_of_t<const Sender&> {
        return stdexec::get_env(sender_);
      }
    };
  } // namespace repeat_n_detail

  struct repeat_n_t {
    template <stdexec::sender Sender, stdexec::__sender_adaptor_closure Closure>
    auto operator()(Sender&& __sndr, Closure closure, std::size_t n) const noexcept
      -> repeat_n_detail::repeat_n_sender_t<stdexec::__id<Sender>, Closure> {
      return repeat_n_detail::repeat_n_sender_t<stdexec::__id<Sender>, Closure>{
        std::forward<Sender>(__sndr), closure, n};
    }

    template <stdexec::__sender_adaptor_closure Closure>
    auto operator()(Closure closure, std::size_t n) const
      -> stdexec::__binder_back<repeat_n_t, Closure, std::size_t> {
      return {{static_cast<Closure&&>(closure), n}, {}, {}};
    }
  };

  inline constexpr repeat_n_t repeat_n{};
}

  /* vim: set shiftwidth=2 softtabstop=2 expandtab: */
