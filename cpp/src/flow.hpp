// flow.hpp
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "book.hpp"

enum class EventKind { Limit, Market, Cancel };

struct FlowConfig {
    int64_t initial_mid = 10000;
    int64_t dt_us = 1000;

    double p_limit = 0.40;
    double p_market = 0.35;
    double p_cancel = 0.25;

    int64_t cancel_levels = 5;
    int64_t soft_max_orders = 20000;
    int64_t hard_max_orders = 50000;
    double cancel_boost = 0.50;

    int64_t min_offset = 1;
    int64_t max_offset = 5;
    double offset_power = 2.0;

    int64_t min_qty = 1;
    int64_t max_qty = 3;

    int64_t min_market_qty = 1;
    int64_t max_market_qty = 12;

    int64_t min_touch_qty = 5;

    bool cancel_fallback_to_limit = true;

    void validate() const;
};

class PoissonOrderFlow {
public:
    explicit PoissonOrderFlow(const FlowConfig& config, int64_t seed = 0);

    EventKind sample_event_kind(double p_limit, double p_market, double p_cancel);
    int64_t sample_offset();
    std::pair<EventKind, int64_t> step(LimitOrderBook& book, int64_t ts);

private:
    int64_t mid_tick(const LimitOrderBook& book) const;
    int64_t new_order_id();

    FlowConfig config_;
    std::mt19937 rng_;
    int64_t next_order_id_ = 1;
    int64_t ref_mid_ = 10000;
};
