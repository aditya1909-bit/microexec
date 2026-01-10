// execution.hpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "book.hpp"
#include "flow.hpp"

struct TwapReport {
    std::string side;
    int64_t target_qty = 0;
    int64_t filled_qty = 0;
    std::optional<double> avg_fill_px;
    std::optional<double> arrival_mid;
    std::optional<double> shortfall;
    int64_t n_child_orders = 0;
    int64_t unfilled_qty = 0;
    double completion_rate = 0.0;
    std::optional<double> penalty_per_share;
    std::optional<double> penalized_cost;
    std::optional<double> shortfall_per_share;
    std::optional<double> penalized_cost_per_share;
};

struct VwapReport {
    std::string side;
    int64_t target_qty = 0;
    int64_t filled_qty = 0;
    std::optional<double> avg_fill_px;
    std::optional<double> arrival_mid;
    std::optional<double> shortfall;
    int64_t n_child_orders = 0;
    int64_t unfilled_qty = 0;
    double completion_rate = 0.0;
    std::optional<double> penalty_per_share;
    std::optional<double> penalized_cost;
    std::optional<double> shortfall_per_share;
    std::optional<double> penalized_cost_per_share;

    int64_t n_buckets = 0;
    int64_t bucket_interval = 0;
    int64_t forecast_total_mkt_vol = 0;
};

struct ImpactModel {
    double temporary_impact = 0.0;  // ticks per share per second of rate
    double permanent_impact = 0.0;  // ticks per share of cumulative executed
    double volatility = 0.0;        // ticks per sqrt(second)
    double risk_aversion = 0.0;     // lambda
};

struct AlmgrenChrissReport {
    std::string side;
    int64_t target_qty = 0;
    int64_t filled_qty = 0;
    std::optional<double> avg_fill_px;
    std::optional<double> arrival_mid;
    std::optional<double> shortfall;
    int64_t n_child_orders = 0;
    int64_t unfilled_qty = 0;
    double completion_rate = 0.0;
    std::optional<double> risk_penalty;
    std::optional<double> objective;
    std::optional<double> shortfall_uncapped;
    std::optional<double> objective_uncapped;
    int64_t cap_abs = 0;
    double cap_frac = 0.0;
    int64_t cap_used = 0;
    bool capped = false;
    std::vector<int64_t> child_qtys;
    ImpactModel impact;
};

std::pair<LimitOrderBook, TwapReport> run_twap(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t child_interval,
    const FlowConfig& cfg,
    int64_t latency_us = 0,
    int64_t seed = 0,
    int64_t warmup_events = 500,
    double penalty_per_share = 0.0
);

std::pair<LimitOrderBook, VwapReport> run_vwap(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t bucket_interval,
    const FlowConfig& cfg,
    int64_t latency_us = 0,
    int64_t seed = 0,
    int64_t warmup_events = 500,
    double penalty_per_share = 0.0
);

std::pair<LimitOrderBook, AlmgrenChrissReport> run_almgren_chriss(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t child_interval,
    const FlowConfig& cfg,
    const ImpactModel& impact,
    int64_t cap_abs = 0,
    double cap_frac = 0.0,
    int64_t latency_us = 0,
    int64_t seed = 0,
    int64_t warmup_events = 500
);
