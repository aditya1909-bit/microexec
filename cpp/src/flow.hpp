// flow.hpp
#pragma once

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "book.hpp"

enum class EventKind { Limit, Market, Cancel };
enum class HistoricalFormat { Binary, Csv, TrueFX };

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
    double p_ioc = 0.0;
    double p_fok = 0.0;

    void validate() const;
};

class PoissonOrderFlow {
public:
    explicit PoissonOrderFlow(const FlowConfig& config, int64_t seed = 0);

    EventKind sample_event_kind(double p_limit, double p_market, double p_cancel);
    int64_t sample_offset();
    std::pair<EventKind, int64_t> step(LimitOrderBook& book, int64_t ts);
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> step_n(
        LimitOrderBook& book,
        int64_t start_ts,
        int64_t n_events
    );

private:
    int64_t mid_tick(const LimitOrderBook& book) const;
    int64_t new_order_id();

    FlowConfig config_;
    std::mt19937 rng_;
    int64_t next_order_id_ = 1;
    int64_t ref_mid_ = 10000;

    // Cached RNG distributions for speed (avoid re-allocating each step)
    std::uniform_real_distribution<double> uni01_{0.0, 1.0};

    // Cached offset distribution (built once from config_)
    int64_t offset_lo_ = 1;
    int64_t offset_hi_ = 1;
    std::optional<std::discrete_distribution<int>> offset_dist_;
};

struct HawkesConfig {
    FlowConfig base;
    double alpha_limit = 0.15;
    double alpha_market = 0.15;
    double alpha_cancel = 0.10;
    double decay = 10.0;
};

class HawkesOrderFlow {
public:
    explicit HawkesOrderFlow(const HawkesConfig& config, int64_t seed = 0);

    std::pair<EventKind, int64_t> step(LimitOrderBook& book, int64_t ts);

private:
    OrderType sample_limit_type();
    EventKind sample_kind(double lambda_limit, double lambda_market, double lambda_cancel);
    int64_t sample_offset();
    int64_t mid_tick(const LimitOrderBook& book) const;
    int64_t new_order_id();

    HawkesConfig config_;
    std::mt19937 rng_;
    int64_t next_order_id_ = 1;
    int64_t ref_mid_ = 10000;
    double lambda_limit_;
    double lambda_market_;
    double lambda_cancel_;
    std::uniform_real_distribution<double> uni01_{0.0, 1.0};
};

class HistoricalOrderFlow {
public:
    HistoricalOrderFlow(
        const std::string& path,
        HistoricalFormat format,
        int64_t start_order_id = 1,
        int64_t fixed_qty = 1'000'000,
        int64_t price_scale = 100'000
    );
    ~HistoricalOrderFlow();

    bool step(LimitOrderBook& book, int64_t& ts_out, int64_t& traded_out);
    bool done() const;

private:
    bool read_next_tick(int64_t& ts, Side& side, int64_t& px, int64_t& qty);

    HistoricalFormat format_;
    int64_t next_order_id_;
    bool done_ = false;
    int64_t fixed_qty_ = 1'000'000;
    int64_t price_scale_ = 100'000;
    int64_t lp_bid_id_ = 0;
    int64_t lp_ask_id_ = 0;

    int fd_ = -1;
    const uint8_t* map_ = nullptr;
    size_t map_size_ = 0;
    size_t map_index_ = 0;

    std::ifstream csv_;
};
