// flow.cpp
#include "flow.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void FlowConfig::validate() const {
    const double probs[3] = {p_limit, p_market, p_cancel};
    for (double p : probs) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("Event probabilities must be in [0,1]");
        }
    }
    double sum = p_limit + p_market + p_cancel;
    if (std::abs(sum - 1.0) > 1e-9) {
        throw std::invalid_argument("Event probabilities must sum to 1.0");
    }
}

PoissonOrderFlow::PoissonOrderFlow(const FlowConfig& config, int64_t seed)
    : config_(config),
      rng_(static_cast<uint32_t>(seed)),
      next_order_id_(1),
      ref_mid_(config.initial_mid) {
    config_.validate();
}

int64_t PoissonOrderFlow::mid_tick(const LimitOrderBook& book) const {
    auto m = book.mid();
    if (!m.has_value()) {
        return ref_mid_;
    }
    return static_cast<int64_t>(std::llround(*m));
}

int64_t PoissonOrderFlow::new_order_id() {
    int64_t oid = next_order_id_;
    next_order_id_ += 1;
    return oid;
}

EventKind PoissonOrderFlow::sample_event_kind(double p_limit, double p_market, double p_cancel) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    if (r < p_limit) {
        return EventKind::Limit;
    }
    if (r < p_limit + p_market) {
        return EventKind::Market;
    }
    return EventKind::Cancel;
}

int64_t PoissonOrderFlow::sample_offset() {
    int64_t lo = config_.min_offset;
    int64_t hi = config_.max_offset;
    if (hi <= lo) {
        return lo;
    }
    std::vector<double> weights;
    weights.reserve(static_cast<size_t>(hi - lo + 1));
    for (int64_t k = lo; k <= hi; ++k) {
        weights.push_back(1.0 / std::pow(static_cast<double>(k), config_.offset_power));
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int idx = dist(rng_);
    return lo + idx;
}

std::pair<EventKind, int64_t> PoissonOrderFlow::step(LimitOrderBook& book, int64_t ts) {
    int64_t n_orders = static_cast<int64_t>(book.order_count());
    int64_t soft = config_.soft_max_orders;
    int64_t hard = config_.hard_max_orders;
    if (hard <= soft) {
        hard = soft + 1;
    }

    double pressure = 0.0;
    if (n_orders > soft) {
        pressure = std::min(1.0, static_cast<double>(n_orders - soft) / static_cast<double>(hard - soft));
    }

    double p_market = config_.p_market;
    double p_cancel = std::min(0.99, config_.p_cancel + config_.cancel_boost * pressure);
    double p_limit = std::max(0.0, 1.0 - p_market - p_cancel);

    EventKind kind = sample_event_kind(p_limit, p_market, p_cancel);

    // Keep the book alive: force LIMIT if empty or touch liquidity is too low.
    if (!book.best_bid().has_value() || !book.best_ask().has_value()) {
        kind = EventKind::Limit;
    } else {
        auto bid_touch = book.depth(Side::Bid, 1);
        auto ask_touch = book.depth(Side::Ask, 1);
        int64_t bid_q = bid_touch.empty() ? 0 : bid_touch[0].second;
        int64_t ask_q = ask_touch.empty() ? 0 : ask_touch[0].second;
        if (bid_q < config_.min_touch_qty || ask_q < config_.min_touch_qty) {
            kind = EventKind::Limit;
        }
    }

    if (kind == EventKind::Cancel && book.order_count() == 0) {
        if (config_.cancel_fallback_to_limit) {
            kind = EventKind::Limit;
        } else {
            return {kind, 0};
        }
    }

    if (kind == EventKind::Limit) {
        int64_t mid = mid_tick(book);
        Side side = Side::Bid;
        if (!book.best_bid().has_value() && !book.best_ask().has_value()) {
            side = ((ts / config_.dt_us) % 2 == 0) ? Side::Bid : Side::Ask;
        } else if (!book.best_bid().has_value()) {
            side = Side::Bid;
        } else if (!book.best_ask().has_value()) {
            side = Side::Ask;
        } else {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            side = (dist(rng_) < 0.5) ? Side::Bid : Side::Ask;
        }

        int64_t offset = sample_offset();
        std::uniform_int_distribution<int64_t> qty_dist(config_.min_qty, config_.max_qty);
        int64_t qty = qty_dist(rng_);

        int64_t px = (side == Side::Bid) ? (mid - offset) : (mid + offset);
        int64_t oid = new_order_id();

        auto fills = book.add_limit(Order{oid, side, px, qty, ts});
        int64_t traded = 0;
        for (const auto& f : fills) {
            traded += f.qty;
        }

        return {kind, traded};
    }

    if (kind == EventKind::Market) {
        if (!book.best_bid().has_value() || !book.best_ask().has_value()) {
            kind = EventKind::Limit;
        }
        if (kind == EventKind::Limit) {
            int64_t mid = mid_tick(book);
            Side side = Side::Bid;
            if (!book.best_bid().has_value() && !book.best_ask().has_value()) {
                side = ((ts / config_.dt_us) % 2 == 0) ? Side::Bid : Side::Ask;
            } else if (!book.best_bid().has_value()) {
                side = Side::Bid;
            } else if (!book.best_ask().has_value()) {
                side = Side::Ask;
            } else {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                side = (dist(rng_) < 0.5) ? Side::Bid : Side::Ask;
            }

            int64_t offset = sample_offset();
            std::uniform_int_distribution<int64_t> qty_dist(config_.min_qty, config_.max_qty);
            int64_t qty = qty_dist(rng_);
            int64_t px = (side == Side::Bid) ? (mid - offset) : (mid + offset);
            int64_t oid = new_order_id();

            auto fills = book.add_limit(Order{oid, side, px, qty, ts});
            int64_t traded = 0;
            for (const auto& f : fills) {
                traded += f.qty;
            }
            return {EventKind::Limit, traded};
        }

        double im = book.imbalance(1);
        Side side = Side::Bid;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (im > 0.2) {
            side = (dist(rng_) < 0.75) ? Side::Ask : Side::Bid;
        } else if (im < -0.2) {
            side = (dist(rng_) < 0.75) ? Side::Bid : Side::Ask;
        } else {
            side = (dist(rng_) < 0.5) ? Side::Bid : Side::Ask;
        }

        std::uniform_int_distribution<int64_t> qty_dist(config_.min_market_qty, config_.max_market_qty);
        int64_t qty = qty_dist(rng_);

        auto fills = book.add_market(side, qty, ts);
        int64_t traded = 0;
        for (const auto& f : fills) {
            traded += f.qty;
        }
        return {kind, traded};
    }

    int64_t cancel_levels = config_.cancel_levels;
    if (book.order_count() == 0) {
        return {EventKind::Limit, 0};
    }

    double im = book.imbalance(1);
    Side side = Side::Bid;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (im > 0.2) {
        side = (dist(rng_) < 0.75) ? Side::Bid : Side::Ask;
    } else if (im < -0.2) {
        side = (dist(rng_) < 0.75) ? Side::Ask : Side::Bid;
    } else {
        side = (dist(rng_) < 0.5) ? Side::Bid : Side::Ask;
    }

    auto px_levels = book.depth(side, static_cast<int>(cancel_levels));
    if (px_levels.empty()) {
        side = (side == Side::Bid) ? Side::Ask : Side::Bid;
        px_levels = book.depth(side, static_cast<int>(cancel_levels));
    }

    if (!px_levels.empty()) {
        std::vector<double> weights;
        weights.reserve(px_levels.size());
        for (const auto& level : px_levels) {
            weights.push_back(static_cast<double>(std::max<int64_t>(1, level.second)));
        }
        std::discrete_distribution<int> price_dist(weights.begin(), weights.end());
        int idx = price_dist(rng_);
        int64_t px = px_levels[static_cast<size_t>(idx)].first;

        auto ids = book.order_ids_at_price(side, px);
        if (!ids.empty()) {
            std::uniform_int_distribution<size_t> id_dist(0, ids.size() - 1);
            int64_t oid = ids[id_dist(rng_)];
            book.cancel(oid, ts);
            return {kind, 0};
        }
    }

    auto ids = book.all_order_ids();
    if (!ids.empty()) {
        std::uniform_int_distribution<size_t> id_dist(0, ids.size() - 1);
        int64_t oid = ids[id_dist(rng_)];
        book.cancel(oid, ts);
    }

    return {kind, 0};
}
