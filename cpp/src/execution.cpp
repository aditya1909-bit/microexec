// execution.cpp
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

double shortfall(const std::string& side, double arrival_mid, double avg_px, int64_t filled_qty) {
    if (side == "BID") {
        return (avg_px - arrival_mid) * static_cast<double>(filled_qty);
    }
    return (arrival_mid - avg_px) * static_cast<double>(filled_qty);
}

std::vector<int64_t> compute_bucket_targets(int64_t total_qty, const std::vector<int64_t>& vols) {
    const int64_t n = static_cast<int64_t>(vols.size());
    if (n == 0) {
        return {};
    }
    if (total_qty <= 0) {
        return std::vector<int64_t>(n, 0);
    }

    int64_t total_vol = 0;
    for (int64_t v : vols) {
        total_vol += v;
    }

    if (total_vol <= 0) {
        int64_t base = total_qty / n;
        int64_t rem = total_qty - base * n;
        std::vector<int64_t> out(n, base);
        for (int64_t i = 0; i < rem; ++i) {
            out[static_cast<size_t>(i)] += 1;
        }
        return out;
    }

    std::vector<double> raw;
    raw.reserve(static_cast<size_t>(n));
    for (int64_t v : vols) {
        raw.push_back(static_cast<double>(total_qty) * (static_cast<double>(v) / static_cast<double>(total_vol)));
    }

    std::vector<int64_t> floors;
    floors.reserve(static_cast<size_t>(n));
    int64_t rem = total_qty;
    for (double x : raw) {
        int64_t f = static_cast<int64_t>(x);
        floors.push_back(f);
        rem -= f;
    }

    std::vector<std::pair<double, int64_t>> fracs;
    fracs.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        fracs.emplace_back(raw[static_cast<size_t>(i)] - static_cast<double>(floors[static_cast<size_t>(i)]), i);
    }
    std::sort(fracs.begin(), fracs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::vector<int64_t> out = floors;
    for (int64_t k = 0; k < rem; ++k) {
        out[static_cast<size_t>(fracs[static_cast<size_t>(k)].second)] += 1;
    }

    int64_t s = 0;
    for (int64_t v : out) {
        s += v;
    }
    if (s != total_qty) {
        out.back() += (total_qty - s);
    }

    return out;
}

std::vector<int64_t> forecast_market_volume(
    int64_t horizon_events,
    int64_t bucket_interval,
    const FlowConfig& cfg,
    int64_t seed,
    int64_t warmup_events
) {
    LimitOrderBook book;
    PoissonOrderFlow flow(cfg, seed);

    int64_t ts = 0;
    for (int64_t i = 0; i < warmup_events; ++i) {
        ts += cfg.dt_us;
        flow.step(book, ts);
    }

    int64_t n_buckets = std::max<int64_t>(1, (horizon_events + bucket_interval - 1) / bucket_interval);
    std::vector<int64_t> vols(static_cast<size_t>(n_buckets), 0);

    for (int64_t i = 0; i < horizon_events; ++i) {
        ts += cfg.dt_us;
        auto [kind, traded] = flow.step(book, ts);
        if (kind == EventKind::Market) {
            vols[static_cast<size_t>(i / bucket_interval)] += traded;
        }
    }

    return vols;
}

Side parse_side(const std::string& side) {
    if (side == "BID") {
        return Side::Bid;
    }
    if (side == "ASK") {
        return Side::Ask;
    }
    throw std::invalid_argument("side must be 'BID' or 'ASK'");
}

}  // namespace

std::pair<LimitOrderBook, TwapReport> run_twap(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t child_interval,
    const FlowConfig& cfg,
    int64_t seed,
    int64_t warmup_events,
    double penalty_per_share
) {
    if (side != "BID" && side != "ASK") {
        throw std::invalid_argument("side must be 'BID' or 'ASK'");
    }
    if (total_qty <= 0) {
        throw std::invalid_argument("total_qty must be > 0");
    }
    if (horizon_events <= 0) {
        throw std::invalid_argument("horizon_events must be > 0");
    }
    if (child_interval <= 0) {
        throw std::invalid_argument("child_interval must be > 0");
    }
    if (penalty_per_share < 0) {
        throw std::invalid_argument("penalty_per_share must be >= 0");
    }

    LimitOrderBook book;
    PoissonOrderFlow flow(cfg, seed);

    int64_t ts = 0;
    for (int64_t i = 0; i < warmup_events; ++i) {
        ts += cfg.dt_us;
        flow.step(book, ts);
    }

    auto arrival_mid = book.mid();

    int64_t remaining = total_qty;
    int64_t filled_qty = 0;
    int64_t notional = 0;
    int64_t n_child = 0;

    int64_t n_slices = std::max<int64_t>(1, (horizon_events + child_interval - 1) / child_interval);
    Side side_enum = parse_side(side);

    for (int64_t i = 0; i < horizon_events; ++i) {
        ts += cfg.dt_us;

        flow.step(book, ts);

        if ((i % child_interval) == 0 && remaining > 0) {
            int64_t slices_left = std::max<int64_t>(1, n_slices - (i / child_interval));
            int64_t child_qty = (remaining + slices_left - 1) / slices_left;

            auto fills = book.add_market(side_enum, child_qty, ts);
            n_child += 1;

            for (const auto& f : fills) {
                filled_qty += f.qty;
                notional += f.qty * f.px;
            }

            remaining = total_qty - filled_qty;
            if (remaining <= 0) {
                remaining = 0;
            }
        }
    }

    std::optional<double> avg_px;
    if (filled_qty > 0) {
        avg_px = static_cast<double>(notional) / static_cast<double>(filled_qty);
    }

    int64_t unfilled_qty = std::max<int64_t>(0, total_qty - filled_qty);
    double completion_rate = static_cast<double>(filled_qty) / static_cast<double>(total_qty);

    std::optional<double> sf;
    std::optional<double> penalized;
    if (arrival_mid.has_value() && avg_px.has_value()) {
        sf = shortfall(side, *arrival_mid, *avg_px, filled_qty);
        penalized = *sf + penalty_per_share * static_cast<double>(unfilled_qty);
    }

    std::optional<double> sf_per_share;
    if (sf.has_value() && filled_qty > 0) {
        sf_per_share = *sf / static_cast<double>(filled_qty);
    }

    std::optional<double> penalized_per_share;
    if (penalized.has_value() && total_qty > 0) {
        penalized_per_share = *penalized / static_cast<double>(total_qty);
    }

    TwapReport report;
    report.side = side;
    report.target_qty = total_qty;
    report.filled_qty = filled_qty;
    report.avg_fill_px = avg_px;
    report.arrival_mid = arrival_mid;
    report.shortfall = sf;
    report.n_child_orders = n_child;
    report.unfilled_qty = unfilled_qty;
    report.completion_rate = completion_rate;
    report.penalty_per_share = penalty_per_share;
    report.penalized_cost = penalized;
    report.shortfall_per_share = sf_per_share;
    report.penalized_cost_per_share = penalized_per_share;

    return {book, report};
}

std::pair<LimitOrderBook, VwapReport> run_vwap(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t bucket_interval,
    const FlowConfig& cfg,
    int64_t seed,
    int64_t warmup_events,
    double penalty_per_share
) {
    if (side != "BID" && side != "ASK") {
        throw std::invalid_argument("side must be 'BID' or 'ASK'");
    }
    if (total_qty <= 0) {
        throw std::invalid_argument("total_qty must be > 0");
    }
    if (horizon_events <= 0) {
        throw std::invalid_argument("horizon_events must be > 0");
    }
    if (bucket_interval <= 0) {
        throw std::invalid_argument("bucket_interval must be > 0");
    }
    if (penalty_per_share < 0) {
        throw std::invalid_argument("penalty_per_share must be >= 0");
    }

    auto vols = forecast_market_volume(horizon_events, bucket_interval, cfg, seed, warmup_events);
    int64_t n_buckets = static_cast<int64_t>(vols.size());
    int64_t forecast_total_mkt_vol = 0;
    for (int64_t v : vols) {
        forecast_total_mkt_vol += v;
    }

    auto bucket_targets = compute_bucket_targets(total_qty, vols);

    LimitOrderBook book;
    PoissonOrderFlow flow(cfg, seed);

    int64_t ts = 0;
    for (int64_t i = 0; i < warmup_events; ++i) {
        ts += cfg.dt_us;
        flow.step(book, ts);
    }

    auto arrival_mid = book.mid();

    int64_t filled_qty = 0;
    int64_t notional = 0;
    int64_t n_child = 0;
    int64_t remaining = total_qty;
    Side side_enum = parse_side(side);

    for (int64_t i = 0; i < horizon_events; ++i) {
        ts += cfg.dt_us;

        flow.step(book, ts);

        if ((i % bucket_interval) == 0 && remaining > 0) {
            int64_t b = i / bucket_interval;
            int64_t child_qty = remaining;
            if (b < n_buckets) {
                child_qty = std::min(remaining, bucket_targets[static_cast<size_t>(b)]);
            }

            if (child_qty > 0) {
                auto fills = book.add_market(side_enum, child_qty, ts);
                n_child += 1;
                for (const auto& f : fills) {
                    filled_qty += f.qty;
                    notional += f.qty * f.px;
                }
                remaining = total_qty - filled_qty;
                if (remaining <= 0) {
                    remaining = 0;
                }
            }
        }
    }

    std::optional<double> avg_px;
    if (filled_qty > 0) {
        avg_px = static_cast<double>(notional) / static_cast<double>(filled_qty);
    }

    int64_t unfilled_qty = std::max<int64_t>(0, total_qty - filled_qty);
    double completion_rate = static_cast<double>(filled_qty) / static_cast<double>(total_qty);

    std::optional<double> sf;
    std::optional<double> penalized;
    if (arrival_mid.has_value() && avg_px.has_value()) {
        sf = shortfall(side, *arrival_mid, *avg_px, filled_qty);
        penalized = *sf + penalty_per_share * static_cast<double>(unfilled_qty);
    }

    std::optional<double> sf_per_share;
    if (sf.has_value() && filled_qty > 0) {
        sf_per_share = *sf / static_cast<double>(filled_qty);
    }

    std::optional<double> penalty_per_share_out;
    if (penalized.has_value() && total_qty > 0) {
        penalty_per_share_out = *penalized / static_cast<double>(total_qty);
    }

    VwapReport report;
    report.side = side;
    report.target_qty = total_qty;
    report.filled_qty = filled_qty;
    report.avg_fill_px = avg_px;
    report.arrival_mid = arrival_mid;
    report.shortfall = sf;
    report.n_child_orders = n_child;
    report.unfilled_qty = unfilled_qty;
    report.completion_rate = completion_rate;
    report.penalty_per_share = penalty_per_share_out;
    report.penalized_cost = penalized;
    report.shortfall_per_share = sf_per_share;
    report.penalized_cost_per_share = penalty_per_share_out;
    report.n_buckets = n_buckets;
    report.bucket_interval = bucket_interval;
    report.forecast_total_mkt_vol = forecast_total_mkt_vol;

    return {book, report};
}
