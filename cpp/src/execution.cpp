// execution.cpp
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
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

double safe_dt_seconds(int64_t dt_us, int64_t child_interval) {
    if (dt_us <= 0 || child_interval <= 0) {
        throw std::invalid_argument("dt_us and child_interval must be > 0");
    }
    return static_cast<double>(dt_us) * static_cast<double>(child_interval) / 1'000'000.0;
}

std::vector<int64_t> ac_schedule(
    int64_t total_qty,
    int64_t n_slices,
    double dt_seconds,
    const ImpactModel& impact
) {
    if (n_slices <= 0) {
        return {};
    }

    std::vector<int64_t> qtys;
    qtys.reserve(static_cast<size_t>(n_slices));

    if (impact.risk_aversion <= 0.0 || impact.volatility <= 0.0 || impact.temporary_impact <= 0.0) {
        int64_t base = total_qty / n_slices;
        int64_t rem = total_qty - base * n_slices;
        for (int64_t i = 0; i < n_slices; ++i) {
            qtys.push_back(base + (i < rem ? 1 : 0));
        }
        return qtys;
    }

    double kappa = std::sqrt(impact.risk_aversion * impact.volatility * impact.volatility / impact.temporary_impact);
    double T = static_cast<double>(n_slices) * dt_seconds;
    if (kappa < 1e-10 || T <= 0.0) {
        int64_t base = total_qty / n_slices;
        int64_t rem = total_qty - base * n_slices;
        for (int64_t i = 0; i < n_slices; ++i) {
            qtys.push_back(base + (i < rem ? 1 : 0));
        }
        return qtys;
    }

    double denom = std::sinh(kappa * T);
    std::vector<double> holdings;
    holdings.reserve(static_cast<size_t>(n_slices + 1));
    for (int64_t k = 0; k <= n_slices; ++k) {
        double t = static_cast<double>(k) * dt_seconds;
        double x = (denom == 0.0)
            ? static_cast<double>(total_qty) * (1.0 - t / T)
            : static_cast<double>(total_qty) * std::sinh(kappa * (T - t)) / denom;
        holdings.push_back(x);
    }

    int64_t sum = 0;
    for (int64_t k = 0; k < n_slices; ++k) {
        double raw = holdings[static_cast<size_t>(k)] - holdings[static_cast<size_t>(k + 1)];
        int64_t q = static_cast<int64_t>(std::llround(raw));
        if (q < 0) {
            q = 0;
        }
        qtys.push_back(q);
        sum += q;
    }

    int64_t diff = total_qty - sum;
    for (int64_t k = n_slices - 1; k >= 0 && diff != 0; --k) {
        int64_t adjust = diff;
        if (adjust < 0) {
            int64_t can_remove = qtys[static_cast<size_t>(k)];
            if (-adjust > can_remove) {
                adjust = -can_remove;
            }
        }
        qtys[static_cast<size_t>(k)] += adjust;
        diff -= adjust;
    }

    return qtys;
}

int64_t derive_cap(int64_t total_qty, int64_t cap_abs, double cap_frac) {
    int64_t cap = 0;
    if (cap_abs > 0) {
        cap = cap_abs;
    }
    if (cap_frac > 0.0) {
        int64_t frac_cap = static_cast<int64_t>(std::llround(static_cast<double>(total_qty) * cap_frac));
        if (cap == 0 || frac_cap < cap) {
            cap = frac_cap;
        }
    }
    return cap;
}

void apply_cap(std::vector<int64_t>& qtys, int64_t total_qty, int64_t cap) {
    if (cap <= 0) {
        return;
    }
    int64_t sum = 0;
    for (auto& q : qtys) {
        if (q > cap) {
            q = cap;
        }
        sum += q;
    }

    if (sum < total_qty) {
        int64_t remaining = total_qty - sum;
        for (size_t i = 0; i < qtys.size() && remaining > 0; ++i) {
            int64_t headroom = cap - qtys[i];
            if (headroom <= 0) {
                continue;
            }
            int64_t add = std::min(headroom, remaining);
            qtys[i] += add;
            remaining -= add;
        }
    } else if (sum > total_qty) {
        int64_t excess = sum - total_qty;
        for (size_t i = qtys.size(); i > 0 && excess > 0; --i) {
            size_t idx = i - 1;
            int64_t can_remove = std::min(qtys[idx], excess);
            qtys[idx] -= can_remove;
            excess -= can_remove;
        }
    }
}

struct AcMetrics {
    int64_t filled_qty = 0;
    double notional = 0.0;
    double risk_penalty = 0.0;
    std::optional<double> avg_px;
    std::optional<double> shortfall;
    std::optional<double> objective;
};

AcMetrics compute_ac_metrics(
    const std::vector<int64_t>& qtys,
    const std::string& side,
    double arrival_mid,
    int64_t total_qty,
    double dt_seconds,
    const ImpactModel& impact
) {
    AcMetrics out;
    double side_sign = (side == "BID") ? 1.0 : -1.0;
    double cumulative = 0.0;
    double x = static_cast<double>(total_qty);

    for (int64_t q : qtys) {
        out.risk_penalty += impact.risk_aversion * impact.volatility * impact.volatility * x * x * dt_seconds;
        if (q <= 0) {
            continue;
        }
        cumulative += static_cast<double>(q);
        double perm = impact.permanent_impact * (cumulative / static_cast<double>(total_qty));
        double rate = static_cast<double>(q) / dt_seconds;
        double temp = impact.temporary_impact * rate;
        double exec_px = arrival_mid + side_sign * (perm + temp);

        out.notional += exec_px * static_cast<double>(q);
        out.filled_qty += q;
        x -= static_cast<double>(q);
    }

    if (out.filled_qty > 0) {
        out.avg_px = out.notional / static_cast<double>(out.filled_qty);
        out.shortfall = shortfall(side, arrival_mid, *out.avg_px, out.filled_qty);
        out.objective = *out.shortfall + out.risk_penalty;
    }

    return out;
}

}  // namespace

std::pair<LimitOrderBook, TwapReport> run_twap(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t child_interval,
    const FlowConfig& cfg,
    int64_t latency_us,
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
    int64_t latency_events = 0;
    if (latency_us > 0 && cfg.dt_us > 0) {
        latency_events = (latency_us + cfg.dt_us - 1) / cfg.dt_us;
    }
    struct PendingOrder {
        int64_t due_i;
        int64_t qty;
    };
    std::deque<PendingOrder> pending;

    for (int64_t i = 0; i < horizon_events; ++i) {
        ts += cfg.dt_us;

        flow.step(book, ts);

        while (!pending.empty() && pending.front().due_i <= i) {
            int64_t qty = pending.front().qty;
            pending.pop_front();
            if (qty <= 0) {
                continue;
            }
            auto fills = book.add_market(side_enum, qty, ts);
            for (const auto& f : fills) {
                filled_qty += f.qty;
                notional += f.qty * f.px;
            }
        }

        if ((i % child_interval) == 0 && remaining > 0) {
            int64_t slices_left = std::max<int64_t>(1, n_slices - (i / child_interval));
            int64_t child_qty = (remaining + slices_left - 1) / slices_left;

            n_child += 1;
            if (latency_events == 0) {
                auto fills = book.add_market(side_enum, child_qty, ts);
                for (const auto& f : fills) {
                    filled_qty += f.qty;
                    notional += f.qty * f.px;
                }
            } else {
                pending.push_back(PendingOrder{i + latency_events, child_qty});
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
    int64_t latency_us,
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
    int64_t latency_events = 0;
    if (latency_us > 0 && cfg.dt_us > 0) {
        latency_events = (latency_us + cfg.dt_us - 1) / cfg.dt_us;
    }
    struct PendingOrder {
        int64_t due_i;
        int64_t qty;
    };
    std::deque<PendingOrder> pending;

    for (int64_t i = 0; i < horizon_events; ++i) {
        ts += cfg.dt_us;

        flow.step(book, ts);

        while (!pending.empty() && pending.front().due_i <= i) {
            int64_t qty = pending.front().qty;
            pending.pop_front();
            if (qty <= 0) {
                continue;
            }
            auto fills = book.add_market(side_enum, qty, ts);
            for (const auto& f : fills) {
                filled_qty += f.qty;
                notional += f.qty * f.px;
            }
        }

        if ((i % bucket_interval) == 0 && remaining > 0) {
            int64_t b = i / bucket_interval;
            int64_t child_qty = remaining;
            if (b < n_buckets) {
                child_qty = std::min(remaining, bucket_targets[static_cast<size_t>(b)]);
            }

            if (child_qty > 0) {
                n_child += 1;
                if (latency_events == 0) {
                    auto fills = book.add_market(side_enum, child_qty, ts);
                    for (const auto& f : fills) {
                        filled_qty += f.qty;
                        notional += f.qty * f.px;
                    }
                } else {
                    pending.push_back(PendingOrder{i + latency_events, child_qty});
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

std::pair<LimitOrderBook, AlmgrenChrissReport> run_almgren_chriss(
    const std::string& side,
    int64_t total_qty,
    int64_t horizon_events,
    int64_t child_interval,
    const FlowConfig& cfg,
    const ImpactModel& impact,
    int64_t cap_abs,
    double cap_frac,
    int64_t latency_us,
    int64_t seed,
    int64_t warmup_events
) {
    (void)latency_us;
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

    LimitOrderBook book;
    PoissonOrderFlow flow(cfg, seed);

    int64_t ts = 0;
    for (int64_t i = 0; i < warmup_events; ++i) {
        ts += cfg.dt_us;
        flow.step(book, ts);
    }

    auto arrival_mid = book.mid();
    double arrival_mid_value = arrival_mid.value_or(static_cast<double>(cfg.initial_mid));

    int64_t n_slices = std::max<int64_t>(1, (horizon_events + child_interval - 1) / child_interval);
    double dt_seconds = safe_dt_seconds(cfg.dt_us, child_interval);

    auto child_qtys_raw = ac_schedule(total_qty, n_slices, dt_seconds, impact);
    auto uncapped_metrics = compute_ac_metrics(child_qtys_raw, side, arrival_mid_value, total_qty, dt_seconds, impact);

    int64_t cap_value = derive_cap(total_qty, cap_abs, cap_frac);
    auto child_qtys = child_qtys_raw;
    if (cap_value > 0) {
        apply_cap(child_qtys, total_qty, cap_value);
    }
    auto metrics = compute_ac_metrics(child_qtys, side, arrival_mid_value, total_qty, dt_seconds, impact);

    int64_t filled_qty = metrics.filled_qty;
    int64_t unfilled_qty = std::max<int64_t>(0, total_qty - filled_qty);
    double completion_rate = static_cast<double>(filled_qty) / static_cast<double>(total_qty);

    AlmgrenChrissReport report;
    report.side = side;
    report.target_qty = total_qty;
    report.filled_qty = filled_qty;
    report.avg_fill_px = metrics.avg_px;
    report.arrival_mid = arrival_mid_value;
    report.shortfall = metrics.shortfall;
    report.n_child_orders = static_cast<int64_t>(child_qtys.size());
    report.unfilled_qty = unfilled_qty;
    report.completion_rate = completion_rate;
    report.risk_penalty = metrics.risk_penalty;
    report.objective = metrics.objective;
    report.shortfall_uncapped = uncapped_metrics.shortfall;
    report.objective_uncapped = uncapped_metrics.objective;
    report.cap_abs = cap_abs;
    report.cap_frac = cap_frac;
    report.cap_used = cap_value;
    report.capped = cap_value > 0;
    report.child_qtys = std::move(child_qtys);
    report.impact = impact;

    return {book, report};
}
