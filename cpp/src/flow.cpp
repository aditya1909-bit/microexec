// flow.cpp
#include "flow.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr size_t kBinaryTickFields = 4;

std::string trim_ascii(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])) != 0) {
        start += 1;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        end -= 1;
    }
    return s.substr(start, end - start);
}

bool parse_int64(const std::string& token, int64_t& out) {
    const std::string trimmed = trim_ascii(token);
    if (trimmed.empty()) {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    long long val = std::strtoll(trimmed.c_str(), &end, 10);
    if (errno != 0 || end == trimmed.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int64_t>(val);
    return true;
}

Side parse_side_token(const std::string& token) {
    std::string t = trim_ascii(token);
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (t == "BID" || t == "B" || t == "1") {
        return Side::Bid;
    }
    if (t == "ASK" || t == "A" || t == "-1") {
        return Side::Ask;
    }
    int64_t side_num = 0;
    if (parse_int64(t, side_num)) {
        return side_num >= 0 ? Side::Bid : Side::Ask;
    }
    throw std::invalid_argument("Unknown side token: " + token);
}

bool parse_csv_tick(
    const std::string& line,
    int64_t& ts,
    Side& side,
    int64_t& px,
    int64_t& qty
) {
    if (line.empty()) {
        return false;
    }
    if (line[0] == '#') {
        return false;
    }
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    if (tokens.size() < 4) {
        return false;
    }
    int64_t ts_val = 0;
    if (!parse_int64(tokens[0], ts_val)) {
        for (unsigned char c : tokens[0]) {
            if (std::isalpha(c) != 0) {
                return false;
            }
        }
        throw std::invalid_argument("Invalid CSV ts value: " + tokens[0]);
    }
    ts = ts_val;
    side = parse_side_token(tokens[1]);
    if (!parse_int64(tokens[2], px) || !parse_int64(tokens[3], qty)) {
        throw std::invalid_argument("Invalid CSV px/qty value");
    }
    return true;
}

bool parse_fixed_int(const std::string& s, size_t pos, size_t len, int& out) {
    if (pos + len > s.size()) {
        return false;
    }
    int val = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = s[pos + i];
        if (c < '0' || c > '9') {
            return false;
        }
        val = val * 10 + (c - '0');
    }
    out = val;
    return true;
}

int64_t parse_truefx_ts_us(const std::string& token) {
    const std::string s = trim_ascii(token);
    int year = 0;
    int mon = 0;
    int day = 0;
    int hour = 0;
    int min = 0;
    int sec = 0;
    int msec = 0;
    if (
        !parse_fixed_int(s, 0, 4, year) ||
        !parse_fixed_int(s, 4, 2, mon) ||
        !parse_fixed_int(s, 6, 2, day) ||
        s.size() < 21 ||
        !parse_fixed_int(s, 9, 2, hour) ||
        !parse_fixed_int(s, 12, 2, min) ||
        !parse_fixed_int(s, 15, 2, sec) ||
        !parse_fixed_int(s, 18, 3, msec)
    ) {
        throw std::invalid_argument("Invalid TrueFX timestamp: " + token);
    }

    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = mon - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = min;
    tm.tm_sec = sec;
    tm.tm_isdst = 0;

    time_t epoch = timegm(&tm);
    if (epoch == static_cast<time_t>(-1)) {
        throw std::invalid_argument("Failed to convert TrueFX timestamp: " + token);
    }

    return static_cast<int64_t>(epoch) * 1'000'000 + static_cast<int64_t>(msec) * 1'000;
}

bool parse_truefx_line(
    const std::string& line,
    int64_t& ts,
    int64_t& bid_px,
    int64_t& ask_px,
    int64_t price_scale
) {
    if (line.empty()) {
        return false;
    }
    if (line[0] == '#') {
        return false;
    }
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }
    if (tokens.size() < 4) {
        return false;
    }

    ts = parse_truefx_ts_us(tokens[1]);
    double bid = std::stod(trim_ascii(tokens[2]));
    double ask = std::stod(trim_ascii(tokens[3]));
    if (!std::isfinite(bid) || !std::isfinite(ask)) {
        throw std::invalid_argument("Invalid TrueFX price values");
    }
    bid_px = static_cast<int64_t>(std::llround(bid * static_cast<double>(price_scale)));
    ask_px = static_cast<int64_t>(std::llround(ask * static_cast<double>(price_scale)));
    return true;
}

}  // namespace

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

    // Basic parameter sanity checks
    if (dt_us <= 0) {
        throw std::invalid_argument("dt_us must be positive");
    }
    if (soft_max_orders < 0 || hard_max_orders < 0) {
        throw std::invalid_argument("max order caps must be non-negative");
    }
    if (hard_max_orders <= soft_max_orders) {
        throw std::invalid_argument("hard_max_orders must be > soft_max_orders");
    }
    if (min_offset <= 0 || max_offset <= 0) {
        throw std::invalid_argument("offset bounds must be positive");
    }
    if (max_offset < min_offset) {
        throw std::invalid_argument("max_offset must be >= min_offset");
    }
    if (offset_power <= 0.0 || !std::isfinite(offset_power)) {
        throw std::invalid_argument("offset_power must be positive and finite");
    }
    if (min_qty <= 0 || max_qty <= 0 || max_qty < min_qty) {
        throw std::invalid_argument("limit qty bounds must be positive and max_qty >= min_qty");
    }
    if (min_market_qty <= 0 || max_market_qty <= 0 || max_market_qty < min_market_qty) {
        throw std::invalid_argument("market qty bounds must be positive and max_market_qty >= min_market_qty");
    }
    if (cancel_levels <= 0) {
        throw std::invalid_argument("cancel_levels must be positive");
    }
    if (min_touch_qty < 0) {
        throw std::invalid_argument("min_touch_qty must be non-negative");
    }
    if (!std::isfinite(cancel_boost) || cancel_boost < 0.0) {
        throw std::invalid_argument("cancel_boost must be finite and non-negative");
    }
    if (p_ioc < 0.0 || p_fok < 0.0 || p_ioc > 1.0 || p_fok > 1.0) {
        throw std::invalid_argument("IOC/FOK probabilities must be in [0,1]");
    }
    if (p_ioc + p_fok > 1.0) {
        throw std::invalid_argument("IOC/FOK probabilities must sum to <= 1.0");
    }
}

PoissonOrderFlow::PoissonOrderFlow(const FlowConfig& config, int64_t seed)
    : config_(config),
      rng_(static_cast<uint32_t>(seed)),
      next_order_id_(1),
      ref_mid_(config.initial_mid),
      offset_lo_(config.min_offset),
      offset_hi_(config.max_offset) {
    config_.validate();

    // Precompute offset distribution once for speed.
    if (offset_hi_ > offset_lo_) {
        std::vector<double> weights;
        weights.reserve(static_cast<size_t>(offset_hi_ - offset_lo_ + 1));
        for (int64_t k = offset_lo_; k <= offset_hi_; ++k) {
            weights.push_back(1.0 / std::pow(static_cast<double>(k), config_.offset_power));
        }
        offset_dist_.emplace(weights.begin(), weights.end());
    }
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
    (void)p_cancel; // p_cancel implied by remaining probability mass
    double r = uni01_(rng_);
    if (r < p_limit) {
        return EventKind::Limit;
    }
    if (r < p_limit + p_market) {
        return EventKind::Market;
    }
    return EventKind::Cancel;
}

int64_t PoissonOrderFlow::sample_offset() {
    if (offset_hi_ <= offset_lo_) {
        return offset_lo_;
    }
    if (offset_dist_.has_value()) {
        int idx = (*offset_dist_)(rng_);
        return offset_lo_ + static_cast<int64_t>(idx);
    }
    return offset_lo_;
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
            side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
        }

        int64_t offset = sample_offset();
        std::uniform_int_distribution<int64_t> qty_dist(config_.min_qty, config_.max_qty);
        int64_t qty = qty_dist(rng_);

        int64_t px = (side == Side::Bid) ? (mid - offset) : (mid + offset);
        int64_t oid = new_order_id();
        OrderType order_type = OrderType::Limit;
        double r = uni01_(rng_);
        if (r < config_.p_ioc) {
            order_type = OrderType::IOC;
        } else if (r < config_.p_ioc + config_.p_fok) {
            order_type = OrderType::FOK;
        }

        auto fills = book.add_limit(Order{oid, side, px, qty, ts, "SIM", order_type});
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
                side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
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
        if (im > 0.2) {
            side = (uni01_(rng_) < 0.75) ? Side::Ask : Side::Bid;
        } else if (im < -0.2) {
            side = (uni01_(rng_) < 0.75) ? Side::Bid : Side::Ask;
        } else {
            side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
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
    if (im > 0.2) {
        side = (uni01_(rng_) < 0.75) ? Side::Bid : Side::Ask;
    } else if (im < -0.2) {
        side = (uni01_(rng_) < 0.75) ? Side::Ask : Side::Bid;
    } else {
        side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
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

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> PoissonOrderFlow::step_n(
    LimitOrderBook& book,
    int64_t start_ts,
    int64_t n_events
) {
    int64_t n_limit = 0;
    int64_t n_market = 0;
    int64_t n_cancel = 0;
    int64_t traded_qty = 0;
    int64_t ts = start_ts;

    for (int64_t i = 0; i < n_events; ++i) {
        ts += config_.dt_us;
        auto [kind, traded] = step(book, ts);
        traded_qty += traded;
        if (kind == EventKind::Limit) {
            n_limit += 1;
        } else if (kind == EventKind::Market) {
            n_market += 1;
        } else if (kind == EventKind::Cancel) {
            n_cancel += 1;
        }
    }

    return {ts, n_limit, n_market, n_cancel, traded_qty};
}

HawkesOrderFlow::HawkesOrderFlow(const HawkesConfig& config, int64_t seed)
    : config_(config),
      rng_(static_cast<uint32_t>(seed)),
      next_order_id_(1),
      ref_mid_(config.base.initial_mid),
      lambda_limit_(config.base.p_limit),
      lambda_market_(config.base.p_market),
      lambda_cancel_(config.base.p_cancel) {
    config_.base.validate();
}

OrderType HawkesOrderFlow::sample_limit_type() {
    double r = uni01_(rng_);
    if (r < config_.base.p_ioc) {
        return OrderType::IOC;
    }
    if (r < config_.base.p_ioc + config_.base.p_fok) {
        return OrderType::FOK;
    }
    return OrderType::Limit;
}

EventKind HawkesOrderFlow::sample_kind(double lambda_limit, double lambda_market, double lambda_cancel) {
    double total = lambda_limit + lambda_market + lambda_cancel;
    if (total <= 0.0) {
        return EventKind::Limit;
    }
    double r = uni01_(rng_) * total;
    if (r < lambda_limit) {
        return EventKind::Limit;
    }
    if (r < lambda_limit + lambda_market) {
        return EventKind::Market;
    }
    return EventKind::Cancel;
}

int64_t HawkesOrderFlow::sample_offset() {
    int64_t lo = config_.base.min_offset;
    int64_t hi = config_.base.max_offset;
    if (hi <= lo) {
        return lo;
    }
    std::vector<double> weights;
    weights.reserve(static_cast<size_t>(hi - lo + 1));
    for (int64_t k = lo; k <= hi; ++k) {
        weights.push_back(1.0 / std::pow(static_cast<double>(k), config_.base.offset_power));
    }
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int idx = dist(rng_);
    return lo + idx;
}

int64_t HawkesOrderFlow::mid_tick(const LimitOrderBook& book) const {
    auto m = book.mid();
    if (!m.has_value()) {
        return ref_mid_;
    }
    return static_cast<int64_t>(std::llround(*m));
}

int64_t HawkesOrderFlow::new_order_id() {
    int64_t oid = next_order_id_;
    next_order_id_ += 1;
    return oid;
}

std::pair<EventKind, int64_t> HawkesOrderFlow::step(LimitOrderBook& book, int64_t ts) {
    double dt = static_cast<double>(config_.base.dt_us) / 1'000'000.0;
    double decay = std::exp(-config_.decay * dt);
    lambda_limit_ = config_.base.p_limit + (lambda_limit_ - config_.base.p_limit) * decay;
    lambda_market_ = config_.base.p_market + (lambda_market_ - config_.base.p_market) * decay;
    lambda_cancel_ = config_.base.p_cancel + (lambda_cancel_ - config_.base.p_cancel) * decay;

    EventKind kind = sample_kind(lambda_limit_, lambda_market_, lambda_cancel_);

    if (!book.best_bid().has_value() || !book.best_ask().has_value()) {
        kind = EventKind::Limit;
    } else {
        auto bid_touch = book.depth(Side::Bid, 1);
        auto ask_touch = book.depth(Side::Ask, 1);
        int64_t bid_q = bid_touch.empty() ? 0 : bid_touch[0].second;
        int64_t ask_q = ask_touch.empty() ? 0 : ask_touch[0].second;
        if (bid_q < config_.base.min_touch_qty || ask_q < config_.base.min_touch_qty) {
            kind = EventKind::Limit;
        }
    }

    if (kind == EventKind::Cancel && book.order_count() == 0) {
        if (config_.base.cancel_fallback_to_limit) {
            kind = EventKind::Limit;
        } else {
            return {kind, 0};
        }
    }

    if (kind == EventKind::Limit) {
        int64_t mid = mid_tick(book);
        Side side = Side::Bid;
        if (!book.best_bid().has_value() && !book.best_ask().has_value()) {
            side = ((ts / config_.base.dt_us) % 2 == 0) ? Side::Bid : Side::Ask;
        } else if (!book.best_bid().has_value()) {
            side = Side::Bid;
        } else if (!book.best_ask().has_value()) {
            side = Side::Ask;
        } else {
            side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
        }

        int64_t offset = sample_offset();
        std::uniform_int_distribution<int64_t> qty_dist(config_.base.min_qty, config_.base.max_qty);
        int64_t qty = qty_dist(rng_);
        int64_t px = (side == Side::Bid) ? (mid - offset) : (mid + offset);
        int64_t oid = new_order_id();
        OrderType order_type = sample_limit_type();

        auto fills = book.add_limit(Order{oid, side, px, qty, ts, "SIM", order_type});
        int64_t traded = 0;
        for (const auto& f : fills) {
            traded += f.qty;
        }

        lambda_limit_ += config_.alpha_limit;
        return {kind, traded};
    }

    if (kind == EventKind::Market) {
        double im = book.imbalance(1);
        Side side = Side::Bid;
        if (im > 0.2) {
            side = (uni01_(rng_) < 0.75) ? Side::Ask : Side::Bid;
        } else if (im < -0.2) {
            side = (uni01_(rng_) < 0.75) ? Side::Bid : Side::Ask;
        } else {
            side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
        }

        std::uniform_int_distribution<int64_t> qty_dist(config_.base.min_market_qty, config_.base.max_market_qty);
        int64_t qty = qty_dist(rng_);

        auto fills = book.add_market(side, qty, ts);
        int64_t traded = 0;
        for (const auto& f : fills) {
            traded += f.qty;
        }

        lambda_market_ += config_.alpha_market;
        return {kind, traded};
    }

    int64_t cancel_levels = config_.base.cancel_levels;
    if (book.order_count() == 0) {
        return {EventKind::Limit, 0};
    }

    double im = book.imbalance(1);
    Side side = Side::Bid;
    if (im > 0.2) {
        side = (uni01_(rng_) < 0.75) ? Side::Bid : Side::Ask;
    } else if (im < -0.2) {
        side = (uni01_(rng_) < 0.75) ? Side::Ask : Side::Bid;
    } else {
        side = (uni01_(rng_) < 0.5) ? Side::Bid : Side::Ask;
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
            lambda_cancel_ += config_.alpha_cancel;
            return {kind, 0};
        }
    }

    auto ids = book.all_order_ids();
    if (!ids.empty()) {
        std::uniform_int_distribution<size_t> id_dist(0, ids.size() - 1);
        int64_t oid = ids[id_dist(rng_)];
        book.cancel(oid, ts);
    }

    lambda_cancel_ += config_.alpha_cancel;
    return {kind, 0};
}

HistoricalOrderFlow::HistoricalOrderFlow(
    const std::string& path,
    HistoricalFormat format,
    int64_t start_order_id,
    int64_t fixed_qty,
    int64_t price_scale
) : format_(format),
    next_order_id_(start_order_id),
    fixed_qty_(fixed_qty),
    price_scale_(price_scale) {
    if (start_order_id <= 0) {
        throw std::invalid_argument("start_order_id must be > 0");
    }
    if (fixed_qty_ <= 0) {
        throw std::invalid_argument("fixed_qty must be > 0");
    }
    if (price_scale_ <= 0) {
        throw std::invalid_argument("price_scale must be > 0");
    }
    if (format_ == HistoricalFormat::Binary) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open binary tick file: " + path);
        }
        struct stat st;
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to stat binary tick file: " + path);
        }
        if (st.st_size <= 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::invalid_argument("Binary tick file is empty: " + path);
        }
        map_size_ = static_cast<size_t>(st.st_size);
        size_t record_size = sizeof(int64_t) * kBinaryTickFields;
        if (map_size_ % record_size != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::invalid_argument("Binary tick file size is not a multiple of record size");
        }
        void* mapped = ::mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Failed to mmap binary tick file");
        }
        map_ = static_cast<const uint8_t*>(mapped);
        map_index_ = 0;
    } else {
        csv_.open(path);
        if (!csv_) {
            throw std::runtime_error("Failed to open CSV tick file: " + path);
        }
    }
}

HistoricalOrderFlow::~HistoricalOrderFlow() {
    if (map_ != nullptr && map_ != reinterpret_cast<const uint8_t*>(MAP_FAILED)) {
        ::munmap(const_cast<uint8_t*>(map_), map_size_);
        map_ = nullptr;
        map_size_ = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    if (csv_.is_open()) {
        csv_.close();
    }
}

bool HistoricalOrderFlow::done() const {
    return done_;
}

bool HistoricalOrderFlow::read_next_tick(int64_t& ts, Side& side, int64_t& px, int64_t& qty) {
    if (done_) {
        return false;
    }
    if (format_ == HistoricalFormat::Binary) {
        size_t record_size = sizeof(int64_t) * kBinaryTickFields;
        size_t total_records = map_size_ / record_size;
        if (map_index_ >= total_records) {
            done_ = true;
            return false;
        }
        const int64_t* data = reinterpret_cast<const int64_t*>(map_);
        size_t offset = map_index_ * kBinaryTickFields;
        ts = data[offset];
        int64_t side_val = data[offset + 1];
        px = data[offset + 2];
        qty = data[offset + 3];
        side = side_val >= 0 ? Side::Bid : Side::Ask;
        map_index_ += 1;
        return true;
    }

    std::string line;
    while (std::getline(csv_, line)) {
        int64_t ts_val = 0;
        Side side_val = Side::Bid;
        int64_t px_val = 0;
        int64_t qty_val = 0;
        if (!parse_csv_tick(line, ts_val, side_val, px_val, qty_val)) {
            continue;
        }
        ts = ts_val;
        side = side_val;
        px = px_val;
        qty = qty_val;
        return true;
    }
    done_ = true;
    return false;
}

bool HistoricalOrderFlow::step(LimitOrderBook& book, int64_t& ts_out, int64_t& traded_out) {
    if (format_ == HistoricalFormat::TrueFX) {
        if (done_) {
            return false;
        }
        std::string line;
        int64_t ts = 0;
        int64_t bid_px = 0;
        int64_t ask_px = 0;
        bool parsed = false;
        while (std::getline(csv_, line)) {
            if (parse_truefx_line(line, ts, bid_px, ask_px, price_scale_)) {
                parsed = true;
                break;
            }
        }
        if (!parsed) {
            done_ = true;
            return false;
        }

        if (lp_bid_id_ > 0) {
            book.cancel(lp_bid_id_, ts);
        }
        if (lp_ask_id_ > 0) {
            book.cancel(lp_ask_id_, ts);
        }

        int64_t bid_id = next_order_id_++;
        int64_t ask_id = next_order_id_++;
        auto bid_fills = book.add_limit(Order{bid_id, Side::Bid, bid_px, fixed_qty_, ts, "LP", OrderType::Limit});
        auto ask_fills = book.add_limit(Order{ask_id, Side::Ask, ask_px, fixed_qty_, ts, "LP", OrderType::Limit});
        int64_t traded = 0;
        for (const auto& f : bid_fills) {
            traded += f.qty;
        }
        for (const auto& f : ask_fills) {
            traded += f.qty;
        }
        lp_bid_id_ = bid_id;
        lp_ask_id_ = ask_id;
        ts_out = ts;
        traded_out = traded;
        return true;
    }

    int64_t ts = 0;
    Side side = Side::Bid;
    int64_t px = 0;
    int64_t qty = 0;
    if (!read_next_tick(ts, side, px, qty)) {
        return false;
    }

    int64_t traded = 0;
    int64_t oid = next_order_id_++;
    auto fills = book.add_limit(Order{oid, side, px, qty, ts, "HIST", OrderType::Limit});
    for (const auto& f : fills) {
        traded += f.qty;
    }
    ts_out = ts;
    traded_out = traded;
    return true;
}
