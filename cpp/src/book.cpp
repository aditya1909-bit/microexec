#include "book.hpp"

#include <algorithm>
#include <cmath>

std::optional<int64_t> LimitOrderBook::best_bid() const {
    if (bids_.empty()) {
        return std::nullopt;
    }
    return bids_.begin()->first;
}

std::optional<int64_t> LimitOrderBook::best_ask() const {
    if (asks_.empty()) {
        return std::nullopt;
    }
    return asks_.begin()->first;
}

std::optional<double> LimitOrderBook::mid() const {
    if (!best_bid().has_value() || !best_ask().has_value()) {
        return std::nullopt;
    }
    return 0.5 * (static_cast<double>(*best_bid()) + static_cast<double>(*best_ask()));
}

std::optional<int64_t> LimitOrderBook::spread() const {
    if (!best_bid().has_value() || !best_ask().has_value()) {
        return std::nullopt;
    }
    return *best_ask() - *best_bid();
}

std::vector<std::pair<int64_t, int64_t>> LimitOrderBook::depth(Side side, int levels) const {
    if (levels <= 0) {
        return {};
    }

    std::vector<std::pair<int64_t, int64_t>> out;
    out.reserve(static_cast<size_t>(levels));

    if (side == Side::Bid) {
        int count = 0;
        for (const auto& [px, queue] : bids_) {
            int64_t total = 0;
            for (int64_t oid : queue) {
                auto it = orders_.find(oid);
                if (it != orders_.end()) {
                    total += it->second.qty;
                }
            }
            out.emplace_back(px, total);
            count += 1;
            if (count >= levels) {
                break;
            }
        }
    } else {
        int count = 0;
        for (const auto& [px, queue] : asks_) {
            int64_t total = 0;
            for (int64_t oid : queue) {
                auto it = orders_.find(oid);
                if (it != orders_.end()) {
                    total += it->second.qty;
                }
            }
            out.emplace_back(px, total);
            count += 1;
            if (count >= levels) {
                break;
            }
        }
    }

    return out;
}

double LimitOrderBook::imbalance(int levels) const {
    int64_t bid_vol = 0;
    int64_t ask_vol = 0;
    for (const auto& [_, qty] : depth(Side::Bid, levels)) {
        bid_vol += qty;
    }
    for (const auto& [_, qty] : depth(Side::Ask, levels)) {
        ask_vol += qty;
    }
    int64_t denom = bid_vol + ask_vol;
    if (denom == 0) {
        return 0.0;
    }
    return static_cast<double>(bid_vol - ask_vol) / static_cast<double>(denom);
}

std::vector<Fill> LimitOrderBook::add_limit(Order order) {
    std::vector<Fill> fills;

    if (order.side == Side::Bid) {
        while (order.qty > 0 && best_ask().has_value() && order.px >= *best_ask()) {
            auto f = match(order, Side::Ask);
            fills.insert(fills.end(), f.begin(), f.end());
        }
    } else {
        while (order.qty > 0 && best_bid().has_value() && order.px <= *best_bid()) {
            auto f = match(order, Side::Bid);
            fills.insert(fills.end(), f.begin(), f.end());
        }
    }

    if (order.qty > 0) {
        add_to_book(order);
    }

    return fills;
}

std::vector<Fill> LimitOrderBook::add_market(Side side, int64_t qty, int64_t ts) {
    Order dummy{
        -1,
        side,
        0,
        qty,
        ts,
        "SIM",
        OrderType::Market,
    };

    std::vector<Fill> fills;
    Side opp = (side == Side::Bid) ? Side::Ask : Side::Bid;

    while (dummy.qty > 0) {
        if (opp == Side::Ask && !best_ask().has_value()) {
            break;
        }
        if (opp == Side::Bid && !best_bid().has_value()) {
            break;
        }
        auto f = match(dummy, opp);
        fills.insert(fills.end(), f.begin(), f.end());
    }

    return fills;
}

bool LimitOrderBook::cancel(int64_t order_id, int64_t /*ts*/) {
    auto it = orders_.find(order_id);
    if (it == orders_.end()) {
        return false;
    }

    const Order order = it->second;

    if (order.side == Side::Bid) {
        auto book_it = bids_.find(order.px);
        if (book_it == bids_.end()) {
            return false;
        }
        auto& queue = book_it->second;
        auto qit = std::find(queue.begin(), queue.end(), order_id);
        if (qit == queue.end()) {
            return false;
        }
        queue.erase(qit);
        if (queue.empty()) {
            bids_.erase(book_it);
        }
    } else {
        auto book_it = asks_.find(order.px);
        if (book_it == asks_.end()) {
            return false;
        }
        auto& queue = book_it->second;
        auto qit = std::find(queue.begin(), queue.end(), order_id);
        if (qit == queue.end()) {
            return false;
        }
        queue.erase(qit);
        if (queue.empty()) {
            asks_.erase(book_it);
        }
    }

    // Only erase from orders_ after we successfully removed it from the book structure.
    orders_.erase(it);
    return true;
}

size_t LimitOrderBook::order_count() const {
    return orders_.size();
}

std::vector<int64_t> LimitOrderBook::all_order_ids() const {
    std::vector<int64_t> ids;
    ids.reserve(orders_.size());
    for (const auto& kv : orders_) {
        ids.push_back(kv.first);
    }
    return ids;
}

std::vector<int64_t> LimitOrderBook::order_ids_at_price(Side side, int64_t px) const {
    std::vector<int64_t> ids;
    if (side == Side::Bid) {
        auto it = bids_.find(px);
        if (it == bids_.end()) {
            return ids;
        }
        ids.assign(it->second.begin(), it->second.end());
    } else {
        auto it = asks_.find(px);
        if (it == asks_.end()) {
            return ids;
        }
        ids.assign(it->second.begin(), it->second.end());
    }
    return ids;
}

void LimitOrderBook::add_to_book(const Order& order) {
    if (order.side == Side::Bid) {
        auto& queue = bids_[order.px];
        queue.push_back(order.id);
    } else {
        auto& queue = asks_[order.px];
        queue.push_back(order.id);
    }
    orders_[order.id] = order;
}

std::vector<Fill> LimitOrderBook::match(Order& incoming, Side resting_side) {
    std::vector<Fill> fills;

    if (resting_side == Side::Ask) {
        if (asks_.empty()) {
            return fills;
        }
        auto it = asks_.begin();
        int64_t px = it->first;
        auto& queue = it->second;

        while (incoming.qty > 0 && !queue.empty()) {
            int64_t order_id = queue.front();
            auto rest_it = orders_.find(order_id);
            if (rest_it == orders_.end()) {
                queue.pop_front();
                continue;
            }
            Order& resting = rest_it->second;
            int64_t traded = std::min(incoming.qty, resting.qty);

            incoming.qty -= traded;
            resting.qty -= traded;

            fills.push_back(Fill{resting.id, px, traded, Liquidity::Maker, incoming.ts});

            if (resting.qty == 0) {
                queue.pop_front();
                orders_.erase(rest_it);
            }
        }

        if (queue.empty()) {
            asks_.erase(it);
        }
    } else {
        if (bids_.empty()) {
            return fills;
        }
        auto it = bids_.begin();
        int64_t px = it->first;
        auto& queue = it->second;

        while (incoming.qty > 0 && !queue.empty()) {
            int64_t order_id = queue.front();
            auto rest_it = orders_.find(order_id);
            if (rest_it == orders_.end()) {
                queue.pop_front();
                continue;
            }
            Order& resting = rest_it->second;
            int64_t traded = std::min(incoming.qty, resting.qty);

            incoming.qty -= traded;
            resting.qty -= traded;

            fills.push_back(Fill{resting.id, px, traded, Liquidity::Maker, incoming.ts});

            if (resting.qty == 0) {
                queue.pop_front();
                orders_.erase(rest_it);
            }
        }

        if (queue.empty()) {
            bids_.erase(it);
        }
    }

    return fills;
}
