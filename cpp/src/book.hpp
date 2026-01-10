#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

enum class Side { Bid, Ask };
enum class OrderType { Limit, Market, IOC, FOK };
enum class Liquidity { Maker, Taker };

struct Order {
    int64_t id;
    Side side;
    int64_t px;
    int64_t qty;
    int64_t ts;
    std::string owner = "SIM";
    OrderType order_type = OrderType::Limit;
};

struct Fill {
    int64_t order_id;
    int64_t px;
    int64_t qty;
    Liquidity liquidity;
    int64_t ts;
};

class LimitOrderBook {
public:
    LimitOrderBook() = default;

    std::optional<int64_t> best_bid() const;
    std::optional<int64_t> best_ask() const;
    std::optional<double> mid() const;
    std::optional<int64_t> spread() const;
    std::vector<std::pair<int64_t, int64_t>> depth(Side side, int levels = 5) const;
    double imbalance(int levels = 1) const;

    std::vector<Fill> add_limit(Order order);
    std::vector<Fill> add_market(Side side, int64_t qty, int64_t ts);
    bool cancel(int64_t order_id, int64_t ts);

    size_t order_count() const;
    std::vector<int64_t> all_order_ids() const;
    std::vector<int64_t> order_ids_at_price(Side side, int64_t px) const;

private:
    std::map<int64_t, std::deque<int64_t>, std::greater<int64_t>> bids_;
    std::map<int64_t, std::deque<int64_t>> asks_;
    std::unordered_map<int64_t, Order> orders_;

    int64_t available_qty_for_limit(const Order& order) const;
    void add_to_book(const Order& order);
    std::vector<Fill> match(Order& incoming, Side resting_side);
};
