// stats.hpp
#pragma once
#include <cstdint>
#include <cmath>

struct SimStats{
    int64_t n_events = 0;
    int64_t n_limit = 0;
    int64_t n_market = 0;
    int64_t n_cancel = 0;
    int64_t traded_qty = 0;

    int64_t spread_samples = 0;
    int64_t spread_sum = 0;

    template <class Book>
    void record_spread(const Book& book){
        auto s = book.spread();
        if (s.has_value()){
            spread_sum += *s;
            spread_samples += 1;
        }
    }

    double avg_spread() const{
        if (spread_samples == 0){
            return std::nan("");
        }
        return static_cast<double>(spread_sum) / static_cast<double>(spread_samples);
    }
};