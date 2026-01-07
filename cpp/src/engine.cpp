#include "engine.hpp"
#include "book.hpp"
#include "flow.hpp"

std::pair<LimitOrderBook, SimStats>
run_sim(int64_t n_events, const FlowConfig& config, int64_t seed){
    LimitOrderBook book;
    PoissonOrderFlow flow(config, seed);
    SimStats stats;

    int64_t ts = 0;

    for (int64_t i = 0; i < n_events; ++i){
        ts += config.dt_us;

        auto [kind, traded] = flow.step(book, ts);

        stats.n_events += 1;
        stats.traded_qty += traded;

        if (kind == EventKind::Limit) stats.n_limit += 1;
        else if (kind == EventKind::Market) stats.n_market += 1;
        else if (kind == EventKind::Cancel) stats.n_cancel += 1;

        stats.record_spread(book);
    }

    return {book, stats};
}