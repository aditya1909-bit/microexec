#pragma once

#include <cstdint>
#include <utility>

#include "stats.hpp"
#include "book.hpp"
#include "flow.hpp"

std::pair<LimitOrderBook, SimStats>
run_sim(int64_t n_events, const FlowConfig& config, int64_t seed = 0);