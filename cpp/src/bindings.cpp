#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "book.hpp"
#include "engine.hpp"
#include "flow.hpp"
#include "stats.hpp"

PYBIND11_MODULE(lob_cpp, m){
    py::enum_<Side>(m, "Side")
        .value("BID", Side::Bid)
        .value("ASK", Side::Ask)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT", OrderType::Limit)
        .value("MARKET", OrderType::Market)
        .export_values();

    py::enum_<Liquidity>(m, "Liquidity")
        .value("MAKER", Liquidity::Maker)
        .value("TAKER", Liquidity::Taker)
        .export_values();

    py::enum_<EventKind>(m, "EventKind")
        .value("LIMIT", EventKind::Limit)
        .value("MARKET", EventKind::Market)
        .value("CANCEL", EventKind::Cancel)
        .export_values();

    py::class_<Order>(m, "Order")
        .def(
            py::init<int64_t, Side, int64_t, int64_t, int64_t, std::string, OrderType>(),
            py::arg("id"),
            py::arg("side"),
            py::arg("px"),
            py::arg("qty"),
            py::arg("ts"),
            py::arg("owner") = "SIM",
            py::arg("order_type") = OrderType::Limit
        )
        .def_readwrite("id", &Order::id)
        .def_readwrite("side", &Order::side)
        .def_readwrite("px", &Order::px)
        .def_readwrite("qty", &Order::qty)
        .def_readwrite("ts", &Order::ts)
        .def_readwrite("owner", &Order::owner)
        .def_readwrite("order_type", &Order::order_type);

    py::class_<Fill>(m, "Fill")
        .def_readwrite("order_id", &Fill::order_id)
        .def_readwrite("px", &Fill::px)
        .def_readwrite("qty", &Fill::qty)
        .def_readwrite("liquidity", &Fill::liquidity)
        .def_readwrite("ts", &Fill::ts);

    py::class_<LimitOrderBook>(m, "LimitOrderBook")
        .def(py::init<>())
        .def("best_bid", &LimitOrderBook::best_bid)
        .def("best_ask", &LimitOrderBook::best_ask)
        .def("mid", &LimitOrderBook::mid)
        .def("spread", &LimitOrderBook::spread)
        .def("depth", &LimitOrderBook::depth, py::arg("side"), py::arg("levels") = 5)
        .def("imbalance", &LimitOrderBook::imbalance, py::arg("levels") = 1)
        .def("add_limit", &LimitOrderBook::add_limit)
        .def("add_market", &LimitOrderBook::add_market, py::arg("side"), py::arg("qty"), py::arg("ts"))
        .def("cancel", &LimitOrderBook::cancel, py::arg("order_id"), py::arg("ts"));

    py::class_<FlowConfig>(m, "FlowConfig")
        .def(py::init<>())
        .def_readwrite("initial_mid", &FlowConfig::initial_mid)
        .def_readwrite("dt_us", &FlowConfig::dt_us)
        .def_readwrite("p_limit", &FlowConfig::p_limit)
        .def_readwrite("p_market", &FlowConfig::p_market)
        .def_readwrite("p_cancel", &FlowConfig::p_cancel)
        .def_readwrite("cancel_levels", &FlowConfig::cancel_levels)
        .def_readwrite("soft_max_orders", &FlowConfig::soft_max_orders)
        .def_readwrite("hard_max_orders", &FlowConfig::hard_max_orders)
        .def_readwrite("cancel_boost", &FlowConfig::cancel_boost)
        .def_readwrite("min_offset", &FlowConfig::min_offset)
        .def_readwrite("max_offset", &FlowConfig::max_offset)
        .def_readwrite("offset_power", &FlowConfig::offset_power)
        .def_readwrite("min_qty", &FlowConfig::min_qty)
        .def_readwrite("max_qty", &FlowConfig::max_qty)
        .def_readwrite("min_market_qty", &FlowConfig::min_market_qty)
        .def_readwrite("max_market_qty", &FlowConfig::max_market_qty)
        .def_readwrite("min_touch_qty", &FlowConfig::min_touch_qty)
        .def_readwrite("cancel_fallback_to_limit", &FlowConfig::cancel_fallback_to_limit)
        .def("validate", &FlowConfig::validate);

    py::class_<SimStats>(m, "SimStats")
        .def(py::init<>())
        .def_readwrite("n_events", &SimStats::n_events)
        .def_readwrite("n_limit", &SimStats::n_limit)
        .def_readwrite("n_market", &SimStats::n_market)
        .def_readwrite("n_cancel", &SimStats::n_cancel)
        .def_readwrite("traded_qty", &SimStats::traded_qty)
        .def_readwrite("spread_samples", &SimStats::spread_samples)
        .def_readwrite("spread_sum", &SimStats::spread_sum)
        .def("avg_spread", &SimStats::avg_spread);

    // Smoke-test function to verify Python -> C++ bindings work.
    m.def("run_sim_stub", [](int64_t n_events) {
        SimStats s;
        s.n_events = n_events;
        s.n_limit = n_events;
        return s;
    }, py::arg("n_events"));

    m.def("run_sim", &run_sim, py::arg("n_events"), py::arg("config"), py::arg("seed") = 0);
}
