#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "book.hpp"
#include "execution.hpp"
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
        .value("IOC", OrderType::IOC)
        .value("FOK", OrderType::FOK)
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

    py::enum_<HistoricalFormat>(m, "HistoricalFormat")
        .value("BINARY", HistoricalFormat::Binary)
        .value("CSV", HistoricalFormat::Csv)
        .value("TRUEFX", HistoricalFormat::TrueFX)
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
        .def_readwrite("p_ioc", &FlowConfig::p_ioc)
        .def_readwrite("p_fok", &FlowConfig::p_fok)
        .def("validate", &FlowConfig::validate);

    py::class_<PoissonOrderFlow>(m, "PoissonOrderFlow")
        .def(py::init<const FlowConfig&, int64_t>(), py::arg("config"), py::arg("seed") = 0)
        .def("step", &PoissonOrderFlow::step, py::arg("book"), py::arg("ts"))
        .def("step_n", &PoissonOrderFlow::step_n, py::arg("book"), py::arg("start_ts"), py::arg("n_events"));

    py::class_<HawkesConfig>(m, "HawkesConfig")
        .def(py::init<>())
        .def_readwrite("base", &HawkesConfig::base)
        .def_readwrite("alpha_limit", &HawkesConfig::alpha_limit)
        .def_readwrite("alpha_market", &HawkesConfig::alpha_market)
        .def_readwrite("alpha_cancel", &HawkesConfig::alpha_cancel)
        .def_readwrite("decay", &HawkesConfig::decay);

    py::class_<HawkesOrderFlow>(m, "HawkesOrderFlow")
        .def(py::init<const HawkesConfig&, int64_t>(), py::arg("config"), py::arg("seed") = 0)
        .def("step", &HawkesOrderFlow::step, py::arg("book"), py::arg("ts"));

    py::class_<HistoricalOrderFlow>(m, "HistoricalOrderFlow")
        .def(
            py::init<const std::string&, HistoricalFormat, int64_t, int64_t, int64_t>(),
            py::arg("path"),
            py::arg("format"),
            py::arg("start_order_id") = 1,
            py::arg("fixed_qty") = 1'000'000,
            py::arg("price_scale") = 100'000
        )
        .def("done", &HistoricalOrderFlow::done)
        .def("step", [](HistoricalOrderFlow& flow, LimitOrderBook& book) {
            int64_t ts = 0;
            int64_t traded = 0;
            bool ok = flow.step(book, ts, traded);
            return py::make_tuple(ok, ts, traded);
        });

    py::class_<TwapReport>(m, "TwapReport")
        .def_readwrite("side", &TwapReport::side)
        .def_readwrite("target_qty", &TwapReport::target_qty)
        .def_readwrite("filled_qty", &TwapReport::filled_qty)
        .def_readwrite("avg_fill_px", &TwapReport::avg_fill_px)
        .def_readwrite("arrival_mid", &TwapReport::arrival_mid)
        .def_readwrite("shortfall", &TwapReport::shortfall)
        .def_readwrite("n_child_orders", &TwapReport::n_child_orders)
        .def_readwrite("unfilled_qty", &TwapReport::unfilled_qty)
        .def_readwrite("completion_rate", &TwapReport::completion_rate)
        .def_readwrite("penalty_per_share", &TwapReport::penalty_per_share)
        .def_readwrite("penalized_cost", &TwapReport::penalized_cost)
        .def_readwrite("shortfall_per_share", &TwapReport::shortfall_per_share)
        .def_readwrite("penalized_cost_per_share", &TwapReport::penalized_cost_per_share);

    py::class_<VwapReport>(m, "VwapReport")
        .def_readwrite("side", &VwapReport::side)
        .def_readwrite("target_qty", &VwapReport::target_qty)
        .def_readwrite("filled_qty", &VwapReport::filled_qty)
        .def_readwrite("avg_fill_px", &VwapReport::avg_fill_px)
        .def_readwrite("arrival_mid", &VwapReport::arrival_mid)
        .def_readwrite("shortfall", &VwapReport::shortfall)
        .def_readwrite("n_child_orders", &VwapReport::n_child_orders)
        .def_readwrite("unfilled_qty", &VwapReport::unfilled_qty)
        .def_readwrite("completion_rate", &VwapReport::completion_rate)
        .def_readwrite("penalty_per_share", &VwapReport::penalty_per_share)
        .def_readwrite("penalized_cost", &VwapReport::penalized_cost)
        .def_readwrite("shortfall_per_share", &VwapReport::shortfall_per_share)
        .def_readwrite("penalized_cost_per_share", &VwapReport::penalized_cost_per_share)
        .def_readwrite("n_buckets", &VwapReport::n_buckets)
        .def_readwrite("bucket_interval", &VwapReport::bucket_interval)
        .def_readwrite("forecast_total_mkt_vol", &VwapReport::forecast_total_mkt_vol);

    py::class_<ImpactModel>(m, "ImpactModel")
        .def(py::init<>())
        .def_readwrite("temporary_impact", &ImpactModel::temporary_impact)
        .def_readwrite("permanent_impact", &ImpactModel::permanent_impact)
        .def_readwrite("volatility", &ImpactModel::volatility)
        .def_readwrite("risk_aversion", &ImpactModel::risk_aversion);

    py::class_<AlmgrenChrissReport>(m, "AlmgrenChrissReport")
        .def_readwrite("side", &AlmgrenChrissReport::side)
        .def_readwrite("target_qty", &AlmgrenChrissReport::target_qty)
        .def_readwrite("filled_qty", &AlmgrenChrissReport::filled_qty)
        .def_readwrite("avg_fill_px", &AlmgrenChrissReport::avg_fill_px)
        .def_readwrite("arrival_mid", &AlmgrenChrissReport::arrival_mid)
        .def_readwrite("shortfall", &AlmgrenChrissReport::shortfall)
        .def_readwrite("n_child_orders", &AlmgrenChrissReport::n_child_orders)
        .def_readwrite("unfilled_qty", &AlmgrenChrissReport::unfilled_qty)
        .def_readwrite("completion_rate", &AlmgrenChrissReport::completion_rate)
        .def_readwrite("risk_penalty", &AlmgrenChrissReport::risk_penalty)
        .def_readwrite("objective", &AlmgrenChrissReport::objective)
        .def_readwrite("shortfall_uncapped", &AlmgrenChrissReport::shortfall_uncapped)
        .def_readwrite("objective_uncapped", &AlmgrenChrissReport::objective_uncapped)
        .def_readwrite("cap_abs", &AlmgrenChrissReport::cap_abs)
        .def_readwrite("cap_frac", &AlmgrenChrissReport::cap_frac)
        .def_readwrite("cap_used", &AlmgrenChrissReport::cap_used)
        .def_readwrite("capped", &AlmgrenChrissReport::capped)
        .def_readwrite("child_qtys", &AlmgrenChrissReport::child_qtys)
        .def_readwrite("impact", &AlmgrenChrissReport::impact);

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
    m.def(
        "run_twap",
        &run_twap,
        py::arg("side"),
        py::arg("total_qty"),
        py::arg("horizon_events"),
        py::arg("child_interval"),
        py::arg("cfg"),
        py::arg("latency_us") = 0,
        py::arg("seed") = 0,
        py::arg("warmup_events") = 500,
        py::arg("penalty_per_share") = 0.0
    );
    m.def(
        "run_vwap",
        &run_vwap,
        py::arg("side"),
        py::arg("total_qty"),
        py::arg("horizon_events"),
        py::arg("bucket_interval"),
        py::arg("cfg"),
        py::arg("latency_us") = 0,
        py::arg("seed") = 0,
        py::arg("warmup_events") = 500,
        py::arg("penalty_per_share") = 0.0
    );
    m.def(
        "run_almgren_chriss",
        &run_almgren_chriss,
        py::arg("side"),
        py::arg("total_qty"),
        py::arg("horizon_events"),
        py::arg("child_interval"),
        py::arg("cfg"),
        py::arg("impact"),
        py::arg("cap_abs") = 0,
        py::arg("cap_frac") = 0.0,
        py::arg("latency_us") = 0,
        py::arg("seed") = 0,
        py::arg("warmup_events") = 500
    );
}
