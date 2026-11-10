"""Independent, ex-post AC checks of executed recovery trajectories.

The dispatch model owns energy/SOC evolution and service feasibility. This
module only maps executed, served powers onto a small pandapower feeder and
checks grid-connected balanced steady-state snapshots. It is not AC-OPF,
islanded inverter simulation, or a proof of safety between sample times.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ACValidationConfig:
    """Declared feeder assumptions; all limits are experimental choices.

    ``line_length_scale=1`` preserves the bundled network's two 0.5 km lines.
    Changing it creates an explicitly adapted feeder, not the stock benchmark.
    The load is divided equally between its two original load buses; PV has
    the same split. The battery is placed at the most remote load bus.
    """

    line_length_scale: float = 1.0
    load_power_factor: float = 0.97
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    loading_max_percent: float = 100.0
    storage_capacity_kwh: float = 100.0
    power_tolerance_kw: float = 1e-5
    grid_voltage_pu: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("AC configuration values must be finite.")
        if self.line_length_scale <= 0 or self.storage_capacity_kwh <= 0:
            raise ValueError("Line-length scale and storage capacity must be positive.")
        if not 0 < self.load_power_factor <= 1:
            raise ValueError("Load power factor must lie in (0, 1].")
        if not 0 < self.voltage_min_pu < self.voltage_max_pu:
            raise ValueError("Voltage limits must be positive and ordered.")
        if self.loading_max_percent <= 0 or self.power_tolerance_kw < 0:
            raise ValueError("Loading limit must be positive; tolerance nonnegative.")
        if self.grid_voltage_pu <= 0:
            raise ValueError("Grid voltage must be positive.")


_REQUIRED_POWERS = (
    "load_served_kw",
    "pv_used_kw",
    "battery_charge_kw",
    "battery_discharge_kw",
)
_OUTPUT_COLUMNS = (
    "ac_status",
    "ac_converged",
    "ac_min_voltage_pu",
    "ac_max_voltage_pu",
    "ac_max_line_loading_percent",
    "ac_max_trafo_loading_percent",
    "ac_grid_kw",
    "ac_export_kw",
    "ac_loss_kw",
    "ac_ideal_grid_kw",
    "ac_grid_cap_excess_kw",
    "ac_power_balance_error_kw",
    "ac_voltage_violation",
    "ac_loading_violation",
    "ac_grid_cap_violation",
    "ac_network_passed",
    "ac_passed",
)


def _build_network(config: ACValidationConfig):
    """Create, then explicitly adapt the public four-bus example network."""
    import pandapower as pp
    from pandapower.networks import simple_four_bus_system

    net = simple_four_bus_system()
    net.line.loc[:, "length_km"] *= config.line_length_scale
    net.ext_grid.loc[:, "vm_pu"] = config.grid_voltage_pu
    net.sgen.loc[:, "q_mvar"] = 0.0
    net.sgen.loc[:, "p_mw"] = 0.0
    net.load.loc[:, "p_mw"] = 0.0
    net.load.loc[:, "q_mvar"] = 0.0
    terminal_bus = int(net.load.iloc[-1].bus)
    storage_id = pp.create_storage(
        net,
        bus=terminal_bus,
        p_mw=0.0,
        q_mvar=0.0,
        max_e_mwh=config.storage_capacity_kwh / 1000.0,
        name="recovery_battery",
    )
    return net, storage_id


def validate_ac_dataframe(
    traces: pd.DataFrame,
    *,
    config: ACValidationConfig | None = None,
) -> pd.DataFrame:
    """Return input rows plus transparent AC validation columns.

    Required columns, in kW: ``load_served_kw``, ``pv_used_kw``,
    ``battery_charge_kw`` and ``battery_discharge_kw``. These must contain
    *executed* powers after device clipping, PV curtailment and load shedding.
    Both battery columns are nonnegative. Demand not served must not be passed
    as a load: report it separately in the experiment's service metrics.

    Optional ``grid_available`` is a boolean connection flag. If omitted,
    ``grid_limit_kw <= power_tolerance_kw`` marks an islanded snapshot; absent
    both columns, all snapshots are treated as grid connected. Islanded rows
    receive ``ac_status='not_checked_islanded'`` and missing physical results;
    they do not receive a false pass. ``grid_limit_kw`` otherwise declares a
    source-side import cap and is checked against the AC slack import, which
    includes feeder losses. This may exceed a lossless dispatch balance cap.

    ``grid_import_kw`` and metadata such as ``seed``, ``scenario_id``,
    ``method``, ``t`` and ``dt_hours`` are preserved, not changed or inferred.
    No storage SOC is evolved by pandapower; the dispatch simulator owns it.
    Network-only pass and combined pass (including import cap) are separate.
    A missing import cap means the combined pass covers network checks only.
    """
    import pandapower as pp

    cfg = config or ACValidationConfig()
    absent = set(_REQUIRED_POWERS) - set(traces.columns)
    if absent:
        raise ValueError(f"Missing executed power columns: {sorted(absent)}")
    if set(_OUTPUT_COLUMNS) & set(traces.columns):
        raise ValueError("Input already contains AC validation output columns.")
    numeric = traces[list(_REQUIRED_POWERS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Executed powers must be finite.")
    if (numeric < -cfg.power_tolerance_kw).any():
        raise ValueError("Required power columns must be nonnegative.")
    if "grid_limit_kw" in traces:
        caps = traces.grid_limit_kw.to_numpy(dtype=float)
        if not np.isfinite(caps).all() or (caps < 0).any():
            raise ValueError("Grid limits must be finite and nonnegative.")
    else:
        caps = np.full(len(traces), np.nan)
    if "grid_available" in traces:
        flags = traces.grid_available
        if flags.isna().any() or not flags.isin([True, False, 0, 1]).all():
            raise ValueError("grid_available must contain boolean flags.")
        connected = flags.to_numpy(dtype=bool)
    else:
        connected = np.isnan(caps) | (caps > cfg.power_tolerance_kw)

    net, storage_id = _build_network(cfg)
    reactive_multiplier = np.tan(np.arccos(cfg.load_power_factor))
    records: list[dict] = []
    for powers, cap, is_connected in zip(numeric, caps, connected, strict=True):
        served, pv, charge, discharge = np.maximum(powers, 0.0)
        ideal_grid = float(served + charge - discharge - pv)
        record = {name: np.nan for name in _OUTPUT_COLUMNS}
        record.update(ac_ideal_grid_kw=ideal_grid)
        if not is_connected:
            record.update(ac_status="not_checked_islanded", ac_converged=False)
            records.append(record)
            continue

        net.load.loc[:, "p_mw"] = served / 1000.0 / len(net.load)
        net.load.loc[:, "q_mvar"] = (
            served / 1000.0 / len(net.load) * reactive_multiplier
        )
        net.sgen.loc[:, "p_mw"] = pv / 1000.0 / len(net.sgen)
        # pandapower's storage convention is positive charging, not injection.
        net.storage.at[storage_id, "p_mw"] = (charge - discharge) / 1000.0
        try:
            pp.runpp(
                net,
                algorithm="nr",
                init="flat",
                calculate_voltage_angles=False,
                numba=False,
                max_iteration=30,
                tolerance_mva=1e-9,
            )
        except pp.LoadflowNotConverged:
            record.update(
                ac_status="power_flow_not_converged",
                ac_converged=False,
                ac_network_passed=False,
                ac_passed=False,
            )
            records.append(record)
            continue

        voltages = net.res_bus.vm_pu.to_numpy(dtype=float)
        if not np.isfinite(voltages).all():
            record.update(
                ac_status="unsupplied_or_nonfinite_bus",
                ac_converged=False,
                ac_network_passed=False,
                ac_passed=False,
            )
            records.append(record)
            continue
        min_voltage, max_voltage = float(voltages.min()), float(voltages.max())
        line_loading = float(net.res_line.loading_percent.max())
        trafo_loading = float(net.res_trafo.loading_percent.max())
        grid_power = float(net.res_ext_grid.p_mw.sum() * 1000.0)
        losses = float((net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * 1000.0)
        excess = float(max(0.0, grid_power - cap)) if np.isfinite(cap) else np.nan
        voltage_violation = bool(
            min_voltage < cfg.voltage_min_pu - 1e-8
            or max_voltage > cfg.voltage_max_pu + 1e-8
        )
        loading_violation = bool(
            max(line_loading, trafo_loading) > cfg.loading_max_percent + 1e-6
        )
        cap_violation = bool(np.isfinite(cap) and excess > cfg.power_tolerance_kw)
        network_passed = not (voltage_violation or loading_violation)
        record.update(
            ac_status="checked_grid_connected",
            ac_converged=True,
            ac_min_voltage_pu=min_voltage,
            ac_max_voltage_pu=max_voltage,
            ac_max_line_loading_percent=line_loading,
            ac_max_trafo_loading_percent=trafo_loading,
            ac_grid_kw=grid_power,
            ac_export_kw=max(0.0, -grid_power),
            ac_loss_kw=losses,
            ac_grid_cap_excess_kw=excess,
            ac_power_balance_error_kw=grid_power - ideal_grid - losses,
            ac_voltage_violation=voltage_violation,
            ac_loading_violation=loading_violation,
            ac_grid_cap_violation=cap_violation,
            ac_network_passed=network_passed,
            ac_passed=network_passed and not cap_violation,
        )
        records.append(record)

    out = traces.copy()
    results = pd.DataFrame(records, columns=_OUTPUT_COLUMNS, index=out.index)
    for column in _OUTPUT_COLUMNS:
        out[column] = results[column]
    out.attrs["ac_validation"] = {
        "pandapower_version": pp.__version__,
        "network": "pandapower.networks.simple_four_bus_system",
        "configuration": asdict(cfg),
        "scope": "ex_post_balanced_AC_grid_connected_snapshots",
        "line_lengths_km": net.line.length_km.tolist(),
        "load_split": [0.5, 0.5],
        "pv_split": [0.5, 0.5],
        "battery_bus": int(net.storage.at[storage_id, "bus"]),
    }
    return out


def validate_ac_trajectory(
    load_kw: Sequence[float],
    pv_kw: Sequence[float],
    battery_kw: Sequence[float],
    *,
    grid_available: Sequence[bool] | None = None,
    grid_limit_kw: Sequence[float] | None = None,
    config: ACValidationConfig | None = None,
) -> pd.DataFrame:
    """Array interface; positive ``battery_kw`` discharges, negative charges.

    ``load_kw`` is served load and ``pv_kw`` is utilized PV. To preserve
    experiment identifiers, use :func:`validate_ac_dataframe` instead.
    """
    battery = np.asarray(battery_kw, dtype=float)
    if battery.ndim != 1:
        raise ValueError("battery_kw must be one-dimensional.")
    frame = pd.DataFrame(
        {
            "t": np.arange(len(battery)),
            "load_served_kw": load_kw,
            "pv_used_kw": pv_kw,
            "battery_charge_kw": np.maximum(-battery, 0.0),
            "battery_discharge_kw": np.maximum(battery, 0.0),
        }
    )
    if grid_available is not None:
        frame["grid_available"] = grid_available
    if grid_limit_kw is not None:
        frame["grid_limit_kw"] = grid_limit_kw
    return validate_ac_dataframe(frame, config=config)
