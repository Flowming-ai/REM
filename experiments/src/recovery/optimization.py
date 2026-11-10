"""Constrained battery recovery with exact physical bounds and lexicographic rescue.

All loads are critical. Monetary cost contains grid purchases only. A counterfactual
objective offset is recorded, but deliberately cannot affect an optimal control.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, vstack


@dataclass(frozen=True)
class BatterySpec:
    capacity_kwh: float = 12.0
    min_energy_kwh: float = 1.2
    max_charge_kw: float = 4.0
    max_discharge_kw: float = 4.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    dt_hours: float = 0.25

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Battery parameters must be finite.")
        if not 0 <= self.min_energy_kwh < self.capacity_kwh:
            raise ValueError("Battery energy bounds must satisfy 0 <= minimum < capacity.")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("Battery power limits cannot be negative.")
        if not 0 < self.eta_charge <= 1 or not 0 < self.eta_discharge <= 1:
            raise ValueError("Battery efficiencies must be in (0, 1].")
        if self.dt_hours <= 0:
            raise ValueError("Time step must be positive.")


@dataclass(frozen=True)
class BatteryForecast:
    load_kw: np.ndarray
    pv_kw: np.ndarray
    price_per_kwh: np.ndarray
    grid_limit_kw: np.ndarray

    def __post_init__(self):
        arrays = []
        for name in ("load_kw", "pv_kw", "price_per_kwh", "grid_limit_kw"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 1 or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be a finite, nonnegative one-dimensional array.")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("All forecast arrays must have the same length.")

    def __len__(self):
        return len(self.load_kw)

    @property
    def horizon(self):
        return len(self)


@dataclass
class BatteryPlan:
    success: bool
    strict_feasible: bool
    status: str
    energy_kwh: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    grid_kw: np.ndarray
    pv_used_kw: np.ndarray
    shed_kw: np.ndarray
    cost: float
    objective_with_offset: float
    runtime_seconds: float
    deadline_shortfall_kwh: float
    terminal_shortfall_kwh: float
    solver_kind: str


def _problem(initial, forecast, spec, target, deadline, rescue):
    n = len(forecast)
    # Variable blocks: E[0:N+1], C, D, G, used-PV, shed, two target shortfalls.
    slices = {"energy": slice(0, n + 1)}
    start = n + 1
    for name in ("charge", "discharge", "grid", "pv", "shed"):
        slices[name] = slice(start, start + n)
        start += n
    sd, st = start, start + 1
    count = start + 2
    lower = np.zeros(count)
    upper = np.full(count, np.inf)
    lower[slices["energy"]] = spec.min_energy_kwh
    upper[slices["energy"]] = spec.capacity_kwh
    lower[0] = upper[0] = initial
    upper[slices["charge"]] = spec.max_charge_kw
    upper[slices["discharge"]] = spec.max_discharge_kw
    upper[slices["grid"]] = forecast.grid_limit_kw
    upper[slices["pv"]] = forecast.pv_kw
    upper[slices["shed"]] = forecast.load_kw if rescue else 0.0
    upper[[sd, st]] = np.inf if rescue else 0.0
    eq = np.zeros((2 * n, count))
    rhs = np.zeros(2 * n)
    for t in range(n):
        eq[t, t + 1] = 1
        eq[t, t] = -1
        eq[t, slices["charge"].start + t] = -spec.eta_charge * spec.dt_hours
        eq[t, slices["discharge"].start + t] = spec.dt_hours / spec.eta_discharge
        eq[n + t, slices["grid"].start + t] = 1
        eq[n + t, slices["pv"].start + t] = 1
        eq[n + t, slices["discharge"].start + t] = 1
        eq[n + t, slices["shed"].start + t] = 1
        eq[n + t, slices["charge"].start + t] = -1
        rhs[n + t] = forecast.load_kw[t]
    ub = np.zeros((2, count))
    ub[0, deadline] = ub[0, sd] = -1
    ub[1, n] = ub[1, st] = -1
    rhs_ub = np.array([-target, -target])
    purchase = np.zeros(count)
    purchase[slices["grid"]] = forecast.price_per_kwh * spec.dt_hours
    service = np.zeros(count)
    service[slices["shed"]] = spec.dt_hours
    shortfall = np.zeros(count)
    shortfall[[sd, st]] = 1
    return slices, lower, upper, csr_matrix(eq), rhs, csr_matrix(ub), rhs_ub, purchase, service, shortfall


def _optimize(problem, spec, n, rescue, use_milp):
    slices, lower, upper, eq, rhs, ub, rhs_ub, purchase, service, shortfall = problem
    width = len(lower)
    if use_milp:
        eq = hstack([eq, csr_matrix((eq.shape[0], n))], format="csr")
        ub = hstack([ub, csr_matrix((ub.shape[0], n))], format="csr")
        modes = np.zeros((2 * n, width + n))
        for t in range(n):
            modes[t, slices["charge"].start + t] = 1
            modes[t, width + t] = -spec.max_charge_kw
            modes[n + t, slices["discharge"].start + t] = 1
            modes[n + t, width + t] = spec.max_discharge_kw
        ub = vstack([ub, csr_matrix(modes)], format="csr")
        rhs_ub = np.r_[rhs_ub, np.zeros(n), np.full(n, spec.max_discharge_kw)]
        lower, upper = np.r_[lower, np.zeros(n)], np.r_[upper, np.ones(n)]
        purchase, service, shortfall = (np.r_[v, np.zeros(n)] for v in (purchase, service, shortfall))
    objectives = [service, shortfall, purchase] if rescue else [purchase]
    result = None
    for stage, objective in enumerate(objectives):
        if use_milp:
            constraints = [LinearConstraint(eq, rhs, rhs), LinearConstraint(ub, -np.inf, rhs_ub)]
            result = milp(objective, integrality=np.r_[np.zeros(width), np.ones(n)],
                          bounds=Bounds(lower, upper), constraints=constraints,
                          options={"mip_rel_gap": 0.0})
        else:
            result = linprog(objective, A_ub=ub, b_ub=rhs_ub, A_eq=eq, b_eq=rhs,
                             bounds=np.column_stack([lower, upper]), method="highs")
        if not result.success:
            return result
        if stage < len(objectives) - 1:
            # Numerical feasibility allowance, not an economic tradeoff weight.
            allowance = 1e-9 * max(1.0, abs(float(result.fun)))
            ub = vstack([ub, csr_matrix(objective[None, :])], format="csr")
            rhs_ub = np.r_[rhs_ub, float(result.fun) + allowance]
    return result


def solve_battery(initial_energy_kwh, forecast: BatteryForecast, spec: BatterySpec, *,
                  target_energy_kwh=9.0, deadline_index=None, allow_rescue=False,
                  objective_offset=0.0) -> BatteryPlan:
    """Solve minimum-purchase-cost recovery with a target at deadline and horizon.

    ``deadline_index`` indexes the state array, so zero means immediately and N
    means after the last interval. Rescue first minimizes total missed critical
    energy, then the sum of deadline/terminal energy shortfalls, then purchases.
    The forecast and all hard plant constraints remain unchanged in rescue.
    """
    started = perf_counter()
    n = len(forecast)
    initial = float(initial_energy_kwh)
    target = float(target_energy_kwh)
    if not np.isfinite(initial) or not spec.min_energy_kwh <= initial <= spec.capacity_kwh:
        raise ValueError("Initial energy must lie within the hard battery bounds.")
    if not np.isfinite(target) or target < 0:
        raise ValueError("Recovery target must be finite and nonnegative.")
    if not np.isfinite(objective_offset):
        raise ValueError("Objective offset must be finite.")
    deadline = n if deadline_index is None else int(deadline_index)
    if deadline_index is not None and deadline != deadline_index:
        raise ValueError("Deadline index must be an integer.")
    if not 0 <= deadline <= n:
        raise ValueError("Deadline index must lie in [0, horizon].")
    strict_feasible = True
    rescue = False
    kind = "highs-lp"
    problem = _problem(initial, forecast, spec, target, deadline, rescue=False)
    result = _optimize(problem, spec, n, rescue=False, use_milp=False)
    if not result.success and result.status == 2 and allow_rescue:
        strict_feasible = False
        rescue = True
        problem = _problem(initial, forecast, spec, target, deadline, rescue=True)
        result = _optimize(problem, spec, n, rescue=True, use_milp=False)
    if result.success:
        slices = problem[0]
        simultaneous = np.minimum(result.x[slices["charge"]], result.x[slices["discharge"]])
        if np.any(simultaneous > 1e-7):
            kind = "highs-milp"
            result = _optimize(problem, spec, n, rescue=rescue, use_milp=True)
            # A relaxation may be feasible when its exact formulation is not.
            if not result.success and result.status == 2 and not rescue and allow_rescue:
                strict_feasible = False
                rescue = True
                problem = _problem(initial, forecast, spec, target, deadline, rescue=True)
                result = _optimize(problem, spec, n, rescue=True, use_milp=True)
    if not result.success:
        energy = np.full(n + 1, np.nan)
        energy[0] = initial
        return BatteryPlan(False, False, str(result.message), energy,
                           *[np.full(n, np.nan) for _ in range(5)],
                           np.nan, np.nan, perf_counter() - started, np.nan, np.nan, kind)
    slices = problem[0]
    vectors = {name: result.x[sl].copy() for name, sl in slices.items()}
    cost = float(np.dot(vectors["grid"], forecast.price_per_kwh) * spec.dt_hours)
    energy = vectors["energy"]
    return BatteryPlan(True, strict_feasible, "optimal_rescue" if rescue else "optimal", energy,
                       vectors["charge"], vectors["discharge"], vectors["grid"], vectors["pv"],
                       vectors["shed"], cost, cost - float(objective_offset),
                       perf_counter() - started, max(0.0, target - energy[deadline]),
                       max(0.0, target - energy[-1]), kind)


def minimum_recovery_steps(initial_energy_kwh, forecast: BatteryForecast, spec: BatterySpec,
                           target_energy_kwh=9.0):
    """Earliest feasible target-entry index, retaining the complete service horizon.

    This is first feasible entry, not a requirement to remain above the target at
    every subsequent step. The target is required again at the terminal state.
    """
    for deadline in range(len(forecast) + 1):
        plan = solve_battery(initial_energy_kwh, forecast, spec,
                             target_energy_kwh=target_energy_kwh, deadline_index=deadline)
        if plan.success:
            return deadline
    return None
