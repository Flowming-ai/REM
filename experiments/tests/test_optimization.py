import numpy as np
import pytest

from recovery.optimization import (BatteryForecast, BatterySpec, minimum_recovery_steps,
                                   solve_battery)


def forecast(n, load=0., pv=0., price=0.2, grid=8.):
    def a(value):
        return np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value)
    return BatteryForecast(a(load), a(pv), a(price), a(grid))


def assert_physics(plan, f, s):
    assert plan.success
    np.testing.assert_allclose(np.diff(plan.energy_kwh),
        (s.eta_charge * plan.charge_kw - plan.discharge_kw / s.eta_discharge) * s.dt_hours,
        atol=1e-6)
    np.testing.assert_allclose(plan.grid_kw + plan.pv_used_kw + plan.discharge_kw + plan.shed_kw,
                               f.load_kw + plan.charge_kw, atol=1e-6)
    assert np.min(plan.energy_kwh) >= s.min_energy_kwh - 1e-6
    assert np.max(plan.energy_kwh) <= s.capacity_kwh + 1e-6
    assert np.all(np.minimum(plan.charge_kw, plan.discharge_kw) <= 1e-7)
    assert np.all(plan.grid_kw <= f.grid_limit_kw + 1e-6)
    assert np.all(plan.pv_used_kw <= f.pv_kw + 1e-6)


def test_hand_calculated_one_step_cost():
    s = BatterySpec()
    f = forecast(1, price=.3)
    result = solve_battery(5, f, s, target_energy_kwh=5.5)
    assert_physics(result, f, s)
    assert result.strict_feasible
    assert result.cost == pytest.approx(.5 / .95 * .3, abs=1e-8)
    assert result.energy_kwh[-1] == pytest.approx(5.5)


def test_deadline_infeasible_and_minimum_recovery_time():
    s = BatterySpec()
    f = forecast(4)
    # At most 0.95 kWh can be charged in one 15-minute slot.
    p = solve_battery(5, f, s, target_energy_kwh=7, deadline_index=2)
    assert not p.success
    assert not p.strict_feasible
    assert minimum_recovery_steps(5, f, s, 7) == 3
    assert minimum_recovery_steps(5, forecast(2), s, 7) is None


def test_immediate_deadline_is_enforced():
    s = BatterySpec()
    f = forecast(4)
    assert not solve_battery(5, f, s, target_energy_kwh=6, deadline_index=0).success
    assert minimum_recovery_steps(7, f, s, 6) == 0


def test_no_simultaneous_charge_discharge_and_balance():
    s = BatterySpec()
    f = forecast(8, load=[1, 2, 5, 2, 1, 4, 3, 2],
                 pv=[5, 4, 0, 0, 6, 0, 1, 0],
                 price=[0, 0, .4, .4, 0, .5, .2, .1], grid=5)
    p = solve_battery(7, f, s, target_energy_kwh=7)
    assert_physics(p, f, s)
    assert np.sum(p.shed_kw) == pytest.approx(0)
    assert p.cost == pytest.approx(np.dot(p.grid_kw, f.price_per_kwh) * s.dt_hours)


def test_constant_offset_changes_no_controls():
    s = BatterySpec()
    f = forecast(6, load=[2, 3, 4, 3, 2, 1], price=[.1, .2, .5, .4, .3, .15])
    p = solve_battery(6, f, s, target_energy_kwh=7)
    q = solve_battery(6, f, s, target_energy_kwh=7, objective_offset=42.5)
    for name in ("energy_kwh", "charge_kw", "discharge_kw", "grid_kw", "shed_kw"):
        np.testing.assert_array_equal(getattr(p, name), getattr(q, name))
    assert p.cost == q.cost
    assert q.objective_with_offset == pytest.approx(p.cost - 42.5)


def test_rescue_serves_load_before_restoring_target():
    s = BatterySpec()
    f = forecast(1, load=4, grid=2, price=100)
    strict = solve_battery(3, f, s, target_energy_kwh=3)
    assert not strict.success
    rescue = solve_battery(3, f, s, target_energy_kwh=3, allow_rescue=True)
    assert_physics(rescue, f, s)
    assert not rescue.strict_feasible
    assert np.sum(rescue.shed_kw) * s.dt_hours < 1e-7
    assert rescue.deadline_shortfall_kwh == pytest.approx(.5 / .95, abs=1e-6)
    assert rescue.terminal_shortfall_kwh == pytest.approx(.5 / .95, abs=1e-6)
    assert rescue.cost == pytest.approx(50, abs=1e-5)


def test_rescue_reports_unavoidable_missed_energy():
    s = BatterySpec()
    f = forecast(1, load=4, grid=2)
    p = solve_battery(s.min_energy_kwh, f, s, target_energy_kwh=3, allow_rescue=True)
    assert_physics(p, f, s)
    assert p.shed_kw[0] * s.dt_hours == pytest.approx(.5, abs=1e-6)
    assert p.energy_kwh[-1] == pytest.approx(s.min_energy_kwh, abs=1e-6)


def test_reject_bad_forecasts():
    with pytest.raises(ValueError):
        BatteryForecast([1, 2], [0], [1, 1], [5, 5])
    with pytest.raises(ValueError):
        forecast(1, price=-1)
