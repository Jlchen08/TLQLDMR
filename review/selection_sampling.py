"""Deterministic training-only label allocation by month and weather event.

The selector never reads power labels, feature errors, validation outcomes,
calibration outcomes, or test rows. It is a distinct, predeclared acquisition
policy from the v1 random sample and must be shared by all compared models.
"""
from __future__ import annotations

import numpy as np


EVENT_ORDER = ("cold", "hot", "wind")


def _time_uniform_indices(indices, times, count):
    """Nearest distinct observations to evenly spaced actual-time targets."""
    indices = np.asarray(indices, dtype=np.int64)
    if count < 1 or count > len(indices):
        raise ValueError("Requested time-uniform count is outside the cell capacity.")
    order = np.argsort(times[indices], kind="stable")
    ordered = indices[order]
    ticks = times[ordered].astype("datetime64[ns]").astype(np.int64)
    if len(ticks) > 1 and np.any(np.diff(ticks) <= 0):
        raise ValueError("Sampling requires unambiguous distinct timestamps within each cell.")
    if count == len(ordered):
        return ordered
    if count == 1:
        target_ticks = [int(ticks[0]) + (int(ticks[-1]) - int(ticks[0])) // 2]
    else:
        span = int(ticks[-1]) - int(ticks[0])
        target_ticks = [int(ticks[0]) + span * j // (count - 1) for j in range(count)]
    chosen = []
    last = -1
    for j, target in enumerate(target_ticks):
        low, high = last + 1, len(ordered) - (count - j)
        insertion = int(np.searchsorted(ticks, target, side="left"))
        candidates = {max(low, min(high, insertion)), max(low, min(high, insertion - 1))}
        position = min(candidates, key=lambda pos: (abs(int(ticks[pos]) - target), pos))
        chosen.append(ordered[position])
        last = position
    return np.asarray(chosen, dtype=np.int64)


def stratified_target_indices(data, budget=128):
    """Return ``(window_indices, audit)`` for the fixed training target budget.

    Cells are training month x exclusive event, with cold preceding hot and
    wind/cut-out assigned last. Every nonempty cell receives one label. Remaining
    labels are proportional to remaining cell capacities, using exact integer
    largest-remainder allocation; ties follow month then cold/hot/wind order.
    Within-cell selection follows equally spaced clock-time targets, with ties
    assigned to the earlier available observation. No labels are inspected.
    """
    if isinstance(budget, bool) or int(budget) != budget or budget < 1:
        raise ValueError("Budget must be a positive integer.")
    budget = int(budget)
    times = np.asarray(data.timestamps, dtype="datetime64[ns]")
    split = np.asarray(data.split)
    extreme = np.asarray(data.is_extreme, dtype=bool)
    if not (len(times) == len(split) == len(extreme)):
        raise ValueError("Temporal/domain arrays have inconsistent lengths.")
    protocol = data.preprocessing["protocol"]
    start, end = np.datetime64(protocol["train_start"], "ns"), np.datetime64(protocol["train_end"], "ns")
    train_mask = split == "train"
    if np.any(train_mask & ((times < start) | (times >= end) | np.isnat(times))):
        raise ValueError("A row marked training lies outside the declared training interval.")
    target = np.flatnonzero(train_mask & extreme)
    if not len(target):
        raise ValueError("No training target observations are available.")
    thresholds = data.preprocessing["domain_thresholds"]
    # Index the train subset before evaluating weather to keep the selection
    # completely independent of all held-out observations.
    wind = np.asarray(data.target_weather["wind_hub"])[target]
    temperature = np.asarray(data.target_weather["temperature"])[target]
    if not (np.isfinite(wind).all() and np.isfinite(temperature).all()):
        raise ValueError("Training target event membership requires finite wind and temperature.")
    cold = temperature < thresholds["temperature_low_c"]
    hot = temperature > thresholds["temperature_high_c"]
    high_wind = (wind > thresholds["wind_ms"]) | (wind > thresholds["cutout_ms"])
    if not np.all(cold | hot | high_wind):
        raise ValueError("Training target membership disagrees with the frozen event thresholds.")
    events = np.where(cold, "cold", np.where(hot, "hot", "wind"))
    months = np.datetime_as_string(times[target], unit="M")
    cells = []
    for month in sorted(set(months.tolist())):
        for event in EVENT_ORDER:
            ids = target[(months == month) & (events == event)]
            if len(ids):
                cells.append({"month": month, "event": event, "ids": ids})
    effective = min(budget, len(target))
    if effective < len(cells):
        raise ValueError(f"Budget {effective} cannot assign at least one label to each of {len(cells)} nonempty cells.")
    capacities = np.asarray([len(cell["ids"]) - 1 for cell in cells], dtype=np.int64)
    remaining = effective - len(cells)
    total_capacity = int(capacities.sum())
    extra = np.zeros(len(cells), dtype=np.int64)
    remainders = np.zeros(len(cells), dtype=np.int64)
    if remaining and total_capacity:
        numerators = remaining * capacities
        extra = numerators // total_capacity
        remainders = numerators % total_capacity
        leftover = remaining - int(extra.sum())
        order = sorted(range(len(cells)), key=lambda i: (-int(remainders[i]), i))
        eligible = [i for i in order if extra[i] < capacities[i]]
        if leftover > len(eligible):
            raise AssertionError("Largest-remainder allocation exceeded cell capacity.")
        extra[eligible[:leftover]] += 1
    quotas = 1 + extra
    selected, cell_audits = [], []
    for i, (cell, quota) in enumerate(zip(cells, quotas)):
        chosen = _time_uniform_indices(cell["ids"], times, int(quota))
        selected.extend(chosen.tolist())
        cell_audits.append({
            "month": cell["month"], "exclusive_event": cell["event"],
            "available": len(cell["ids"]), "selected": int(quota),
            "remaining_capacity": int(capacities[i]), "largest_remainder_numerator": int(remainders[i]),
            "selected_window_indices": chosen.tolist(),
            "selected_original_rows": np.asarray(data.target_row_indices)[chosen].tolist(),
            "selected_times": np.datetime_as_string(times[chosen], unit="s").tolist(),
        })
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(times[selected], kind="stable")]
    if len(selected) != effective or len(np.unique(selected)) != effective:
        raise AssertionError("Stratified selection did not preserve the declared distinct-label budget.")
    audit = {
        "policy": "month_by_exclusive_weather_event_largest_remainder_time_uniform_v1",
        "event_priority": list(EVENT_ORDER), "wind_includes_cutout": True,
        "requested_budget": budget, "effective_budget": effective,
        "available_training_target_count": len(target), "nonempty_cells": len(cells),
        "initial_allocation_per_nonempty_cell": 1, "remaining_budget": remaining,
        "largest_remainder_denominator": total_capacity,
        "tie_breaking": "month ascending, then cold/hot/wind; within time targets choose the earlier equally close observation",
        "within_cell_rule": "nearest distinct observations to equally spaced actual-clock-time targets; single label nearest the temporal midpoint",
        "uses_power_label_values": False, "uses_heldout_rows_or_outcomes": False,
        "train_start": protocol["train_start"], "train_end": protocol["train_end"],
        "thresholds": {key: thresholds[key] for key in
                       ("wind_ms", "temperature_low_c", "temperature_high_c", "cutout_ms")},
        "selected_window_indices": selected.tolist(),
        "selected_original_rows": np.asarray(data.target_row_indices)[selected].tolist(),
        "cells": cell_audits,
    }
    return selected, audit
