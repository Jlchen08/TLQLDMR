#!/usr/bin/env python3
"""Summarize the predeclared meteorological-tail domain without model fitting.

The analysis uses the immutable chronological caches.  An episode is a maximal
run of target-domain forecast timestamps separated by exactly 15 minutes.  The
longest episode at each site (earliest in a tie) is used as the objective
representative episode; no prediction errors enter the selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from review.selection_data import load_all_selection_data


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review" / "extreme_event_summary.json"
TEX = ROOT / "TLQLDMR_R1" / "event_composition.tex"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def episode_records(times: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict]:
    indices = np.flatnonzero(masks["union"])
    if not len(indices):
        return []
    target_times = times[indices].astype("datetime64[ns]")
    starts = np.r_[0, np.flatnonzero(np.diff(target_times) != np.timedelta64(15, "m")) + 1]
    stops = np.r_[starts[1:], len(indices)]
    records = []
    for lo, hi in zip(starts, stops):
        rows = indices[lo:hi]
        first = pd.Timestamp(times[rows[0]])
        last = pd.Timestamp(times[rows[-1]])
        records.append({
            "start": first.isoformat(),
            "end": last.isoformat(),
            "samples": int(len(rows)),
            "duration_hours": float(len(rows) * 0.25),
            "rule_counts": {name: int(values[rows].sum()) for name, values in masks.items() if name != "union"},
        })
    return records


def analyze_site(site: int, data) -> dict:
    thresholds = data.preprocessing["domain_thresholds"]
    wind = np.asarray(data.target_weather["wind_hub"])
    temperature = np.asarray(data.target_weather["temperature"])
    masks = {
        "wind": wind > thresholds["wind_ms"],
        "cold": temperature < thresholds["temperature_low_c"],
        "hot": temperature > thresholds["temperature_high_c"],
        "cutout": wind > thresholds["cutout_ms"],
    }
    masks["union"] = masks["wind"] | masks["cold"] | masks["hot"] | masks["cutout"]
    if not np.array_equal(masks["union"], np.asarray(data.is_extreme)):
        raise AssertionError(f"Site {site}: reconstructed union differs from frozen target mask")
    if np.any(masks["cutout"] & ~masks["wind"]):
        raise AssertionError(f"Site {site}: 25 m/s rule is not nested in percentile wind rule")

    timestamps = np.asarray(data.timestamps).astype("datetime64[ns]")
    episodes = episode_records(timestamps, masks)
    durations = np.asarray([item["duration_hours"] for item in episodes], dtype=float)
    longest = sorted(episodes, key=lambda item: (-item["samples"], item["start"]))[0]
    months = pd.DatetimeIndex(timestamps[masks["union"]]).month
    seasonal_counts = {name: int(sum(season(int(month)) == name for month in months))
                       for name in ("DJF", "MAM", "JJA", "SON")}
    union_count = int(masks["union"].sum())
    rule_episodes = {}
    for name in ("wind", "cold", "hot", "cutout"):
        values = episode_records(timestamps, {name: masks[name], "union": masks[name]})
        if values:
            rule_durations = np.asarray([item["duration_hours"] for item in values], dtype=float)
            rule_episodes[name] = {
                "count": int(len(values)),
                "median_duration_hours": float(np.median(rule_durations)),
                "p95_duration_hours": float(np.quantile(rule_durations, 0.95)),
                "max_duration_hours": float(rule_durations.max()),
                "representative_longest": sorted(values, key=lambda item: (-item["samples"], item["start"]))[0],
            }
        else:
            rule_episodes[name] = None
    return {
        "site": site,
        "eligible_windows": int(len(timestamps)),
        "thresholds": {key: float(value) for key, value in thresholds.items()
                       if key in ("wind_ms", "temperature_low_c", "temperature_high_c", "cutout_ms")},
        "rule_counts": {name: int(masks[name].sum()) for name in ("wind", "cold", "hot", "cutout")},
        "overlaps": {
            "wind_and_cold": int((masks["wind"] & masks["cold"]).sum()),
            "wind_and_hot": int((masks["wind"] & masks["hot"]).sum()),
            "cold_and_hot": int((masks["cold"] & masks["hot"]).sum()),
            "wind_and_temperature": int((masks["wind"] & (masks["cold"] | masks["hot"])).sum()),
            "multiple_base_rules": int(((masks["wind"].astype(int) + masks["cold"].astype(int) + masks["hot"].astype(int)) >= 2).sum()),
        },
        "union_count": union_count,
        "union_percent": 100.0 * union_count / len(timestamps),
        "seasonal_counts": seasonal_counts,
        "seasonal_percent_of_union": {key: 100.0 * value / union_count for key, value in seasonal_counts.items()},
        "episodes": {
            "count": int(len(episodes)),
            "median_duration_hours": float(np.median(durations)),
            "p95_duration_hours": float(np.quantile(durations, 0.95)),
            "max_duration_hours": float(durations.max()),
            "representative_selection": "longest episode; earliest start breaks a duration tie; prediction errors are not used",
            "representative_longest": longest,
        },
        "rule_episodes": rule_episodes,
    }


def fmt_int(value: int) -> str:
    return f"{value:,}"


def rules_for_episode(item: dict) -> str:
    labels = [("W", "wind"), ("C", "cold"), ("H", "hot"), (">25", "cutout")]
    return "+".join(label for label, key in labels if item["rule_counts"][key]) or "---"


def render_tex(rows: list[dict]) -> str:
    count_rows = []
    episode_rows = []
    for row in rows:
        rc, ov = row["rule_counts"], row["overlaps"]
        count_rows.append(
            f"{row['site']} & {fmt_int(row['eligible_windows'])} & {fmt_int(rc['wind'])} & "
            f"{fmt_int(rc['cold'])} & {fmt_int(rc['hot'])} & {fmt_int(rc['cutout'])} & "
            f"{fmt_int(ov['wind_and_cold'])} & {fmt_int(ov['wind_and_hot'])} & "
            f"{fmt_int(row['union_count'])} "
            f"({row['union_percent']:.2f}\\%) \\\\"
        )
        ep = row["episodes"]
        rep = ep["representative_longest"]
        start = pd.Timestamp(rep["start"]).strftime("%Y-%m-%d %H:%M")
        end = pd.Timestamp(rep["end"]).strftime("%Y-%m-%d %H:%M")
        sp = row["seasonal_percent_of_union"]
        episode_rows.append(
            f"{row['site']} & {ep['count']} & {ep['median_duration_hours']:.2f} & "
            f"{ep['p95_duration_hours']:.2f} & {start}--{end} & {ep['max_duration_hours']:.2f} & "
            f"{rules_for_episode(rep)} & {sp['DJF']:.1f}/{sp['MAM']:.1f}/{sp['JJA']:.1f}/{sp['SON']:.1f} \\\\"
        )
    minimum = min(row["union_percent"] for row in rows)
    maximum = max(row["union_percent"] for row in rows)
    farm6 = next(row for row in rows if row["site"] == 6)
    farm6_examples = []
    for label, key in (("W", "wind"), ("C", "cold"), ("H", "hot")):
        item = farm6["rule_episodes"][key]["representative_longest"]
        start = pd.Timestamp(item["start"]).strftime("%Y-%m-%d %H:%M")
        end = pd.Timestamp(item["end"]).strftime("%Y-%m-%d %H:%M")
        farm6_examples.append(f"{label}: {start}--{end} ({item['duration_hours']:.2f}\\,h)")
    return r"""% Generated by review/summarize_extreme_events.py; do not edit manually.
% BEGIN STUDY BLOCK extreme_composition
\subsection{Meteorological-tail domain composition}\label{sec:extreme_composition}
{\color{blue}Table~\ref{tab:extreme_rule_counts} describes all eligible 2019--2020 forecast targets before any model fitting. Here W denotes the site-specific upper-2\% wind rule, C and H the lower-2\% and upper-2\% temperature rules, and $>25$ the nominal 25\,m/s rule. The cold and hot rules are mutually exclusive, so the two displayed wind--temperature intersections report every possible overlap between the three base rules. At every site the fitted percentile wind threshold is below 25\,m/s, so the $>25$ rule is nested in W and adds no observation to the union. The union represents """ + f"{minimum:.2f}--{maximum:.2f}" + r"""\% of eligible windows across sites; departures from 6\% reflect ties, cross-rule overlap, and temporal distribution shift relative to the 2019 threshold-fitting period.}

\begin{table}[htbp]
\centering
\caption{\textcolor{blue}{Composition of the sample-level meteorological target domain over all eligible 2019--2020 windows. Percentages in the last column use all eligible windows at that site as the denominator.}}
\label{tab:extreme_rule_counts}
{\color{blue}\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{crrrrrrrr}
\toprule
Site & Eligible & W & C & H & $>25$ & W$\cap$C & W$\cap$H & Union (\%) \\
\midrule
""" + "\n".join(count_rows) + r"""
\bottomrule
\end{tabular}}}
\end{table}

{\color{blue}For temporal description, an episode is a maximal sequence of union-member target timestamps exactly 15 minutes apart. Duration is the represented exposure, calculated as the number of retained 15-minute target slots times 0.25\,h; it therefore exceeds the difference between the last and first displayed timestamps by 0.25\,h. The P95 column uses the linearly interpolated empirical 95th percentile. Table~\ref{tab:extreme_episodes} reports all such episodes and chooses the longest episode at each site as a reproducible representative; the earliest start breaks a duration tie, and prediction errors are never used. Seasonal entries are the percentages of that site's union windows in December--February, March--May, June--August, and September--November. At the detailed Farm~6 site, applying the same objective rule separately gives """ + "; ".join(farm6_examples) + r"""; no $>25$\,m/s episode occurs. These are descriptive sample episodes, not independently named storms. The domain definition does not include power-ramp events: ramp-risk forecasting would require a separately declared ramp magnitude, horizon, and onset target, and is outside the present next-step conditional power task.}

\begin{table*}[htbp]
\centering
\caption{\textcolor{blue}{Duration and seasonality of the sample-level meteorological target domain. Representative spans are the objectively selected longest episodes. W, C, H, and $>25$ identify rules present at least once in that episode.}}
\label{tab:extreme_episodes}
{\color{blue}\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{crrrlrlc}
\toprule
Site & Episodes & Median (h) & P95 (h) & Longest target-time span & Max (h) & Rules & DJF/MAM/JJA/SON (\%) \\
\midrule
""" + "\n".join(episode_rows) + r"""
\bottomrule
\end{tabular}}}
\end{table*}
\FloatBarrier
% END STUDY BLOCK extreme_composition
"""


def main() -> None:
    data = load_all_selection_data(mmap_mode="r")
    rows = [analyze_site(site, data[site]) for site in sorted(data)]
    payload = {
        "scope": "all eligible chronological target timestamps from 2019-01-01 through 2020-12-31",
        "episode_definition": "maximal target-domain sequence with exactly 15 minutes between adjacent target timestamps",
        "representative_rule": "longest episode at each site; earliest start breaks a duration tie; prediction errors are not used",
        "source_script": str(Path(__file__).relative_to(ROOT)),
        "source_script_sha256": digest(Path(__file__)),
        "sites": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    TEX.write_text(render_tex(rows))
    print(json.dumps({"json": str(OUT), "tex": str(TEX), "sites": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
