from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "review" / "integrate_v2_manuscript_prose.py"
TEMPLATE = ROOT / "review" / "final_selection_study_template.tex"


def _load_module():
    spec = importlib.util.spec_from_file_location("integrate_v2_manuscript_prose", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def module():
    return _load_module()


def _scope(scope: str, winner: str, winner_value: float, full_value: float,
           position: int, member: bool, rank: float, wins: int):
    labels = {
        "Full": "TL-QLDMR",
        "WeightedSVQR": "Weighted SVQR",
        "HistGBR": "HistGBR",
    }
    members = [winner] if not member else list(dict.fromkeys([winner, "Full"]))
    return {
        "scope": scope,
        "role": (
            "primary_within_farm"
            if scope == "within"
            else "secondary_cross_farm_stress"
        ),
        "empirical_winner": {
            "method": winner,
            "method_label": labels[winner],
            "macro_capacity_normalized_rmse_percent": winner_value,
        },
        "full": {
            "macro_capacity_normalized_rmse_percent": full_value,
            "average_task_rank": rank,
            "task_wins_including_ties": wins,
            "macro_nrmse_position": position,
            "in_conservative_nonrejection_set": member,
            "holm_p_vs_empirical_winner": 0.2 if member else 0.01,
        },
        "first_tier_members": [
            {"method": method, "method_label": labels[method]} for method in members
        ],
        "point_sentence": f"Synthetic point sentence for {scope}.",
        "first_tier_sentence": f"Synthetic first-tier sentence for {scope}.",
    }


def _matched_scope(scope: str, full_value: float, full_position: int, full_rank: float):
    raw = [
        ("Full", "TL-QLDMR", full_value),
        ("NoVariance", "No variance", full_value - 0.2),
        ("NoDiscrepancy", "No discrepancy", full_value + 0.1),
        ("WeightedSVQR", "Weighted SVQR", full_value + 0.3),
    ]
    rows = [
        {
            "scope": scope,
            "variant": variant,
            "variant_label": label,
            "macro_capacity_normalized_rmse_percent": value,
            "mean_rank": 1.0 + index,
            "macro_nrmse_position": 1 + sum(other[2] < value for other in raw),
        }
        for index, (variant, label, value) in enumerate(raw)
    ]
    rows.sort(key=lambda item: item["macro_capacity_normalized_rmse_percent"])
    return {
        "scope": scope,
        "variants_sorted_by_macro_nrmse": rows,
        "full": {
            "macro_capacity_normalized_rmse_percent": full_value,
            "mean_rank": full_rank,
            "macro_nrmse_position": full_position,
        },
        "aggregate_sentence": f"Synthetic matched aggregate for {scope}.",
    }


def _payload():
    return {
        "version": "v2_result_prose_1",
        "input_policy": "Completed hash-verified synthetic analysis summaries only.",
        "source_sha256": {"synthetic/source": "a" * 64},
        "interpretation_rules": {
            "first_tier": (
                "Membership means non-rejection and does not establish equivalence, equal "
                "performance, or superiority."
            ),
            "matched": (
                "Effects are conditional, variants were not independently retuned, and failure "
                "to reject is not equivalence."
            ),
        },
        "scope_results": [
            _scope("within", "WeightedSVQR", 10.0, 10.4, 2, True, 2.1, 2),
            _scope("cross", "HistGBR", 11.0, 13.5, 4, False, 3.8, 1),
        ],
        "full_task_inference": {
            "significant_count": 3,
            "significant_favorable_to_full": [{"task": "within_site1"}],
            "significant_adverse_to_full": [
                {"task": "cross_site2"}, {"task": "cross_site5"}
            ],
            "significant_equal_observed": [],
            "all_significant_sentences": [],
            "complete_sentence": "Synthetic complete task inference.",
        },
        "matched_ablation": {
            "scope_results": [
                _matched_scope("within", 10.5, 2, 2.0),
                _matched_scope("cross", 13.6, 2, 2.4),
            ],
            "significant_negative_full_minus_removal": [{"task": "within_site3"}],
            "significant_positive_full_minus_removal": [{"task": "cross_site4"}],
            "significant_zero_full_minus_removal": [],
            "significant_effect_sentences": [],
            "inference_sentence": "Synthetic matched inference.",
            "conditional_caveat": (
                "Effects are conditional, variants were not independently retuned, and failure "
                "to reject is not equivalence."
            ),
        },
        "manuscript_blocks": {
            "RESULTS_POINT": "POINT-BLOCK-EXACT",
            "RESULTS_TASK_INFERENCE": "TASK-INFERENCE-BLOCK-EXACT",
            "RESULTS_FIRST_TIER": "FIRST-TIER-BLOCK-EXACT",
            "RESULTS_MATCHED_ABLATION": "MATCHED-BLOCK-EXACT",
            "RESULTS_OPTIMIZATION": "OPTIMIZATION-BLOCK-EXACT",
        },
    }


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "manuscript"
    root.mkdir()
    for name in ("TLQLDMR_energy.tex", "Highlights.tex", "selection_appendix.tex"):
        shutil.copy2(ROOT / "TLQLDMR_R1" / name, root / name)
    return root


def _inputs(tmp_path: Path):
    results = tmp_path / "v2_result_prose.json"
    results.write_text(json.dumps(_payload()), encoding="utf-8")
    template = tmp_path / "final_selection_study_template.tex"
    shutil.copy2(TEMPLATE, template)
    return results, template


def test_dry_run_reads_synthetic_summary_only_and_changes_nothing(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    report, code = module.integrate(results, template, root, "dry-run")

    assert code == 0
    assert report["status"] == "would_change"
    assert report["validation"]["result_placeholders_replaced"] == 5
    assert report["validation"]["remaining_result_placeholders"] == 0
    assert max(report["validation"]["highlight_visible_characters"]) <= 85
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_write_then_check_is_idempotent_and_preserves_adverse_results(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)

    written, code = module.integrate(results, template, root, "write")
    assert code == 0
    assert written["status"] == "written"
    checked, code = module.integrate(results, template, root, "check")
    assert code == 0
    assert checked["status"] == "clean"
    assert checked["changed_count"] == 0

    selection = (root / "selection_study.tex").read_text(encoding="utf-8")
    assert "@@RESULTS_" not in selection
    for block in _payload()["manuscript_blocks"].values():
        assert selection.count(block) == 1

    main = (root / "TLQLDMR_energy.tex").read_text(encoding="utf-8")
    assert main.count("% BEGIN V2 GENERATED abstract") == 1
    assert main.count("% BEGIN V2 GENERATED contribution_result") == 1
    assert main.count("% BEGIN V2 GENERATED conclusion_evidence") == 1
    assert "1 Holm-adjusted finding favored TL-QLDMR and 2 favored a comparator" in main
    assert "1 Holm-adjusted effects favored TL-QLDMR and 1 favored a removal variant" in main
    assert "primary within-site" in main and "secondary cross-site stress" in main
    assert "does not establish equivalence" in main
    assert "reused period" in main
    assert "full model" not in main.lower()

    highlights = (root / "Highlights.tex").read_text(encoding="utf-8")
    bullets = re_find_items(highlights)
    assert len(bullets) == 4
    assert all(len(item) <= 85 for item in bullets)
    assert bullets[0].startswith("TL-QLDMR: Transfer Learning")
    assert "% Visible bullet lengths, including spaces and punctuation:" in highlights

    appendix = (root / "selection_appendix.tex").read_text(encoding="utf-8")
    formal = appendix.index("Formal held-out results and paired inference")
    audit = appendix.index("Frozen selection and optimization audit")
    legacy = appendix.index("Legacy exploratory probability configuration")
    assert formal < audit < legacy
    assert "\\input{selection_optimization_summary}" in appendix
    assert "\\input{probabilistic_configs}" in appendix


def re_find_items(text: str) -> list[str]:
    import re
    return [item.strip() for item in re.findall(r"(?m)^\s*\\item (.+)$", text)]


def test_unmarked_changed_abstract_fails_closed(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    main = root / "TLQLDMR_energy.tex"
    source = main.read_text(encoding="utf-8")
    begin = "% BEGIN V2 GENERATED abstract"
    end = "% END V2 GENERATED abstract"
    start = source.index(begin)
    stop = source.index(end, start) + len(end)
    main.write_text(
        source[:start]
        + "Changed wind-power forecasting during meteorological extremes."
        + source[stop:],
        encoding="utf-8",
    )
    with pytest.raises(module.IntegrationError, match="audited old paragraph hash"):
        module.integrate(results, template, root, "dry-run")


def test_incomplete_manuscript_blocks_are_rejected(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    payload = _payload()
    del payload["manuscript_blocks"]["RESULTS_TASK_INFERENCE"]
    results.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.IntegrationError, match="exactly"):
        module.integrate(results, template, root, "dry-run")


@pytest.mark.parametrize(
    ("unsafe", "message"),
    [
        ("An unescaped result is 12.3% and truncates TeX.", "unescaped TeX percent"),
        ("The Full-selected configuration was retained.", "internal method identifier"),
    ],
)
def test_unsafe_result_block_tex_is_rejected(tmp_path, module, unsafe, message):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    payload = _payload()
    payload["manuscript_blocks"]["RESULTS_POINT"] = unsafe
    results.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.IntegrationError, match=message):
        module.integrate(results, template, root, "dry-run")


def test_escaped_percent_is_accepted_in_result_block(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    payload = _payload()
    payload["manuscript_blocks"]["RESULTS_POINT"] = "The endpoint was 12.3\\%."
    results.write_text(json.dumps(payload), encoding="utf-8")
    report, code = module.integrate(results, template, root, "dry-run")
    assert code == 0
    assert report["status"] == "would_change"


def test_unknown_or_remaining_template_placeholder_is_rejected(tmp_path, module):
    root = _root(tmp_path)
    results, template = _inputs(tmp_path)
    template.write_text(
        template.read_text(encoding="utf-8") + "\n% @@RESULTS_UNKNOWN@@\n",
        encoding="utf-8",
    )
    with pytest.raises(module.IntegrationError, match="Unexpected result placeholder"):
        module.integrate(results, template, root, "dry-run")
