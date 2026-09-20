"""Unit tests for the LP/MIP solver (no HTTP layer)."""
import pytest

from app.solver import SourceData, solve_blend, solve_contingency


def src(id, max_volume, cost, ca=50.0, mg=10.0, alk=100.0, batch_size=None,
        minimum_run_volume=None):
    return SourceData(
        id=id, max_volume=max_volume, cost=cost,
        calcium=ca, magnesium=mg, alkalinity=alk,
        batch_size=batch_size, minimum_run_volume=minimum_run_volume,
    )


def test_cheapest_source_wins():
    sources = [src("B", 1000, 2.0), src("A", 1000, 1.0)]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(500.0, abs=1e-6)
    assert vols["B"] == pytest.approx(0.0, abs=1e-6)
    assert out.total_cost == pytest.approx(500.0, abs=1e-6)


def test_indicator_constraint_forces_mixing():
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (40.0, 60.0)})
    assert out.status == "optimal"
    w1, w2 = out.volumes
    assert w1 == pytest.approx(500.0, abs=1e-6)
    assert w2 == pytest.approx(500.0, abs=1e-6)
    assert out.indicators["calcium"] == pytest.approx(60.0, abs=1e-6)
    assert out.total_cost == pytest.approx(1500.0, abs=1e-6)


def test_lexicographic_tie_break_uses_id_order():
    # Identical cost/concentrations: any split is optimal.  The usage vector
    # ordered by source id must be lexicographically minimal.
    sources = [
        src("C", 60, 1.0),
        src("A", 100, 1.0),
        src("B", 100, 1.0),
    ]
    out = solve_blend(sources, 100, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B", "C"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(40.0, abs=1e-6)
    assert vols["C"] == pytest.approx(60.0, abs=1e-6)


def test_infeasible_indicator_reports_achievable_range():
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (70.0, 90.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(36.0, abs=1e-6)
    assert ca["max"] == pytest.approx(68.0, abs=1e-6)
    assert ca["requested_min"] == 70.0
    assert "calcium" in out.message


def test_infeasible_volume_reports_max_available():
    sources = [src("W1", 600, 1.0), src("W2", 800, 2.0)]
    out = solve_blend(sources, 2000, {})
    assert out.status == "infeasible"
    assert out.achievable_ranges["volume"]["max"] == pytest.approx(1400.0)
    # ranges are computed against the actually attainable volume
    assert out.achievable_ranges["calcium"]["min"] >= 0


def test_jointly_incompatible_ranges():
    # Calcium wants lots of W1, alkalinity wants lots of W2; the windows
    # cannot be met at the same time.
    sources = [
        src("W1", 1000, 1.0, ca=100.0, alk=200.0),
        src("W2", 1000, 1.0, ca=0.0, alk=0.0),
    ]
    out = solve_blend(
        sources, 1000, {"calcium": (90.0, 100.0), "alkalinity": (0.0, 20.0)}
    )
    assert out.status == "infeasible"
    assert "jointly incompatible" in out.message


# --------------------------------------------------------------------------- #
# batch_size (mixed-integer) semantics
# --------------------------------------------------------------------------- #

def test_unset_batch_size_stays_continuous():
    # Identical-cost sources stay continuous when batch_size is omitted;
    # the id-sorted usage vector is lex-min, so A is drained first.
    out = solve_blend([src("A", 100, 1.0), src("B", 100, 1.0)], 33.5, {})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([0.0, 33.5], abs=1e-6)


def test_mixed_integer_and_continuous_sources():
    # A continuous (ca 100, cost 2), B delivered in 300 m3 batches (ca 20,
    # cost 1), V = 1000, calcium window [40, 80].  The continuous relaxation
    # would take B=500/A=500 (calcium 60, cost 1500) — which is NOT batch
    # aligned.  Enumerating the lattice:
    #   B=300/A=700 -> calcium 76, cost 1700
    #   B=600/A=400 -> calcium 52, cost 1400  <-- MIP optimum
    #   B=900/A=100 -> calcium 28 (out of window)
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (40.0, 80.0)})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([400.0, 600.0], abs=1e-6)
    assert out.total_cost == pytest.approx(1400.0, abs=1e-6)
    assert out.indicators["calcium"] == pytest.approx(52.0, abs=1e-6)
    # the batch law actually holds
    b = out.volumes[1]
    assert b / 300.0 == pytest.approx(round(b / 300.0), abs=1e-6)


def test_no_rounding_of_continuous_solution():
    # The continuous optimum takes the cheap batched source fractionally;
    # rounding it either way breaks the calcium window.  Only the MIP finds
    # the aligned feasible allocation.
    # W1 ca 100 batch 300 cap 500 (so at most 1 batch), W2 ca 20 continuous,
    # V = 800.  Continuous max-calcium blend W1=500, W2=300 (calcium 70);
    # on the lattice W1 in {0, 300}: max calcium is 50 at W1=300, W2=500.
    sources = [
        src("W1", 500, 1.0, ca=100.0, batch_size=300.0),
        src("W2", 1000, 2.0, ca=20.0),
    ]
    # Feasible window for the MIP but not for the rounded continuous answer:
    out = solve_blend(sources, 800, {"calcium": (45.0, 55.0)})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([300.0, 500.0], abs=1e-6)
    assert out.indicators["calcium"] == pytest.approx(50.0, abs=1e-6)
    # and the window above the discrete maximum must be infeasible with the
    # range computed by the SAME mixed-integer model (max 50, not 70)
    bad = solve_blend(sources, 800, {"calcium": (60.0, 90.0)})
    assert bad.status == "infeasible"
    assert bad.achievable_ranges["calcium"]["max"] == pytest.approx(50.0, abs=1e-6)
    assert "calcium" in bad.message


def test_non_divisible_max_volume_is_floored():
    # B is the cheapest source but max_volume 850 is not a multiple of 300:
    # only 2 batches (600) may be drawn, never 850.
    sources = [
        src("A", 1000, 2.0),
        src("B", 850, 1.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 1000, {})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([400.0, 600.0], abs=1e-6)
    assert out.total_cost == pytest.approx(1400.0, abs=1e-6)


def test_batch_size_larger_than_max_volume_eliminates_source():
    sources = [
        src("A", 1000, 2.0),
        src("B", 50, 1.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 1000, {})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([1000.0, 0.0], abs=1e-6)


def test_batch_size_one_is_still_integer():
    # batch_size 1.0 means whole-cubic-meter batches; a fractional target is
    # not on the lattice even though scale equals the continuous one.
    sources = [
        src("A", 100, 1.0, batch_size=1.0),
        src("B", 100, 2.0, batch_size=1.0),
    ]
    ok = solve_blend(sources, 100, {})
    assert ok.status == "optimal"
    assert ok.volumes == pytest.approx([100.0, 0.0], abs=1e-6)

    bad = solve_blend(sources, 100.25, {})
    assert bad.status == "infeasible"
    assert bad.achievable_ranges["volume"]["max"] == pytest.approx(100.0, abs=1e-9)


def test_integer_tie_break_is_lexicographic_by_id():
    # Three identical batched sources, V = 500, batch 100.  Any five batches
    # are cost-optimal; the id-ascending usage vector must be lex-min, i.e.
    # (0, 200, 300) once caps bind C at 300.
    sources = [
        src("C", 300, 1.0, batch_size=100.0),
        src("A", 300, 1.0, batch_size=100.0),
        src("B", 300, 1.0, batch_size=100.0),
    ]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B", "C"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(200.0, abs=1e-6)
    assert vols["C"] == pytest.approx(300.0, abs=1e-6)
    for v in out.volumes:
        assert (v / 100.0) == pytest.approx(round(v / 100.0), abs=1e-6)


def test_batch_lattice_misses_target_volume():
    # Both sources batch 300; reachable totals are multiples of 300 up to
    # 1200, so 1000 cannot be assembled at all.
    sources = [
        src("A", 600, 1.0, batch_size=300.0),
        src("B", 600, 2.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 1000, {})
    assert out.status == "infeasible"
    assert out.achievable_ranges["volume"]["max"] == pytest.approx(900.0, abs=1e-6)
    assert out.achievable_ranges["volume"]["requested"] == pytest.approx(1000.0)
    # ranges are reported at the largest attainable volume (lattice-honoring)
    assert out.achievable_ranges["calcium"]["min"] == pytest.approx(50.0, abs=1e-6)


def test_mixed_lattice_misses_target_volume():
    # Continuous A caps at 100, B batches of 300; totals are 300k + a with
    # a in [0, 100].  1050 is unreachable; the best volume not above it is
    # 1000 (3 batches + 100).
    sources = [
        src("A", 100, 1.0),
        src("B", 1200, 2.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 1050, {})
    assert out.status == "infeasible"
    assert out.achievable_ranges["volume"]["max"] == pytest.approx(1000.0, abs=1e-6)
    assert "batch-size" in out.message


def test_discrete_achievable_range_differs_from_continuous():
    # W1 ca 100 batch 300 cap 500, W2 ca 20 continuous, V = 800.
    # Continuous max calcium is 70 (W1=500), discrete max is 50 (W1=300).
    sources = [
        src("W1", 500, 1.0, ca=100.0, batch_size=300.0),
        src("W2", 1000, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 800, {"calcium": (60.0, 90.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(20.0, abs=1e-6)
    assert ca["max"] == pytest.approx(50.0, abs=1e-6)
    assert "volume" not in out.achievable_ranges  # 800 is itself attainable


def test_all_batched_indicator_infeasible_range():
    # W1 ca 100 batch 300 cap 600, W2 ca 20 batch 300 cap 900, V = 900.
    # Discrete calcium extremes: 20 (only W2), 73.333 (600 W1 + 300 W2).
    sources = [
        src("W1", 600, 1.0, ca=100.0, batch_size=300.0),
        src("W2", 900, 2.0, ca=20.0, batch_size=300.0),
    ]
    out = solve_blend(sources, 900, {"calcium": (80.0, 90.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(20.0, abs=1e-6)
    assert ca["max"] == pytest.approx(73.333333, abs=1e-6)


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def test_contingency_returns_original_when_cap_not_binding():
    sources = [src("A", 100, 1.0), src("B", 100, 2.0)]
    out = solve_blend(sources, 50, {})
    x0 = dict(zip(["A", "B"], out.volumes))
    cont = solve_contingency(sources, 50, {}, x0, {"A": 100.0})
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([50.0, 0.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(0.0, abs=1e-9)
    assert cont.cost_delta == pytest.approx(0.0, abs=1e-9)


def test_contingency_min_change_then_cost():
    # Allocation (40, 60, 0).  B (expensive) is capped at 40, so 20 units
    # must be redistributed.  Minimum change = 40: A alone (60, 40, 0) or
    # with C involved; the cheapest min-change option avoids C.
    sources = [
        src("A", 100, 1.0),
        src("B", 100, 2.0),
        src("C", 60, 3.0),
    ]
    cont = solve_contingency(
        sources, 100, {}, {"A": 40.0, "B": 60.0, "C": 0.0}, {"B": 40.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([60.0, 40.0, 0.0], abs=1e-6)
    assert cont.deltas == pytest.approx([20.0, -20.0, 0.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(40.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(140.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(-20.0, abs=1e-6)


def test_contingency_cost_increase_possible():
    # Cheap source W1 is cut below what the indicator window requires; the
    # missing volume moves to the more expensive W2.
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    limits = {"calcium": (40.0, 60.0)}
    cont = solve_contingency(
        sources, 1000, limits, {"W1": 500.0, "W2": 500.0}, {"W1": 400.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([400.0, 600.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(200.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(1600.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(100.0, abs=1e-6)
    assert cont.indicators["calcium"] == pytest.approx(52.0, abs=1e-6)


def test_contingency_lexicographic_tie_break():
    # Three identical sources, original (40, 60, 0), both A and B capped at
    # 40: every min-change allocation shifts 20 units away from B; the
    # id-sorted usage vector must be lexicographically minimal.
    sources = [
        src("A", 100, 1.0),
        src("B", 100, 1.0),
        src("C", 60, 1.0),
    ]
    cont = solve_contingency(
        sources, 100, {},
        {"A": 40.0, "B": 60.0, "C": 0.0},
        {"A": 40.0, "B": 40.0},
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([40.0, 40.0, 20.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(40.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(0.0, abs=1e-9)


def test_contingency_zero_cap_eliminates_source():
    sources = [src("A", 100, 1.0), src("B", 100, 2.0)]
    cont = solve_contingency(
        sources, 50, {}, {"A": 50.0, "B": 0.0}, {"A": 0.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([0.0, 50.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(100.0, abs=1e-6)


def test_contingency_infeasible_volume():
    sources = [src("W1", 600, 1.0), src("W2", 800, 2.0)]
    cont = solve_contingency(
        sources, 2000, {},
        {"W1": 600.0, "W2": 800.0}, {"W2": 100.0},
    )
    assert cont.status == "infeasible"
    assert cont.achievable_ranges["volume"]["max"] == pytest.approx(700.0)
    assert "below the target" in cont.message


def test_contingency_infeasible_indicator():
    # Total supply stays at 1000 but cutting the high-calcium W1 to 200
    # makes the calcium minimum 40 unattainable (max achievable is 36).
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    cont = solve_contingency(
        sources, 1000, {"calcium": (40.0, 60.0)},
        {"W1": 500.0, "W2": 500.0}, {"W1": 200.0},
    )
    assert cont.status == "infeasible"
    assert "volume" not in cont.achievable_ranges
    assert cont.achievable_ranges["calcium"]["max"] == pytest.approx(36.0, abs=1e-6)
    assert "calcium" in cont.message


# --------------------------------------------------------------------------- #
# batch-aware supply-reduction contingencies
# --------------------------------------------------------------------------- #

def test_contingency_keeps_batch_lattice():
    # Original plan from the mixed model: A=400 continuous, B=600 (2*300).
    # Cut B's cap to 350: at most one batch (300) may be drawn; the missing
    # 100 m3 moves to continuous A.
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, batch_size=300.0),
    ]
    cont = solve_contingency(
        sources, 1000, {"calcium": (40.0, 80.0)},
        {"A": 400.0, "B": 600.0}, {"B": 350.0},
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([700.0, 300.0], abs=1e-6)
    assert cont.deltas == pytest.approx([300.0, -300.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(600.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(1700.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(300.0, abs=1e-6)
    assert cont.indicators["calcium"] == pytest.approx(76.0, abs=1e-6)


def test_contingency_non_divisible_cap_is_floored():
    # All sources batched at 200 and equally priced; original (0, 400, 600)
    # on V = 1000.  Cutting B's cap to 350 allows at most 1 batch (200) from
    # B.  Minimum change is 400; among equal-cost re-allocations the
    # id-ascending usage vector decides: A stays at 0 and C takes the batch,
    # giving (0, 200, 800) rather than (200, 200, 600).
    sources = [
        src("A", 1000, 1.0, batch_size=200.0),
        src("B", 1000, 1.0, batch_size=200.0),
        src("C", 1000, 1.0, batch_size=200.0),
    ]
    cont = solve_contingency(
        sources, 1000, {},
        {"A": 0.0, "B": 400.0, "C": 600.0}, {"B": 350.0},
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([0.0, 200.0, 800.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(400.0, abs=1e-6)
    for v in cont.volumes:
        assert (v / 200.0) == pytest.approx(round(v / 200.0), abs=1e-6)


def test_contingency_batch_lattice_infeasible():
    # V = 900 assembled as W1=300, W2=600 (both batch 300, calcium window
    # [40, 60], achieved calcium 46.67).  Eliminating W1 leaves only W2 with
    # calcium 20 regardless of batches: infeasible on the indicator.
    sources = [
        src("W1", 600, 1.0, ca=100.0, batch_size=300.0),
        src("W2", 900, 2.0, ca=20.0, batch_size=300.0),
    ]
    cont = solve_contingency(
        sources, 900, {"calcium": (40.0, 60.0)},
        {"W1": 300.0, "W2": 600.0}, {"W1": 0.0},
    )
    assert cont.status == "infeasible"
    assert "volume" not in cont.achievable_ranges
    assert cont.achievable_ranges["calcium"]["max"] == pytest.approx(20.0, abs=1e-6)
    assert "calcium" in cont.message


def test_contingency_batch_cap_cuts_lattice_volume():
    # V = 900 from W1=300, W2=600 (batches of 300).  Cutting W2 to 200 floors
    # its batch count to zero while W1 can supply at most 600: total supply
    # on the lattice is 600 < 900 -> infeasible, reported with ranges.
    sources = [
        src("W1", 600, 1.0, batch_size=300.0),
        src("W2", 900, 2.0, batch_size=300.0),
    ]
    cont = solve_contingency(
        sources, 900, {}, {"W1": 300.0, "W2": 600.0}, {"W2": 200.0},
    )
    assert cont.status == "infeasible"
    assert cont.achievable_ranges["volume"]["max"] == pytest.approx(600.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# minimum_run_volume / minimum_active_sources (binary enable variables)
# --------------------------------------------------------------------------- #

def test_minimum_run_unset_does_not_change_anything():
    out = solve_blend(
        [src("A", 100, 1.0, minimum_run_volume=None),
         src("B", 100, 1.0, minimum_run_volume=None)],
        33.5, {},
    )
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([0.0, 33.5], abs=1e-6)


def test_minimum_run_forces_cheap_source_to_zero():
    # Unconstrained calcium optimum [40, 100] is A=250, B=750 (calcium 40).
    # B's gate of 800 forbids the band (0, 800); at B>=800 calcium drops below
    # 40, so the only feasible optimum is B=0, A=1000 — never B=750.
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, minimum_run_volume=800.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (40.0, 100.0)})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([1000.0, 0.0], abs=1e-6)
    assert out.total_cost == pytest.approx(2000.0, abs=1e-6)


def test_minimum_run_allows_zero_and_above_threshold():
    # Without a window the cheapest gated source B is simply drained at 100.
    sources = [
        src("A", 1000, 2.0),
        src("B", 1000, 1.0, minimum_run_volume=300.0),
    ]
    out = solve_blend(sources, 1000, {})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([0.0, 1000.0], abs=1e-6)

    # V = 100 < gate 300: B cannot be partially used, so A covers everything.
    small = solve_blend(sources, 100, {})
    assert small.status == "optimal"
    assert small.volumes == pytest.approx([100.0, 0.0], abs=1e-6)


def test_minimum_run_never_yields_interior_volume():
    # Whatever the optimum, a gated source volume is either 0 or >= its gate.
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, minimum_run_volume=250.0),
        src("C", 1000, 3.0, ca=60.0, minimum_run_volume=150.0),
    ]
    for lo, hi in [(0.0, 100.0), (30.0, 80.0), (55.0, 95.0)]:
        out = solve_blend(sources, 1000, {"calcium": (lo, hi)})
        assert out.status == "optimal"
        v_b, v_c = out.volumes[1], out.volumes[2]
        assert v_b == pytest.approx(0.0, abs=1e-6) or v_b >= 250.0 - 1e-6
        assert v_c == pytest.approx(0.0, abs=1e-6) or v_c >= 150.0 - 1e-6
        # and the post-solve-prune rule would have changed the answer
        assert sum(out.volumes) == pytest.approx(1000.0, abs=1e-6)


def test_minimum_run_lexicographic_tie_break():
    # Three identical gated sources (gate 40), V = 100.  Feasible cost-tied
    # allocations need every used source at >=40; the id-ascending lex-min is
    # A=0, B=40, C=60 (C capped at 60).
    sources = [
        src("C", 60, 1.0, minimum_run_volume=40.0),
        src("A", 100, 1.0, minimum_run_volume=40.0),
        src("B", 100, 1.0, minimum_run_volume=40.0),
    ]
    out = solve_blend(sources, 100, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B", "C"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(40.0, abs=1e-6)
    assert vols["C"] == pytest.approx(60.0, abs=1e-6)


def test_minimum_active_sources_requires_threshold_sources():
    # Only B and C carry a gate; A has none and must not count toward the
    # minimum.  min_active=2 forces both B and C above their gates.
    sources = [
        src("A", 100, 1.0),
        src("B", 100, 1.0, minimum_run_volume=40.0),
        src("C", 60, 1.0, minimum_run_volume=40.0),
    ]
    out = solve_blend(sources, 100, {}, min_active=2)
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([0.0, 40.0, 60.0], abs=1e-6)

    # min_active=3 cannot be met: only two gated sources exist -> infeasible,
    # reported on the SAME constrained feasible set (nothing assemblable).
    bad = solve_blend(sources, 100, {}, min_active=3)
    assert bad.status == "infeasible"
    assert bad.achievable_ranges["volume"]["max"] == pytest.approx(0.0, abs=1e-9)


def test_minimum_active_zero_is_noop():
    sources = [
        src("A", 100, 1.0, minimum_run_volume=90.0),
        src("B", 100, 1.0, minimum_run_volume=90.0),
    ]
    out = solve_blend(sources, 100, {}, min_active=0)
    assert out.status == "optimal"
    # one source alone covers 100 (>= its gate); the other stays at 0
    assert sorted(out.volumes) == pytest.approx([0.0, 100.0], abs=1e-6)


def test_minimum_run_combined_with_batch_size():
    # B comes in 300 m3 batches with a 450 m3 gate: usable B in
    # {0, 600, 900, ...}.  Calcium window [70, 100] excludes B=600
    # (calcium 52), so B is forced to 0 and A covers 1000.
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, batch_size=300.0,
            minimum_run_volume=450.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (70.0, 100.0)})
    assert out.status == "optimal"
    assert out.volumes == pytest.approx([1000.0, 0.0], abs=1e-6)

    # Wider window: MIP picks B=600 (2 batches, satisfies both lattice and
    # gate) with A=400.
    ok = solve_blend(sources, 1000, {"calcium": (40.0, 100.0)})
    assert ok.status == "optimal"
    assert ok.volumes == pytest.approx([400.0, 600.0], abs=1e-6)


def test_minimum_run_infeasible_ranges_honor_gate():
    # Gate keeps B in {0} U [800, 1000]; with A capped at 800 the discrete
    # feasible set spans calcium [20, 36], not the continuous [20, 44].
    sources = [
        src("A", 800, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, minimum_run_volume=800.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (40.0, 60.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(20.0, abs=1e-6)
    assert ca["max"] == pytest.approx(36.0, abs=1e-6)
    assert ca["requested_min"] == 40.0
    assert "volume" not in out.achievable_ranges
    assert "calcium" in out.message


def test_minimum_active_count_infeasible_ranges():
    # Two gated sources (gate 60), V = 100, min_active=2: no allocation
    # exists (120 > 100); the projection feasible set is empty.
    sources = [
        src("A", 100, 1.0, minimum_run_volume=60.0),
        src("B", 100, 1.0, minimum_run_volume=60.0),
    ]
    out = solve_blend(sources, 100, {}, min_active=2)
    assert out.status == "infeasible"
    assert out.achievable_ranges["volume"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert "source-activation" in out.message


# --------------------------------------------------------------------------- #
# contingencies with minimum_run_volume / minimum_active_sources
# --------------------------------------------------------------------------- #

def test_contingency_preserves_minimum_run():
    # Original (A=0, B=40, C=60) with gates 40, min_active=2 and C capped at
    # 60.  Cutting C to 20 deactivates it; the count constraint forces A on at
    # >=40: minimum-change solution is A=40, B=60, C=0 (change 120).
    sources = [
        src("A", 100, 1.0, minimum_run_volume=40.0),
        src("B", 100, 1.0, minimum_run_volume=40.0),
        src("C", 100, 1.0, minimum_run_volume=40.0),
    ]
    cont = solve_contingency(
        sources, 100, {},
        {"A": 0.0, "B": 40.0, "C": 60.0}, {"C": 20.0}, min_active=2,
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([40.0, 60.0, 0.0], abs=1e-6)
    assert cont.deltas == pytest.approx([40.0, 20.0, -60.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(120.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(0.0, abs=1e-6)


def test_contingency_gate_forces_zero_under_reduced_cap():
    # Original (1000, 0); A cut to 800.  B's gate is 800, so any blend with
    # B in (0, 200] is forbidden; calcium window [40,100] rules B>=800 out ->
    # infeasible, and the reported calcium maximum is 36 on the SAME set.
    sources = [
        src("A", 1000, 2.0, ca=100.0),
        src("B", 1000, 1.0, ca=20.0, minimum_run_volume=800.0),
    ]
    cont = solve_contingency(
        sources, 1000, {"calcium": (40.0, 100.0)},
        {"A": 1000.0, "B": 0.0}, {"A": 800.0},
    )
    assert cont.status == "infeasible"
    assert cont.achievable_ranges["calcium"]["max"] == pytest.approx(36.0, abs=1e-6)
    assert "calcium" in cont.message


def test_contingency_count_infeasible_under_reduced_caps():
    # min_active=2 with three gated sources (gate 40); cutting two of them to
    # 20 leaves at most one activatable source -> infeasible.
    sources = [
        src("A", 100, 1.0, minimum_run_volume=40.0),
        src("B", 100, 1.0, minimum_run_volume=40.0),
        src("C", 100, 1.0, minimum_run_volume=40.0),
    ]
    cont = solve_contingency(
        sources, 100, {},
        {"A": 0.0, "B": 60.0, "C": 40.0}, {"B": 20.0, "C": 20.0},
        min_active=2,
    )
    assert cont.status == "infeasible"
    assert cont.achievable_ranges["volume"]["max"] == pytest.approx(0.0, abs=1e-9)
