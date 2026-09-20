"""Mixed-integer blend optimizer.

Model
-----
Variables z_i >= 0: units drawn from source i.  When source i declares a
``batch_size`` b_i the variable is constrained integer and the volume is
``x_i = b_i * z_i``; otherwise z_i is continuous and ``x_i = z_i``.

    minimize    sum(cost_i * x_i)
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i
                z_i in Z>=0 for batched sources, z_i in R>=0 otherwise

Continuous and batched sources are solved together in a single mixed-integer
program; a continuous optimum is never rounded afterwards.

Source activation (minimum run volume)
--------------------------------------
A source may declare a positive ``minimum_run_volume`` m_i.  Its usage is then
either zero or at least m_i.  This is enforced *inside* the same MIP with a
binary enable variable y_i in {0, 1}:

    m_i * y_i <= x_i <= u_i * y_i

where u_i is the effective supply cap (``max_volume`` for continuous sources,
``b_i * floor(max_volume / b_i)`` for batched ones).  Sources without a
threshold get a y_i fixed at 0 and no such rows.  When the request sets
``minimum_active_sources`` k, the count constraint ``sum(y_i) >= k`` is added
over the threshold-carrying sources.  No small-usage solution is ever computed
and pruned afterwards: the big-M constraints are part of every optimization.

When several optima share the same minimal cost, the usage vector ordered by
ascending source id is minimized lexicographically via a sequence of MIPs.

If the problem is infeasible, the achievable (min, max) concentration range of
every indicator is reported, projected onto the *same* mixed-integer feasible
set (supply caps, volume, batch and activation/count constraints).

Supply-reduction contingency plans (``solve_contingency``) use the same
mixed-integer feasible set with tightened supply caps (including every
activation and count constraint), but optimize a lexicographic objective:
first minimize the total absolute volume change versus the stored allocation,
then minimize cost, and finally the source-id-ordered usage vector
lexicographically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

INDICATORS: Tuple[str, ...] = ("calcium", "magnesium", "alkalinity")

# Half-width (relative) of the interval used to pin a variable after its
# lexicographic minimum has been found.  Comfortably above solver noise and
# below the 6-decimal output rounding for typical magnitudes.
_FIX_RTOL = 1e-8
# Fallback cost slack (relative) if the exact cost-equality refinement fails.
_COST_RTOL = 1e-9


@dataclass
class SourceData:
    id: str
    max_volume: float
    cost: float
    calcium: float
    magnesium: float
    alkalinity: float
    batch_size: Optional[float] = None
    minimum_run_volume: Optional[float] = None


@dataclass
class _Model:
    """Constraint matrix description in the (possibly scaled) variable space.

    Variables are ordered ``(z_1..z_n, y_1..y_n)``; all matrices/vectors use
    that layout.  Volumes are recovered as ``x = scales * z``;
    ``integrality`` marks the batch-count and binary-enable variables.

    ``A_struct`` holds upper-bound rows that define the feasible set together
    with the bounds and the volume equality (activation and source-count rows);
    they are always enforced.  ``A_ub`` holds the *droppable* indicator-window
    rows, which the infeasibility projection deliberately omits.
    """

    n: int
    scales: np.ndarray           # batch_size per source, 1.0 when continuous
    integrality: np.ndarray      # 1 for integer/binary variables, else 0
    bounds: List[Tuple[float, float]]
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ub: Optional[np.ndarray]
    b_ub: Optional[np.ndarray]
    A_struct: Optional[np.ndarray] = None
    b_struct: Optional[np.ndarray] = None


@dataclass
class SolveOutput:
    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    total_cost: Optional[float] = None
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


@dataclass
class ContingencyOutput:
    """Result of a supply-reduction contingency re-optimization."""

    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    deltas: Optional[np.ndarray] = None        # new volume - original volume
    total_adjustment: Optional[float] = None   # sum of absolute deltas
    total_cost: Optional[float] = None
    cost_delta: Optional[float] = None         # new cost - original cost
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #

def _scales(sources: Sequence[SourceData]) -> np.ndarray:
    return np.array(
        [1.0 if s.batch_size is None else float(s.batch_size) for s in sources],
        dtype=float,
    )


def _integrality(sources: Sequence[SourceData]) -> np.ndarray:
    # Decided from the flag, not from scale == 1 (batch_size 1.0 stays integer).
    return np.array([0.0 if s.batch_size is None else 1.0 for s in sources],
                    dtype=float)


def _source_bounds(sources: Sequence[SourceData], scales: np.ndarray,
                   caps: Optional[Mapping[str, float]] = None
                   ) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for s, scale in zip(sources, scales):
        cap = float(s.max_volume) if caps is None else float(caps.get(s.id, s.max_volume))
        if s.batch_size is None:
            hi = cap
        else:
            hi = float(math.floor(cap / scale + 1e-12))
        bounds.append((0.0, hi))
    return bounds


def _activation_structure(
    sources: Sequence[SourceData],
    scales: np.ndarray,
    z_bounds: Sequence[Tuple[float, float]],
    min_active: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Big-M activation rows and the minimum-active-sources count row.

    For every threshold-carrying source i (``m_i = minimum_run_volume``):

        scale_i * z_i - u_i * y_i <= 0      (x_i <= u_i * y_i)
        m_i * y_i - scale_i * z_i <= 0      (x_i >= m_i * y_i)

    with y_i binary and ``u_i`` the effective cap in volume units (tightened
    caps included).  Non-gated sources carry a y_i fixed at zero instead.
    Rows live in the (z, y) variable space; ``None`` when no source gates.
    """
    n = len(sources)
    gated_idx = [
        i for i, s in enumerate(sources) if s.minimum_run_volume is not None
    ]
    if not gated_idx:
        return None, None
    rows, rhs = [], []
    for i in gated_idx:
        cap_volume = scales[i] * z_bounds[i][1]
        # x_i <= u_i * y_i
        row = np.zeros(2 * n)
        row[i] = scales[i]
        row[n + i] = -cap_volume
        rows.append(row)
        rhs.append(0.0)
        # x_i >= m_i * y_i  <=>  m_i * y_i - x_i <= 0
        row = np.zeros(2 * n)
        row[i] = -scales[i]
        row[n + i] = float(sources[i].minimum_run_volume)  # type: ignore[arg-type]
        rows.append(row)
        rhs.append(0.0)
    if min_active > 0:
        row = np.zeros(2 * n)
        for i in gated_idx:
            row[n + i] = -1.0
        rows.append(row)
        rhs.append(float(-min_active))
    return np.asarray(rows), np.asarray(rhs)


def _enable_bounds(sources: Sequence[SourceData]) -> List[Tuple[float, float]]:
    """Binary bounds for gated sources; y fixed at 0 for the others."""
    return [(0.0, 1.0) if s.minimum_run_volume is not None else (0.0, 0.0)
            for s in sources]


def _indicator_constraints(
    sources: Sequence[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    scales: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Indicator-window rows in (z, y)-space:
    ``lo_j*V <= sum(conc_ij * scale_i * z_i) <= hi_j*V``.

    The enable-variable columns are zero (indicator loads depend on volume
    only)."""
    rows, rhs = [], []
    n = len(sources)
    for name in INDICATORS:
        rng = limits.get(name)
        if rng is None:
            continue
        lo, hi = rng
        coeff = np.array([getattr(s, name) for s in sources], dtype=float) * scales
        rows.append(np.concatenate([-coeff, np.zeros(n)]))
        rhs.append(-lo * target_volume)
        rows.append(np.concatenate([coeff, np.zeros(n)]))
        rhs.append(hi * target_volume)
    if not rows:
        return None, None
    return np.asarray(rows), np.asarray(rhs)


def _build_model(
    sources: Sequence[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    caps: Optional[Mapping[str, float]] = None,
    min_active: int = 0,
) -> _Model:
    """Assemble the (mixed-integer) feasible-set description in (z, y)-space."""
    n = len(sources)
    scales = _scales(sources)
    z_bounds = _source_bounds(sources, scales, caps)
    bounds = z_bounds + _enable_bounds(sources)
    A_struct, b_struct = _activation_structure(sources, scales, z_bounds, min_active)
    A_eq = np.concatenate([scales.reshape(1, -1), np.zeros((1, n))], axis=1)
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(sources, target_volume, limits, scales)
    return _Model(
        n=n,
        scales=scales,
        integrality=np.concatenate([_integrality(sources),
                                    np.array([1.0 if s.minimum_run_volume is not None
                                              else 0.0 for s in sources])]),
        bounds=bounds,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        A_struct=A_struct,
        b_struct=b_struct,
    )


_UNSET = object()


def _join_ub(model: _Model, extra):
    """Combine the model's always-on structural rows with caller rows.

    The structural activation/count rows (``model.A_struct``) define the
    feasible set and are therefore prepended to every upper-bound system;
    ``extra`` carries the droppable indicator rows (or replacements built for
    the lexicographic/contingency refinements)."""
    if model.A_struct is None:
        return extra
    if extra is None:
        return model.A_struct
    return np.vstack([model.A_struct, extra])


def _join_rhs(model: _Model, extra):
    if model.b_struct is None:
        return extra
    if extra is None:
        return model.b_struct
    return np.append(model.b_struct, extra)


def _lp(c, model: _Model, bounds=_UNSET, A_ub=_UNSET, b_ub=_UNSET,
        A_eq=_UNSET, b_eq=_UNSET):
    """linprog wrapper forwarding the model's variable integrality.

    Constraint arguments default to the model's matrices; pass ``None``
    explicitly to drop the (droppable indicator) constraint group entirely.
    The structural activation/count rows (``A_struct``) are always enforced."""
    extra = model.A_ub if A_ub is _UNSET else A_ub
    extra_rhs = model.b_ub if b_ub is _UNSET else b_ub
    return linprog(
        c,
        A_ub=_join_ub(model, extra),
        b_ub=_join_rhs(model, extra_rhs),
        A_eq=model.A_eq if A_eq is _UNSET else A_eq,
        b_eq=model.b_eq if b_eq is _UNSET else b_eq,
        bounds=model.bounds if bounds is _UNSET else bounds,
        integrality=model.integrality,
        method="highs",
    )


# --------------------------------------------------------------------------- #
# main solve
# --------------------------------------------------------------------------- #

def solve_blend(
    sources: List[SourceData],
    target_volume: float,
    limits: Dict[str, Tuple[float, float]],
    min_active: int = 0,
) -> SolveOutput:
    """Solve the blending MIP.  ``limits`` maps indicator name -> (min, max).

    ``min_active`` (``minimum_active_sources``) requires that many sources
    carrying a ``minimum_run_volume`` to be used at or above their threshold;
    the binary enable variables live in the same MIP as the volumes."""
    sources = sorted(sources, key=lambda s: s.id)
    model = _build_model(sources, target_volume, limits, min_active=min_active)
    costs = np.array([s.cost for s in sources], dtype=float)
    c = np.concatenate([costs * model.scales, np.zeros(model.n)])

    res = _lp(c, model)
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, model, target_volume, limits, min_active=min_active)
    if res.status != 0:
        raise RuntimeError(f"MIP solver failed (status {res.status}): {res.message}")

    z = _lexicographic_min(c, res.x, model)
    x = model.scales * z
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=float(costs @ x),
        indicators=indicators,
    )


def _lexicographic_min(c, z_star, model: _Model) -> np.ndarray:
    """Among all cost-optimal solutions, minimize (z_1, ..., z_n)
    lexicographically (indices follow the id-sorted source order).  Scaling is
    non-negative so this also lexicographically minimizes (x_1, ..., x_n)."""
    # Pin the optimal cost as an equality; z_star satisfies it by construction.
    cost_value = float(c @ z_star)
    A_eq2 = np.vstack([model.A_eq, c.reshape(1, -1)])
    b_eq2 = np.append(model.b_eq, cost_value)
    try:
        return _sequential_min(model, model.A_ub, model.b_ub, A_eq2, b_eq2)
    except RuntimeError:
        # Fallback: relax the cost pin to a tight inequality.
        tol = _COST_RTOL * max(1.0, abs(cost_value))
        cost_row = c.reshape(1, -1)
        A_ub2 = cost_row if model.A_ub is None else np.vstack([model.A_ub, cost_row])
        b_ub2 = (
            np.array([cost_value + tol])
            if model.b_ub is None
            else np.append(model.b_ub, cost_value + tol)
        )
        return _sequential_min(model, A_ub2, b_ub2, model.A_eq, model.b_eq)


def _sequential_min(model: _Model, A_ub, b_ub, A_eq, b_eq,
                    bounds=None, n_coords: Optional[int] = None) -> np.ndarray:
    """Lexicographically minimize (z_0, ..., z_{n-1}) by solving one MIP per
    coordinate; ``bounds`` may contain additional variables (e.g. slack
    variables), which stay free after the first ``n`` coordinates."""
    n = model.n if n_coords is None else n_coords
    bnds = list(model.bounds if bounds is None else bounds)
    n_vars = len(bnds)
    z = np.zeros(n_vars)
    for k in range(n):
        obj = np.zeros(n_vars)
        obj[k] = 1.0
        r = _lp(obj, model, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bnds)
        if r.status != 0:
            raise RuntimeError(
                f"lexicographic refinement failed at index {k}: {r.message}"
            )
        lo0, hi0 = bnds[k]
        v = min(max(float(r.x[k]), lo0), hi0)
        if model.integrality[k]:
            # Pin integer and binary coordinates at the exact value returned
            # by the MIP.  A fractional pin (v +/- relative eps) may contain no
            # integer at all once v is large, making the next solve spuriously
            # infeasible; for binary coordinates (batch-enable flags) a loose
            # pin also lets the solver drift the linked volume slightly, which
            # shows up as a spurious contingency adjustment.
            iv = int(round(v))
            iv = max(int(math.ceil(lo0 - 1e-9)), min(int(math.floor(hi0 + 1e-9)), iv))
            z[k] = float(iv)
            bnds[k] = (float(iv), float(iv))
        else:
            # Continuous coordinates are pinned exactly as well: that pin only
            # feeds the remaining lexicographic refinements (the MIP itself
            # just returned this exact solution), whereas an interval pin wide
            # enough to matter at solver feasibility tolerance lets later
            # refinements drift when the coordinate is coupled to a binary
            # enable variable through a big-M row (the drift would surface as
            # a spurious non-zero contingency change).
            z[k] = v
            bnds[k] = (v, v)
    return z[: model.n]


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def solve_contingency(
    sources: List[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    original_volumes: Mapping[str, float],
    new_caps: Mapping[str, float],
    min_active: int = 0,
) -> ContingencyOutput:
    """Re-optimize a saved optimal plan under tightened supply caps.

    The lexicographic objective is:

      1. minimize the total absolute volume change
         ``sum_i |x_i - x_i^0|`` versus the stored allocation;
      2. among minimum-change allocations, minimize total cost;
      3. among those, minimize the usage vector (source ids ascending)
         lexicographically.

    Target volume, indicator windows, unit costs, batch sizes and
    minimum-run-volume activations (including ``minimum_active_sources``) are
    unchanged.  ``new_caps`` only needs to list sources whose cap is reduced;
    every other source keeps its original ``max_volume``.
    """
    sources = sorted(sources, key=lambda s: s.id)
    model = _build_model(
        sources, target_volume, limits, caps=new_caps, min_active=min_active
    )
    n = model.n
    scales = model.scales
    # The stored plan honors the batch lattice; snap tiny rounding noise so
    # that the L1 reference point sits exactly on it.
    x0 = np.zeros(n)
    for i, s in enumerate(sources):
        v = float(original_volumes[s.id])
        if s.batch_size is not None:
            v = float(round(v / s.batch_size) * s.batch_size)
        x0[i] = v
    costs = np.array([s.cost for s in sources], dtype=float)

    # Phase 1: minimize sum(d_i) subject to |scale_i z_i - x0_i| <= d_i.
    # Variables are ordered (z_1..z_n, y_1..y_n, d_1..d_n).
    A1, b1, A1_eq, b1_eq, bounds1, integ1 = _l1_model(x0, model)
    p1 = linprog(
        np.concatenate([np.zeros(2 * n), np.ones(n)]),
        A_ub=A1, b_ub=b1, A_eq=A1_eq, b_eq=b1_eq,
        bounds=bounds1, integrality=integ1, method="highs",
    )
    if p1.status == 2:  # provably infeasible under the new caps
        achievable, message = _analyze_infeasible(
            sources, model, target_volume, limits, caps=new_caps,
            min_active=min_active,
        )
        return ContingencyOutput(
            status="infeasible",
            achievable_ranges=achievable,
            message=message,
        )
    if p1.status != 0:
        raise RuntimeError(f"MIP solver failed (status {p1.status}): {p1.message}")
    adj_star = float(p1.fun)

    z = _contingency_lexicographic(
        np.concatenate([costs * scales, np.zeros(n)]),
        x0, adj_star, A1, b1, A1_eq, b1_eq, bounds1, integ1,
        model,
    )

    x = scales * z
    deltas = x - x0
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return ContingencyOutput(
        status="optimal",
        volumes=x,
        deltas=deltas,
        total_adjustment=float(np.abs(deltas).sum()),
        total_cost=float(costs @ x),
        cost_delta=float(costs @ (x - x0)),
        indicators=indicators,
    )


def _l1_model(x0, model: _Model):
    """Build the (z, y, d) model whose feasible set is exactly the allocations
    (z, y) satisfying the original constraints with |scale_i z_i - x0_i| <= d_i.
    Activation/count and indicator rows are embedded unchanged (their d
    columns are zero)."""
    n = model.n
    eye = np.eye(n)
    rows = [
        np.hstack([eye * model.scales.reshape(-1, 1), np.zeros((n, n)), -eye]),
        np.hstack([-eye * model.scales.reshape(-1, 1), np.zeros((n, n)), -eye]),
    ]
    rhs = [x0, -x0]
    if model.A_struct is not None:
        rows.insert(0, np.hstack([model.A_struct, np.zeros((model.A_struct.shape[0], n))]))
        rhs.insert(0, model.b_struct)
    if model.A_ub is not None:
        rows.insert(0, np.hstack([model.A_ub, np.zeros((model.A_ub.shape[0], n))]))
        rhs.insert(0, model.b_ub)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    A_eq_ext = np.hstack([model.A_eq, np.zeros((model.A_eq.shape[0], n))])
    bounds_ext = list(model.bounds) + [(0.0, None)] * n
    integ_ext = np.concatenate([model.integrality, np.zeros(n)])
    return A, b, A_eq_ext, model.b_eq, bounds_ext, integ_ext


def _contingency_lexicographic(
    c, x0, adj_star, A_l1, b_l1, A_eq_ext, b_eq_ext, bounds_ext, integ_ext,
    model: _Model,
) -> np.ndarray:
    """Phase 2+3: within the minimum-change allocations, minimize cost and
    then the id-sorted usage vector lexicographically."""
    n = model.n
    # Variable layout in the extended model is (z_1..z_n, y_1..y_n, d_1..d_n).
    change_row = np.concatenate([np.zeros(2 * n), np.ones(n)])
    cost_row = np.concatenate([c, np.zeros(n)])
    adj_tol = _FIX_RTOL * max(1.0, abs(adj_star))

    # Phase 2: minimum cost with the total change pinned at its minimum.
    A_eq2 = np.vstack([A_eq_ext, change_row])
    b_eq2 = np.append(b_eq_ext, adj_star)
    p2 = linprog(
        cost_row, A_ub=A_l1, b_ub=b_l1, A_eq=A_eq2, b_eq=b_eq2,
        bounds=bounds_ext, integrality=integ_ext, method="highs",
    )
    if p2.status != 0:  # numerical issue: relax the pin to a tight inequality
        p2 = linprog(
            cost_row,
            A_ub=np.vstack([A_l1, change_row]),
            b_ub=np.append(b_l1, adj_star + adj_tol),
            A_eq=A_eq_ext, b_eq=b_eq_ext, bounds=bounds_ext,
            integrality=integ_ext, method="highs",
        )
        if p2.status != 0:
            raise RuntimeError(
                f"contingency cost refinement failed (status {p2.status}): {p2.message}"
            )
    cost_star = float(p2.fun)

    # Phase 3: lex-min (z_1, ..., z_n) among min-change, min-cost solutions.
    A_eq3 = np.vstack([A_eq2, cost_row])
    b_eq3 = np.append(b_eq2, cost_star)
    # A_l1 already embeds the structural activation/count rows, so the wrapper
    # must not prepend them a second time: A_struct stays None on ext_model.
    ext_model = _Model(
        n=n,
        scales=model.scales,
        integrality=integ_ext,
        bounds=bounds_ext,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_struct=None,
        b_struct=None,
    )
    try:
        return _sequential_min(ext_model, A_l1, b_l1, A_eq3, b_eq3)
    except RuntimeError:
        cost_tol = _COST_RTOL * max(1.0, abs(cost_star))
        A_ub3 = np.vstack([A_l1, change_row, cost_row])
        b_ub3 = np.append(b_l1, [adj_star + adj_tol, cost_star + cost_tol])
        return _sequential_min(ext_model, A_ub3, b_ub3, A_eq_ext, b_eq_ext)


# --------------------------------------------------------------------------- #
# infeasibility analysis
# --------------------------------------------------------------------------- #

def _infeasible(sources, model, target_volume, limits,
                caps=None, min_active=0) -> SolveOutput:
    """Explain infeasibility with per-indicator achievable ranges."""
    achievable, message = _analyze_infeasible(
        sources, model, target_volume, limits, caps=caps, min_active=min_active
    )
    return SolveOutput(
        status="infeasible",
        achievable_ranges=achievable,
        message=message,
    )


def _max_attainable_volume(sources, caps, volume=None, min_active=0) -> float:
    """Largest volume attainable under the given caps, batch sizes and
    source-activation/count constraints.

    When ``volume`` is given it acts as an upper bound, so the result is the
    largest attainable volume not exceeding it."""
    m = _build_model(sources, 0.0, {}, caps=caps, min_active=min_active)
    n = m.n
    A_ub, b_ub = None, None
    if volume is not None:
        A_ub = np.concatenate([m.scales.reshape(1, -1), np.zeros((1, n))], axis=1)
        b_ub = np.array([float(volume)])
    res = _lp(
        np.concatenate([-m.scales, np.zeros(n)]),
        m, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None,
    )
    if res.status != 0:
        return 0.0
    return float(m.scales @ res.x[:n])


def _analyze_infeasible(sources, model, target_volume, limits, caps=None,
                        min_active=0):
    """Per-indicator achievable ranges and a human-readable explanation,
    projected onto the mixed-integer supply/volume feasible set.

    The projection reuses the *same* caps as the failing model (original
    supply caps on the solve path, reduced caps on the contingency path) and
    honors every batch-size and source-activation/count constraint.  When the
    exact target volume cannot be assembled, the ranges are reported at the
    largest attainable volume."""
    total_supply = sum(
        hi * scale
        for (_, hi), scale in zip(model.bounds[: model.n], model.scales)
    )
    achievable: Dict[str, dict] = {}
    messages: List[str] = []

    volume = float(target_volume)
    if total_supply < target_volume * (1.0 - 1e-9):
        messages.append(
            f"total available volume {total_supply:g} is below the target {target_volume:g}"
        )
        achievable["volume"] = {"requested": float(target_volume), "max": float(total_supply)}
        volume = float(total_supply)
    elif model.integrality.any():
        # Aggregate supply suffices, but the batch lattice and the
        # zero-or-threshold activation law (plus the active-source count) may
        # rule out the exact target volume: first test the structural feasible
        # set (volume equation with caps, batches, activations and count, but
        # no indicator windows), then fall back to the largest attainable
        # volume.
        feas = _lp(
            np.zeros(2 * model.n), model, A_ub=None, b_ub=None
        )
        if feas.status == 2:
            vmax = _max_attainable_volume(
                sources, caps, volume=target_volume, min_active=min_active
            )
            volume = vmax
            messages.append(
                f"the target volume {target_volume:g} cannot be assembled under the "
                f"batch-size and source-activation constraints; the largest "
                f"attainable volume not above it is {vmax:g}"
            )
            achievable["volume"] = {
                "requested": float(target_volume), "max": float(vmax),
            }

    if volume > 0:
        # Projection model: the same caps, batch lattice and activation/count
        # constraints, but indicator windows are deliberately excluded so the
        # min/max MIPs scan the whole structural feasible set.
        proj_model = _build_model(
            sources, volume, {}, caps=caps, min_active=min_active
        )
        n = proj_model.n
        for name in INDICATORS:
            conc = np.concatenate([
                np.array(
                    [getattr(s, name) for s in sources], dtype=float
                ) * proj_model.scales,
                np.zeros(n),
            ])
            lo_res = _lp(conc, proj_model)
            hi_res = _lp(-conc, proj_model)
            entry: Dict[str, float] = {}
            if lo_res.status == 0:
                entry["min"] = float(lo_res.fun) / volume
            if hi_res.status == 0:
                entry["max"] = float(-hi_res.fun) / volume
            req = limits.get(name)
            if req is not None:
                entry["requested_min"], entry["requested_max"] = float(req[0]), float(req[1])
                if "max" in entry and entry["max"] < req[0]:
                    messages.append(
                        f"{name}: requested minimum {req[0]:g} exceeds the achievable "
                        f"maximum {entry['max']:.6g}"
                    )
                if "min" in entry and entry["min"] > req[1]:
                    messages.append(
                        f"{name}: requested maximum {req[1]:g} is below the achievable "
                        f"minimum {entry['min']:.6g}"
                    )
            achievable[name] = entry
    else:
        messages.append("no source water is available")

    if not messages:
        messages.append(
            "indicator ranges are individually attainable but jointly incompatible "
            "under the supply, volume, batch-size and source-activation constraints"
        )
    return achievable, "; ".join(messages)
