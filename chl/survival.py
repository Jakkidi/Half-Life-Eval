"""Survival analysis for rule-following, standard library only.

Each episode yields a time-to-event: the turn of the first violation, or
"censored" if the model held the rule for every turn we ran. Treating it as
survival data (rather than averaging a pass rate) handles censoring properly
and lets us ask *when* models break, not just *whether*.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

Z95 = 1.959964


@dataclass
class KM:
    times: list[int]          # 0..max_t
    surv: list[float]         # S(t) after turn t (S(0) = 1)
    lo: list[float]           # 95% CI (log-log transform)
    hi: list[float]
    at_risk: list[int]        # n at risk entering turn t
    events: list[int]         # violations at turn t
    n: int
    max_t: int


def durations(records: list[dict]) -> list[tuple[int, bool]]:
    """(time, event) per episode. Censored episodes get time = turns observed."""
    out = []
    for r in records:
        fv = r.get("first_violation")
        out.append((fv, True) if fv is not None else (len(r["turns"]), False))
    return out


def kaplan_meier(data: list[tuple[int, bool]], max_t: int | None = None) -> KM:
    if not data:
        raise ValueError("no episodes")
    max_t = max_t or max(t for t, _ in data)
    s, var_sum = 1.0, 0.0
    times, surv, lo, hi, at_risk, events = [0], [1.0], [1.0], [1.0], [len(data)], [0]
    for t in range(1, max_t + 1):
        n = sum(1 for d, _ in data if d >= t)
        d = sum(1 for d_, e in data if e and d_ == t)
        if n > 0 and d > 0:
            s *= 1 - d / n
            if n > d:
                var_sum += d / (n * (n - d))
        l, h = _loglog_ci(s, var_sum)
        times.append(t); surv.append(s); lo.append(l); hi.append(h)
        at_risk.append(n); events.append(d)
    return KM(times, surv, lo, hi, at_risk, events, len(data), max_t)


def _loglog_ci(s: float, var_sum: float) -> tuple[float, float]:
    if s <= 0.0:
        return 0.0, 0.0
    if s >= 1.0 or var_sum == 0.0:
        return s, s
    ls = math.log(s)
    se = math.sqrt(var_sum) / abs(ls)
    a = math.log(-ls)
    return math.exp(-math.exp(a + Z95 * se)), math.exp(-math.exp(a - Z95 * se))


def half_life(km: KM) -> tuple[int | None, int | None, int | None]:
    """Median survival turn with Brookmeyer-Crowley style CI from the bands.

    Returns (median, ci_low, ci_high); None means "not reached within max_t".
    """
    def first_below(curve: list[float]) -> int | None:
        for t, v in zip(km.times, curve):
            if t > 0 and v <= 0.5:
                return t
        return None
    return first_below(km.surv), first_below(km.lo), first_below(km.hi)


def rmst(km: KM, horizon: int | None = None) -> float:
    """Restricted mean survival time: expected compliant turns within the horizon.

    Always defined, even when fewer than half the episodes broke, which makes
    it the better single number for ranking. A turn counts if the model made it
    through that turn, so RMST = sum_{t=1..H} S(t).
    """
    h = horizon or km.max_t
    return sum(km.surv[t] for t in range(1, min(h, km.max_t) + 1))


def hazard(km: KM) -> list[tuple[int, float, int]]:
    """Per-turn hazard d_t / n_t with the at-risk count (for display/weighting)."""
    return [(t, km.events[t] / km.at_risk[t], km.at_risk[t])
            for t in km.times[1:] if km.at_risk[t] > 0]


def contagion(records: list[dict], alpha: float = 0.5) -> dict:
    """Does a violation make the next turn more likely to violate?

    Naive ratio: P(violate_{t+1} | violated_t) / P(violate_{t+1} | complied_t).
    That is confounded: slips cluster on tempting turns and late turns, and the
    turn after a slip has its own kind and position. So the headline `ratio`
    is a Mantel-Haenszel risk ratio stratified by (next turn's kind, turn
    third). Needs episodes that kept running after a slip.
    """
    strata: dict = {}
    for r in records:
        T = r["max_turns"]
        tl = r["turns"]
        for a, b in zip(tl, tl[1:]):
            k = (b["kind"], (b["t"] - 1) * 3 // T)
            s = strata.setdefault(k, [0, 0, 0, 0])  # vv, vo, ov, oo
            s[(0 if not a["ok"] else 2) + (0 if not b["ok"] else 1)] += 1
    vv = sum(s[0] for s in strata.values()); vo = sum(s[1] for s in strata.values())
    ov = sum(s[2] for s in strata.values()); oo = sum(s[3] for s in strata.values())
    p_after_v = (vv + alpha) / (vv + vo + 2 * alpha)
    p_after_o = (ov + alpha) / (ov + oo + 2 * alpha)
    num = den = 0.0
    for a, b, c, d in strata.values():
        n1, n0, N = a + b, c + d, a + b + c + d
        if n1 == 0 or n0 == 0:
            continue  # stratum carries no comparison
        num += (a + alpha / 2) * n0 / N
        den += (c + alpha / 2) * n1 / N
    return {
        "ratio": num / den if den else float("nan"),
        "naive_ratio": p_after_v / p_after_o,
        "p_violate_after_violation": p_after_v,
        "p_violate_after_compliance": p_after_o,
        "recovery_rate": 1 - p_after_v,
        "n_after_violation": vv + vo,
        "n_after_compliance": ov + oo,
    }


def bootstrap_ci(records: list[dict], stat, n_boot: int = 1000, seed: int = 0,
                 cluster_key=None) -> tuple[float, float]:
    """Percentile CI by resampling episodes (or clusters of them)."""
    rng = random.Random(seed)
    if cluster_key:
        groups: dict = {}
        for r in records:
            groups.setdefault(cluster_key(r), []).append(r)
        units = list(groups.values())
    else:
        units = [[r] for r in records]
    vals = []
    for _ in range(n_boot):
        sample = [r for u in (rng.choice(units) for _ in units) for r in u]
        try:
            vals.append(stat(sample))
        except (ValueError, ZeroDivisionError):
            continue
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def logrank(a: list[tuple[int, bool]], b: list[tuple[int, bool]]) -> tuple[float, float]:
    """Two-group log-rank test. Returns (chi2, p)."""
    times = sorted({t for t, e in a + b if e})
    o_minus_e = var = 0.0
    for t in times:
        na = sum(1 for d, _ in a if d >= t); nb = sum(1 for d, _ in b if d >= t)
        da = sum(1 for d, e in a if e and d == t); db = sum(1 for d, e in b if e and d == t)
        n, d = na + nb, da + db
        if n < 2:
            continue
        o_minus_e += da - d * na / n
        var += d * (na / n) * (nb / n) * (n - d) / (n - 1)
    if var == 0:
        return 0.0, 1.0
    chi2 = o_minus_e ** 2 / var
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def paired_rmst_delta(treat: list[dict], control: list[dict], n_boot: int = 1000,
                      seed: int = 0) -> dict:
    """RMST(treat) - RMST(control) for episodes sharing the same script.

    Pairs are matched on (model, constraint, pressure, episode), so both arms
    saw identical user turns; bootstrap resamples pairs.
    """
    key = lambda r: (r["model"], r["constraint"], r["pressure"], r["episode"])
    cmap = {key(r): r for r in control}
    pairs = [(r, cmap[key(r)]) for r in treat if key(r) in cmap]
    if not pairs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    H = max(r["max_turns"] for r, _ in pairs)

    def delta(ps):
        t = kaplan_meier(durations([p[0] for p in ps]), H)
        c = kaplan_meier(durations([p[1] for p in ps]), H)
        return rmst(t, H) - rmst(c, H)

    rng = random.Random(seed)
    boots = sorted(delta([rng.choice(pairs) for _ in pairs]) for _ in range(n_boot))
    chi2, p = logrank(durations([x for x, _ in pairs]), durations([y for _, y in pairs]))
    return {"delta": delta(pairs), "lo": boots[int(0.025 * (n_boot - 1))],
            "hi": boots[int(0.975 * (n_boot - 1))], "n_pairs": len(pairs), "logrank_p": p}


def truncation_risk(records: list[dict], margin: float = 0.85) -> list[dict]:
    """Turns whose prompt used >= margin of the context window (system prompt at risk)."""
    flagged = []
    for r in records:
        win = r.get("context_window")
        if not win:
            continue
        for x in r["turns"]:
            pt = max(x.get("prompt_tokens") or 0, x.get("est_prompt_tokens") or 0)
            if pt >= margin * win:
                flagged.append({"key": r["key"], "t": x["t"], "prompt_tokens": pt, "window": win})
    return flagged


# --------------------------------------------------------------------------
# Per-turn view. First-violation survival can't see what happens *after* the
# first slip, which is exactly where a periodic reminder acts. These measure
# the share of turns that violate, over the whole conversation.
# --------------------------------------------------------------------------


def violation_rate_by_turn(records: list[dict], H: int | None = None) -> list[dict]:
    """Share of episodes violating at each turn t (all turns, not just the first slip)."""
    H = H or max(r["max_turns"] for r in records)
    out = []
    for t in range(1, H + 1):
        xs = [x for r in records for x in r["turns"] if x["t"] == t]
        if xs:
            out.append({"t": t, "rate": sum(not x["ok"] for x in xs) / len(xs), "n": len(xs)})
    return out


def _rate(r: dict, lo: int, hi: int) -> float | None:
    xs = [x for x in r["turns"] if lo <= x["t"] <= hi]
    return sum(not x["ok"] for x in xs) / len(xs) if xs else None


def paired_rate_delta(treat: list[dict], control: list[dict], lo: int, hi: int,
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean paired difference in violation rate over turns lo..hi (treat - control).

    Pairs share (model, constraint, pressure, episode), hence the same user
    turns. Bootstrap resamples pairs. Run it on the turns *before* the first
    reminder as a placebo: both arms saw identical prompts there, so any
    difference is sampling noise and calibrates the post-reminder effect.
    """
    key = lambda r: (r["model"], r["constraint"], r["pressure"], r["episode"])
    cmap = {key(r): r for r in control}
    diffs = []
    for r in treat:
        c = cmap.get(key(r))
        if c is None:
            continue
        a, b = _rate(r, lo, hi), _rate(c, lo, hi)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return {"delta": sum(diffs) / len(diffs), "lo": boots[int(0.025 * (n_boot - 1))],
            "hi": boots[int(0.975 * (n_boot - 1))], "n_pairs": len(diffs)}
