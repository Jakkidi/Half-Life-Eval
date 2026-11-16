import math

from chl import survival as sv


def ep(oks, model="m", constraint="c", pressure="tempting", episode=0, max_turns=None, kinds=None):
    T = max_turns or len(oks)
    fv = next((i + 1 for i, o in enumerate(oks) if not o), None)
    kinds = kinds or ["neutral"] * len(oks)
    return {"model": model, "constraint": constraint, "pressure": pressure, "episode": episode,
            "max_turns": T, "first_violation": fv,
            "turns": [{"t": i + 1, "ok": o, "kind": k} for i, (o, k) in enumerate(zip(oks, kinds))]}


def test_km_matches_hand_calculation():
    # events at 2, 2, 4; censored at 3 and 5
    data = [(2, True), (2, True), (3, False), (4, True), (5, False)]
    km = sv.kaplan_meier(data, 5)
    assert km.surv[1] == 1.0
    assert math.isclose(km.surv[2], 3 / 5)               # 5 at risk, 2 events
    assert math.isclose(km.surv[3], 3 / 5)               # censoring does not drop S
    assert math.isclose(km.surv[4], 3 / 5 * (1 - 1 / 2))  # 2 at risk, 1 event
    assert km.at_risk[4] == 2
    assert all(lo <= s <= hi for s, lo, hi in zip(km.surv, km.lo, km.hi))


def test_half_life_and_rmst():
    data = [(1, True), (2, True), (3, True), (4, True)]
    km = sv.kaplan_meier(data, 4)
    assert sv.half_life(km)[0] == 2                      # S(2) = 0.5
    assert math.isclose(sv.rmst(km, 4), 0.75 + 0.5 + 0.25 + 0.0)


def test_half_life_not_reached_is_none():
    km = sv.kaplan_meier([(10, False)] * 9 + [(3, True)], 10)
    assert sv.half_life(km)[0] is None
    assert sv.rmst(km, 10) > 9


def test_durations_censors_clean_episodes():
    recs = [ep([True, True, False]), ep([True, True, True, True])]
    assert sv.durations(recs) == [(3, True), (4, False)]


def test_contagion_detects_sticky_violations():
    sticky = [ep([True, True, False, False, False, False], episode=i) for i in range(20)]
    sticky += [ep([True] * 6, episode=100 + i) for i in range(20)]
    c = sv.contagion(sticky)
    assert c["ratio"] > 5 and c["recovery_rate"] < 0.2

    flicker = [ep([True, False, True, False, True, True], episode=i) for i in range(20)]
    assert sv.contagion(flicker)["ratio"] < 1


def test_logrank_separates_obviously_different_groups():
    early = [(2, True)] * 30
    late = [(10, False)] * 30
    chi2, p = sv.logrank(early, late)
    assert p < 1e-6
    chi2, p = sv.logrank(early, early)
    assert p > 0.9


def test_paired_rmst_delta_sign():
    control = [ep([True, False, True, True], episode=i, max_turns=4) for i in range(15)]
    treat = [ep([True, True, True, True], episode=i, max_turns=4) for i in range(15)]
    for r in treat: r["intervention"] = "reminder2"
    d = sv.paired_rmst_delta(treat, control, n_boot=200)
    assert d["n_pairs"] == 15 and d["delta"] > 0 and d["lo"] > 0


def test_truncation_flag():
    r = ep([True, True]); r["key"] = "k"; r["context_window"] = 1000
    r["turns"][1]["est_prompt_tokens"] = 900
    assert [f["t"] for f in sv.truncation_risk([r])] == [2]


def test_violation_rate_by_turn_counts_every_slip():
    recs = [ep([True, False, False]), ep([False, True, False], episode=1)]
    rates = [x["rate"] for x in sv.violation_rate_by_turn(recs)]
    assert rates == [0.5, 0.5, 1.0]


def test_paired_rate_delta_and_placebo_window():
    control = [ep([True, True, False, False, False, False], episode=i) for i in range(10)]
    treat = [ep([True, True, False, True, True, True], episode=i) for i in range(10)]
    after = sv.paired_rate_delta(treat, control, 3, 6, n_boot=200)
    before = sv.paired_rate_delta(treat, control, 1, 2, n_boot=200)
    assert after["delta"] == -0.75 and after["n_pairs"] == 10
    assert before["delta"] == 0.0
