"""End-to-end on the mock provider, including a check that the stats recover
the effects the mock was built with."""

import json

from chl import constraints, providers, report, runner
from chl import survival as sv
from chl.scenarios import build_schedule


def test_schedules_are_deterministic_and_paired():
    c = constraints.CONSTRAINTS["no_digits"]
    a = build_schedule(c, "tempting", 3, 20, seed=7)
    assert a == build_schedule(c, "tempting", 3, 20, seed=7)
    assert a != build_schedule(c, "tempting", 4, 20, seed=7)
    assert {x["kind"] for x in a} == {"neutral", "tempt"}
    assert {x["kind"] for x in build_schedule(c, "benign", 0, 20)} == {"neutral"}
    assert "adversarial" in {x["kind"] for x in build_schedule(c, "adversarial", 0, 20)}


def test_mock_grid_resume_and_report(tmp_path):
    out = tmp_path / "r.jsonl"
    kw = dict(models=["mock-steady", "mock-drifty"], constraints=constraints.get("all"),
              pressures=["benign", "tempting"], interventions=["none", "reminder3"],
              episodes=6, turns=12, out=out, quiet=True)
    runner.run_grid(providers.MockProvider(), **kw)
    n = len(out.read_text().splitlines())
    assert n == 2 * 8 * 2 * 2 * 6
    runner.run_grid(providers.MockProvider(), **kw)       # resume: nothing new
    assert len(out.read_text().splitlines()) == n

    recs = runner.load_records(out)
    r0 = recs[0]
    assert r0["turns"][0]["t"] == 1 and "reply" in r0["turns"][0]
    json.dumps(r0)  # serialisable

    html = report.build(out, tmp_path / "r.html", n_boot=100).read_text()
    assert "Simulated data" in html and "mock-drifty" in html and "%%" not in html


def test_stats_recover_mock_ground_truth(tmp_path):
    out = tmp_path / "g.jsonl"
    runner.run_grid(providers.MockProvider(), models=["mock-steady", "mock-drifty"],
                    constraints=constraints.get("all"), pressures=["benign", "tempting"],
                    interventions=["none"], episodes=10, turns=20, out=out, quiet=True)
    recs = runner.load_records(out)

    def rm(model, pressure):
        rs = [r for r in recs if r["model"] == model and r["pressure"] == pressure]
        return sv.rmst(sv.kaplan_meier(sv.durations(rs), 20), 20)

    # drifty model breaks sooner; temptation shortens survival for both
    assert rm("mock-drifty", "tempting") < rm("mock-steady", "tempting")
    for m in ("mock-steady", "mock-drifty"):
        assert rm(m, "tempting") < rm(m, "benign")
    # both mocks have positive built-in contagion
    for m in ("mock-steady", "mock-drifty"):
        assert sv.contagion([r for r in recs if r["model"] == m])["ratio"] > 1.5
