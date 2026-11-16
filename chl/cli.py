"""Command line: `chl run`, `chl report`, `chl demo`, `chl rules`, `chl check`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import constraints as C
from . import providers, report, runner
from .scenarios import PRESSURES

PRESETS = {
    # model calls per model, no-reminder arm only: quick 360, standard 1920, full 4800
    "quick": dict(constraints="no_digits,word_limit,json_only,no_lists", pressures="tempting",
                  episodes=6, turns=15),
    "standard": dict(constraints="all", pressures="benign,tempting", episodes=6, turns=20),
    "full": dict(constraints="all", pressures="benign,tempting,adversarial", episodes=10, turns=20),
}


def _split(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def cmd_run(a: argparse.Namespace) -> None:
    preset = PRESETS[a.preset]
    cons = C.get(a.constraints or preset["constraints"])
    pressures = _split(a.pressures or preset["pressures"])
    bad = [p for p in pressures if p not in PRESSURES]
    if bad:
        sys.exit(f"unknown pressure(s) {bad}; choose from {PRESSURES}")
    prov = providers.make(a.provider, host=a.host, num_ctx=a.num_ctx, timeout=a.timeout)
    models = _split(a.models)
    if a.provider == "ollama":
        try:
            have = prov.list_models()  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Can't reach Ollama at {a.host} ({e}). Start it with `ollama serve`.")
        if models == ["auto"]:
            chat = [m for m in have if "embed" not in m.lower()]
            if not chat:
                sys.exit("No chat models pulled. Run e.g. `ollama pull llama3.1:8b`.")
            models = chat[:1]
            print(f"--models auto: using {models[0]}", file=sys.stderr)
        missing = [m for m in models if m not in have and f"{m}:latest" not in have]
        if missing:
            sys.exit(f"Not pulled: {missing}. Run `ollama pull <model>`. Available: {have}")
    out = Path(a.out)
    runner.run_grid(
        prov, models=models, constraints=cons, pressures=pressures,
        interventions=_split(a.interventions), episodes=a.episodes or preset["episodes"],
        turns=a.turns or preset["turns"], out=out, seed=a.seed, temperature=a.temperature,
        workers=a.workers, stop_on_violation=a.stop_on_violation,
    )
    print(f"Results: {out}")
    if not a.no_report:
        html = report.build(out, out.with_suffix(".html"), n_boot=a.boot)
        print(f"Report:  {html}")


def cmd_report(a: argparse.Namespace) -> None:
    out = Path(a.out) if a.out else Path(a.results).with_suffix(".html")
    report.build(Path(a.results), out, n_boot=a.boot, title=a.title,
                 models=_split(a.models) if a.models else None)
    print(f"Report: {out}")


def cmd_demo(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.unlink(missing_ok=True)
    runner.run_grid(
        providers.MockProvider(), models=list(providers.MOCK_MODELS), constraints=C.get("all"),
        pressures=list(PRESSURES), interventions=["none", "reminder5"], episodes=10, turns=20,
        out=out, seed=a.seed, quiet=True,
    )
    html = report.build(out, out.with_suffix(".html"), n_boot=a.boot,
                        title="Constraint Half-Life · simulated demo")
    print(f"Mock results: {out}\nReport:       {html}")


def cmd_rules(_: argparse.Namespace) -> None:
    for c in C.CONSTRAINTS.values():
        print(f"{c.id:16} {c.rule}")


def cmd_check(a: argparse.Namespace) -> None:
    c = C.CONSTRAINTS[a.constraint]
    text = sys.stdin.read() if a.text == "-" else a.text
    ok, why = c.check(text)
    print("PASS" if ok else f"FAIL: {why}")
    sys.exit(0 if ok else 1)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="chl", description="Constraint Half-Life: how long do LLMs keep a system-prompt rule?")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the eval grid")
    r.add_argument("--provider", default="ollama", choices=["ollama", "mock"])
    r.add_argument("--models", required=True, help="comma-separated, e.g. llama3.1:8b,qwen2.5:7b, or 'auto' for the first pulled model")
    r.add_argument("--preset", default="quick", choices=list(PRESETS))
    r.add_argument("--constraints", help="comma-separated rule ids or 'all' (overrides preset)")
    r.add_argument("--pressures", help="comma-separated: benign,tempting,adversarial (overrides preset)")
    r.add_argument("--interventions", default="none,reminder5", help="'none' and/or reminderK")
    r.add_argument("--episodes", type=int, help="episodes per cell (overrides preset)")
    r.add_argument("--turns", type=int, help="max turns per episode (overrides preset)")
    r.add_argument("--out", default="results/run.jsonl")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--workers", type=int, default=1, help="parallel episodes (match OLLAMA_NUM_PARALLEL)")
    r.add_argument("--host", default="http://localhost:11434")
    r.add_argument("--num-ctx", type=int, default=16384, dest="num_ctx")
    r.add_argument("--timeout", type=float, default=300.0,
                   help="seconds the model may go silent mid-reply before a retry (default 300)")
    r.add_argument("--stop-on-violation", action="store_true", help="cheaper, but no contagion data")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    r.set_defaults(fn=cmd_run)

    rep = sub.add_parser("report", help="build the HTML report from a results file")
    rep.add_argument("results")
    rep.add_argument("-o", "--out")
    rep.add_argument("--title", default="Constraint Half-Life")
    rep.add_argument("--boot", type=int, default=1000)
    rep.add_argument("--models", help="only include these models (comma-separated)")
    rep.set_defaults(fn=cmd_report)

    d = sub.add_parser("demo", help="offline run with simulated models + report")
    d.add_argument("--out", default="results/demo_mock.jsonl")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--boot", type=int, default=500)
    d.set_defaults(fn=cmd_demo)

    sub.add_parser("rules", help="list the rules").set_defaults(fn=cmd_rules)

    ck = sub.add_parser("check", help="grade one reply against one rule")
    ck.add_argument("constraint", choices=list(C.CONSTRAINTS))
    ck.add_argument("text", help="reply text, or '-' for stdin")
    ck.set_defaults(fn=cmd_check)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
