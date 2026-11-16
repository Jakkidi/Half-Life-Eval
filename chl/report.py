"""Builds a single self-contained HTML report from a results JSONL file."""

from __future__ import annotations

import html
import json
import time
from collections import defaultdict
from pathlib import Path

from . import __version__
from . import survival as sv
from .constraints import CONSTRAINTS
from .scenarios import PRESSURES

HEADLINE_PRESSURE = "tempting"


def _fmt_half(h: tuple, H: int) -> str:
    """Half-life cell: median turn with its 95% CI; '>H' when not reached."""
    m, lo, hi = h
    if m is None:
        return f"&gt;{H}" if lo is None else f"&gt;{H} <span class=ci>(≥{lo})</span>"
    lo_s = "?" if lo is None else lo
    hi_s = f"&gt;{H}" if hi is None else hi
    return f"{m} <span class=ci>[{lo_s}–{hi_s}]</span>"


def analyze(records: list[dict], n_boot: int = 1000) -> dict:
    if not records:
        raise ValueError("no records to report on")
    H = max(r["max_turns"] for r in records)
    models = sorted({r["model"] for r in records})  # sorted, so a model keeps its color across reports
    pressures = [p for p in PRESSURES if any(r["pressure"] == p for r in records)]
    constraints = [c for c in CONSTRAINTS if any(r["constraint"] == c for r in records)]
    interventions = sorted({r["intervention"] for r in records}, key=lambda i: (i != "none", i))
    base = [r for r in records if r["intervention"] == "none"] or records
    base_iv = "none" if any(r["intervention"] == "none" for r in records) else interventions[0]

    groups = defaultdict(list)
    for r in base:
        groups[(r["model"], r["pressure"])].append(r)

    km_data = {}
    summary = {}
    for m in models:
        km_data[m] = {}
        for p in pressures:
            rs = groups.get((m, p), [])
            if not rs:
                continue
            km = sv.kaplan_meier(sv.durations(rs), H)
            half = sv.half_life(km)
            rm = sv.rmst(km, H)
            rm_ci = sv.bootstrap_ci(rs, lambda s: sv.rmst(sv.kaplan_meier(sv.durations(s), H), H),
                                    n_boot=n_boot, cluster_key=lambda r: (r["constraint"], r["episode"]))
            km_data[m][p] = {
                "surv": km.surv, "lo": km.lo, "hi": km.hi, "at_risk": km.at_risk,
                "events": km.events, "n": km.n, "half": list(half), "rmst": rm,
            }
            summary[(m, p)] = {"half": half, "rmst": rm, "rmst_ci": rm_ci, "n": km.n,
                               "survived": sum(1 for r in rs if r["first_violation"] is None) / len(rs)}

    heat = {p: {m: {} for m in models} for p in pressures}
    for p in pressures:
        for m in models:
            for c in constraints:
                rs = [r for r in groups.get((m, p), []) if r["constraint"] == c]
                if rs:
                    km = sv.kaplan_meier(sv.durations(rs), H)
                    heat[p][m][c] = {"rmst": round(sv.rmst(km, H), 2), "half": sv.half_life(km)[0],
                                     "n": len(rs),
                                     "broke": sum(1 for r in rs if r["first_violation"] is not None)}

    contagion = {}
    for m in models:
        rs = [r for r in base if r["model"] == m]
        c = sv.contagion(rs)
        c["ci"] = sv.bootstrap_ci(rs, lambda s: sv.contagion(s)["ratio"], n_boot=max(200, n_boot // 2))
        contagion[m] = c

    reminder = []
    for iv in interventions:
        if iv == base_iv:
            continue
        for m in models:
            t = [r for r in records if r["model"] == m and r["intervention"] == iv]
            c = [r for r in records if r["model"] == m and r["intervention"] == base_iv]
            if t and c:
                d = sv.paired_rmst_delta(t, c, n_boot=n_boot)
                k = int(iv[8:]) if iv.startswith("reminder") and iv[8:].isdigit() else 1
                before = sv.paired_rate_delta(t, c, 1, k - 1, n_boot=n_boot) if k > 1 else None
                after = sv.paired_rate_delta(t, c, k, H, n_boot=n_boot)
                reminder.append({"model": m, "intervention": iv, "k": k, **d,
                                 "rate_before": before, "rate_after": after})

    # Violation rate by turn, per model and intervention, at the headline pressure.
    hp = HEADLINE_PRESSURE if HEADLINE_PRESSURE in pressures else pressures[0]
    drift = {m: {iv: sv.violation_rate_by_turn(
                    [r for r in records if r["model"] == m and r["intervention"] == iv
                     and r["pressure"] == hp], H)
                 for iv in interventions
                 if any(r["model"] == m and r["intervention"] == iv and r["pressure"] == hp for r in records)}
             for m in models}

    examples = []
    for c in constraints:
        hits = sorted(
            ((x["t"], r, x) for r in base if r["constraint"] == c
             for x in r["turns"] if not x["ok"] and x["t"] == r["first_violation"]),
            key=lambda z: (z[0], z[1]["key"]),
        )
        if hits:
            t, r, x = hits[len(hits) // 2]  # a typical first slip, not the most extreme
            examples.append({"constraint": c, "model": r["model"], "pressure": r["pressure"],
                             "t": t, "user": x["user"], "reply": x["reply"], "reason": x["reason"]})

    trunc = sv.truncation_risk(records)
    providers = sorted({r["provider"] for r in records})
    return {
        "H": H, "models": models, "pressures": pressures, "constraints": constraints,
        "interventions": interventions, "km": km_data, "summary": summary, "heat": heat,
        "contagion": contagion, "reminder": reminder, "drift": drift, "examples": examples,
        "truncation": trunc, "providers": providers, "n_records": len(records),
        "n_turns": sum(len(r["turns"]) for r in records),
        "temperature": sorted({r["temperature"] for r in records}),
        "context_window": sorted({r.get("context_window") for r in records if r.get("context_window")}),
    }


def render(a: dict, title: str = "Constraint Half-Life") -> str:
    H, models = a["H"], a["models"]
    esc = html.escape
    mock = "mock" in a["providers"]
    hp = HEADLINE_PRESSURE if HEADLINE_PRESSURE in a["pressures"] else a["pressures"][0]

    # ---- hero tiles: half-life under the headline pressure
    tiles = []
    for i, m in enumerate(models):
        s = a["summary"].get((m, hp))
        if not s:
            continue
        med = s["half"][0]
        big = f"{med}" if med is not None else f"&gt;{H}"
        unit = "turns" if med is not None else "turns (not reached)"
        tiles.append(
            f'<div class="tile"><div class="tile-k"><i class="key" style="background:var(--series-{i % 8 + 1})"></i>{esc(m)}</div>'
            f'<div class="tile-v">{big}<span class="tile-u"> {unit}</span></div>'
            f'<div class="tile-s">RMST {s["rmst"]:.1f} of {H} · {s["survived"]:.0%} never broke</div></div>'
        )

    # ---- leaderboard
    rows = []
    for m in models:
        cells = [f"<td class=lbl>{esc(m)}</td>"]
        for p in a["pressures"]:
            s = a["summary"].get((m, p))
            cells.append(f"<td>{_fmt_half(s['half'], H) if s else '–'}</td>")
        s = a["summary"].get((m, hp))
        if s:
            lo, hi = s["rmst_ci"]
            cells.append(f"<td>{s['rmst']:.1f} <span class=ci>[{lo:.1f}–{hi:.1f}]</span></td>")
        else:
            cells.append("<td>–</td>")
        c = a["contagion"][m]
        clo, chi = c["ci"]
        cells.append(f"<td>{c['ratio']:.1f}× <span class=ci>[{clo:.1f}–{chi:.1f}]</span></td>")
        cells.append(f"<td>{c['recovery_rate']:.0%}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    pressure_heads = "".join(f"<th>Half-life · {esc(p)}</th>" for p in a["pressures"])
    leaderboard = (
        f"<table class=data><thead><tr><th>Model</th>{pressure_heads}<th>RMST · {hp}</th>"
        f"<th>Contagion</th><th>Recovery</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )

    # ---- reminder table
    def _pp(d: dict | None) -> str:
        if not d or d["n_pairs"] == 0:
            return "–"
        return (f"{d['delta'] * 100:+.0f} pts <span class=ci>[{d['lo'] * 100:+.0f}, "
                f"{d['hi'] * 100:+.0f}]</span>")

    if a["reminder"]:
        rrows = "".join(
            f"<tr><td class=lbl>{esc(r['model'])}</td><td>{esc(r['intervention'])}</td>"
            f"<td>{_pp(r['rate_after'])}</td><td>{_pp(r['rate_before'])}</td>"
            f"<td>{r['delta']:+.2f} <span class=ci>[{r['lo']:+.2f}, {r['hi']:+.2f}]</span></td>"
            f"<td>{r['logrank_p']:.2g}</td><td>{r['n_pairs']}</td></tr>" for r in a["reminder"]
        )
        reminder_table = (
            "<table class=data><thead><tr><th>Model</th><th>Intervention</th>"
            "<th>Δ violation rate, from first reminder</th><th>Δ before first reminder (placebo)</th>"
            "<th>ΔRMST to first slip</th><th>Log-rank p</th><th>Pairs</th></tr></thead>"
            f"<tbody>{rrows}</tbody></table>"
        )
    else:
        reminder_table = "<p class=muted>No intervention arm in this run. Add <code>--interventions none,reminder5</code>.</p>"

    # ---- KM table view
    km_rows = []
    for m in models:
        for p in a["pressures"]:
            d = a["km"][m].get(p)
            if not d:
                continue
            pts = " · ".join(f"t{t}: {d['surv'][t]:.0%}" for t in range(0, H + 1, 5) if t <= H)
            km_rows.append(f"<tr><td class=lbl>{esc(m)}</td><td>{esc(p)}</td><td>{d['n']}</td><td>{pts}</td></tr>")
    km_table = ("<table class=data><thead><tr><th>Model</th><th>Pressure</th><th>Episodes</th>"
                f"<th>Still compliant after turn t</th></tr></thead><tbody>{''.join(km_rows)}</tbody></table>")

    # ---- heat table view
    heat_rows = []
    for p in a["pressures"]:
        for m in models:
            cells = "".join(
                f"<td>{a['heat'][p][m][c]['rmst']:.1f}</td>" if c in a["heat"][p][m] else "<td>–</td>"
                for c in a["constraints"]
            )
            heat_rows.append(f"<tr><td>{esc(p)}</td><td class=lbl>{esc(m)}</td>{cells}</tr>")
    heat_table = ("<table class=data><thead><tr><th>Pressure</th><th>Model</th>"
                  + "".join(f"<th>{esc(CONSTRAINTS[c].title)}</th>" for c in a["constraints"])
                  + f"</tr></thead><tbody>{''.join(heat_rows)}</tbody></table>")

    # ---- examples
    ex_html = "".join(
        f"<details class=ex><summary><b>{esc(CONSTRAINTS[e['constraint']].title)}</b> · {esc(e['model'])} · "
        f"{esc(e['pressure'])} · broke at turn {e['t']} <span class=muted>({esc(e['reason'])})</span></summary>"
        f"<div class=turn><div class=who>User</div><pre>{esc(e['user'][:600])}</pre></div>"
        f"<div class=turn><div class=who>Model</div><pre>{esc(e['reply'][:900])}{'…' if len(e['reply']) > 900 else ''}</pre></div>"
        f"</details>"
        for e in a["examples"]
    ) or "<p class=muted>No violations recorded.</p>"

    trunc = a["truncation"]
    trunc_html = (
        '<div class="note">This provider reports no context window, so truncation was not checked.</div>'
        if not a["context_window"] else
        f'<div class="note warn"><b>Context-window warning.</b> {len(trunc)} turns used ≥85% of the '
        f"configured context window ({', '.join(map(str, a['context_window']))} tokens). Ollama drops the "
        "start of an overflowing prompt, which includes the system rule, so breaks at those turns may be "
        "truncation rather than drift. Re-run with a larger <code>--num-ctx</code>.</div>"
        if trunc else
        '<div class="note">No turn came within 85% of the configured context window, so breaks are not '
        "explained by the system prompt being truncated.</div>"
    )

    mock_banner = (
        '<div class="note warn"><b>Simulated data.</b> This report was produced by the built-in mock '
        "provider, whose hazard is set by hand. It demonstrates the pipeline and checks that the statistics "
        "recover known effects. It says nothing about any real model.</div>" if mock else ""
    )

    payload = {
        "H": H, "models": models, "pressures": a["pressures"], "headline": hp,
        "constraints": [{"id": c, "title": CONSTRAINTS[c].title, "rule": CONSTRAINTS[c].rule} for c in a["constraints"]],
        "km": a["km"], "heat": a["heat"], "reminder": a["reminder"], "drift": a["drift"],
        "interventions": a["interventions"],
    }
    data_json = json.dumps(payload).replace("</", "<\\/")
    generated = time.strftime("%Y-%m-%d %H:%M")
    temps = ", ".join(map(str, a["temperature"]))

    fields = dict(
        title=esc(title), mock_banner=mock_banner, tiles="".join(tiles), hp=esc(hp), H=H,
        leaderboard=leaderboard, km_table=km_table, heat_table=heat_table,
        reminder_table=reminder_table, examples=ex_html, trunc=trunc_html,
        n_records=a["n_records"], n_turns=a["n_turns"], n_models=len(models),
        n_constraints=len(a["constraints"]), providers=esc(", ".join(a["providers"])),
        temps=esc(temps), generated=generated, version=__version__, data=data_json,
    )
    page = _TEMPLATE
    for k, v in fields.items():
        page = page.replace("%%" + k + "%%", str(v))
    return page


def build(results: Path, out: Path, n_boot: int = 1000, title: str = "Constraint Half-Life",
          models: list[str] | None = None) -> Path:
    from .runner import load_records
    recs = load_records(results)
    if models:
        recs = [r for r in recs if r["model"] in models]
        if not recs:
            raise ValueError(f"no records for models {models}")
    a = analyze(recs, n_boot=n_boot)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(a, title), encoding="utf-8")
    return out


_TEMPLATE = (Path(__file__).with_name("report_template.html")).read_text(encoding="utf-8")
