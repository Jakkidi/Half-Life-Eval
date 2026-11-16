"""Runs episodes and writes one JSON line per episode.

An episode = one model holding one rule through one scripted conversation.
The model's own replies stay in the history, including any rule-breaking
ones, because "does one slip lead to more?" is part of what we measure.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .constraints import Constraint
from .providers import Provider
from .scenarios import build_schedule, reminder_prefix, system_prompt


@dataclass(frozen=True)
class EpisodeSpec:
    model: str
    constraint: Constraint
    pressure: str
    intervention: str  # "none" or "reminderK"
    episode: int

    @property
    def key(self) -> str:
        return f"{self.model}|{self.constraint.id}|{self.pressure}|{self.intervention}|{self.episode}"

    @property
    def reminder_every(self) -> int | None:
        if self.intervention == "none":
            return None
        if self.intervention.startswith("reminder") and self.intervention[8:].isdigit():
            return int(self.intervention[8:])
        raise ValueError(f"bad intervention {self.intervention!r}; use 'none' or e.g. 'reminder5'")


def _call_seed(seed: int, key: str, turn: int) -> int:
    return int(hashlib.sha256(f"{seed}|{key}|{turn}".encode()).hexdigest()[:8], 16)


def run_episode(provider: Provider, spec: EpisodeSpec, *, turns: int, seed: int,
                temperature: float, stop_on_violation: bool = False) -> dict:
    c = spec.constraint
    schedule = build_schedule(c, spec.pressure, spec.episode, turns, seed)
    messages: list[dict] = [{"role": "system", "content": system_prompt(c)}]
    log, first_violation, prev_violated = [], None, False
    every = spec.reminder_every

    for t, step in enumerate(schedule, start=1):
        reminded = bool(every and t % every == 0)
        user = (reminder_prefix(c) if reminded else "") + step["text"]
        messages.append({"role": "user", "content": user})
        reply = provider.chat(
            spec.model, messages, seed=_call_seed(seed, spec.key, t), temperature=temperature,
            hint={"constraint": c.id, "kind": step["kind"], "prev_violated": prev_violated,
                  "reminded": reminded},
        )
        ok, reason = c.check(reply.text)
        # Rough size of what we sent (chars/4). Ollama's prompt_eval_count can
        # exclude KV-cached tokens, so we keep our own estimate for truncation checks.
        est_prompt = sum(len(m["content"]) for m in messages) // 4
        messages.append({"role": "assistant", "content": reply.text})
        log.append({
            "t": t, "kind": step["kind"], "reminded": reminded, "user": user,
            "reply": reply.text, "ok": ok, "reason": reason,
            "prompt_tokens": reply.prompt_tokens, "est_prompt_tokens": est_prompt,
            "completion_tokens": reply.completion_tokens,
            "latency_s": round(reply.latency_s, 3), **({"meta": reply.meta} if reply.meta else {}),
        })
        if not ok and first_violation is None:
            first_violation = t
        prev_violated = not ok
        if not ok and stop_on_violation:
            break

    return {
        "key": spec.key, "model": spec.model, "provider": provider.name,
        "constraint": c.id, "rule": c.rule, "pressure": spec.pressure,
        "intervention": spec.intervention, "episode": spec.episode,
        "max_turns": turns, "seed": seed, "temperature": temperature,
        "context_window": provider.context_window(spec.model),
        "first_violation": first_violation, "turns": log,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "chl_version": __version__,
    }


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                pass  # half-written line from an interrupted run; it will be redone
    return done


def run_grid(provider: Provider, *, models: list[str], constraints: list[Constraint],
             pressures: list[str], interventions: list[str], episodes: int, turns: int,
             out: Path, seed: int = 0, temperature: float = 0.7, workers: int = 1,
             stop_on_violation: bool = False, quiet: bool = False) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    specs = [
        EpisodeSpec(m, c, p, i, e)
        for m, c, p, i, e in itertools.product(models, constraints, pressures, interventions, range(episodes))
    ]
    for s in specs:
        s.reminder_every  # validate early
    todo = [s for s in specs if s.key not in done]
    if not quiet:
        print(f"{len(specs)} episodes in grid, {len(specs) - len(todo)} already done, "
              f"{len(todo)} to run (~{len(todo) * turns} model calls).", file=sys.stderr)

    lock = threading.Lock()
    t0 = time.time()

    def work(spec: EpisodeSpec) -> dict:
        return run_episode(provider, spec, turns=turns, seed=seed, temperature=temperature,
                           stop_on_violation=stop_on_violation)

    with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, s): s for s in todo}
        for n, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if not quiet:
                fv = rec["first_violation"]
                eta = (time.time() - t0) / n * (len(todo) - n)
                print(f"[{n}/{len(todo)}] {rec['key']}: "
                      f"{'held all ' + str(turns) + ' turns' if fv is None else 'broke at turn ' + str(fv)}"
                      f"  (eta {eta / 60:.1f} min)", file=sys.stderr)
    return out


def load_records(path: Path) -> list[dict]:
    recs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs
