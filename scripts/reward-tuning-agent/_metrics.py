"""Shared metrics.jsonl reader for the reward-tune scripts.

Both `curve_health.py` (live snapshot) and `score_iter.py` (final verdict) read the
same training log, so the format tolerance lives here once. Stdlib only.

Long format  — one {"key", "step", "value"} row per point.
Wide format  — one {"<key>": v, ..., "step": s} row per logging tick.
Both are accepted; a half-written trailing line is ignored so a LIVE file is safe to read.
"""
import json


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # live file: ignore a half-written trailing line
    return rows


def series(rows):
    """Return {key: [(step, value), ...]} sorted by step, for long OR wide format.

    Format is decided by a majority of the rows, not by rows[0]: one wide-format preamble
    row at the head of a long file (a config dump, a run header) would otherwise switch the
    whole reader into the wrong branch and silently yield an empty series.
    """
    out = {}
    long_rows = sum(1 for r in rows if isinstance(r, dict) and "key" in r and "value" in r)
    if rows and long_rows * 2 > len(rows):  # long format
        for r in rows:
            k, s, v = r.get("key"), r.get("step"), r.get("value")
            if k is None or v is None:
                continue
            out.setdefault(k, []).append((s if s is not None else len(out.get(k, [])), v))
    else:  # wide format
        for i, r in enumerate(rows):
            s = r.get("step", r.get("global_step", i))
            # An explicit null step is not the same as an absent one, and `.get(k, i)` only
            # covers the absent case — leaving None to blow up the sort below on a file that
            # is otherwise perfectly readable.
            if s is None:
                s = i
            for k, v in r.items():
                if k in ("step", "global_step") or not isinstance(v, (int, float)):
                    continue
                out.setdefault(k, []).append((s, v))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def load_series(path):
    return series(load_rows(path))


def final_values(ser):
    """{key: last value} — the value at the largest step for each key."""
    return {k: pts[-1][1] for k, pts in ser.items() if pts}


def peak_values(ser):
    """{key: (best value, step at which it occurred)} over the whole curve.

    RL runs routinely peak mid-training and degrade: scoring only the last point can
    understate a candidate several-fold. Reported ALONGSIDE the final value, never instead
    of it — a lone noisy spike is not a policy's quality, so the designer needs to see both.
    """
    out = {}
    for k, pts in ser.items():
        vals = [(v, st) for st, v in pts if isinstance(v, (int, float))]
        if vals:
            out[k] = max(vals, key=lambda t: t[0])
    return out
