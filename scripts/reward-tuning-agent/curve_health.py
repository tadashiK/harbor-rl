#!/usr/bin/env python3
"""curve_health.py — compact health snapshot of a LIVE reward-tune training run.

Reads a training `metrics.jsonl` (long format: one {"key","step","value"} row per
point; wide format {"<key>": v, "step": s} also accepted) and emits a single compact
JSON object describing the shape of the reward + learning curves so far. It GATHERS
evidence and raises advisory flags; it does NOT decide to kill — the reward-tuning-agent
applies the confidence rubric (warmup floor + persistence across snapshots) to the
`flags` / `concern` here. Stdlib only; robust to a partially-written trailing line.

Usage:
  curve_health.py --metrics <trial>/metrics.jsonl [--total-steps N]
                  [--success-term success_bonus] [--success-weight 400]
                  [--stage0-term reaching_dog] [--window 8]
"""
import argparse
import json
import math
import sys

from _metrics import load_series


def _finite(vals):
    return [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]


def _has_nonfinite(vals):
    return any(isinstance(v, (int, float)) and not math.isfinite(v) for v in vals)


def _trend(pts, window):
    """Classify recent trend of a (step,value) series over the last `window` points.

    Returns (label, rel_change) where label ∈ {rising, flat, declining, unknown} and
    rel_change is (last - window_start) / (|window_mean| + eps) — scale-free so a
    deadband of ±5% reads as flat regardless of the metric's magnitude.
    """
    vals = _finite([v for _, v in pts])
    if len(vals) < 3:
        return "unknown", 0.0
    w = vals[-window:] if len(vals) >= window else vals
    rel = (w[-1] - w[0]) / (abs(sum(w) / len(w)) + 1e-6)
    if rel > 0.05:
        return "rising", rel
    if rel < -0.05:
        return "declining", rel
    return "flat", rel


def _last(pts):
    vals = _finite([v for _, v in pts])
    return vals[-1] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--total-steps", type=float, default=None)
    ap.add_argument("--success-term", default=None)
    ap.add_argument("--success-weight", type=float, default=None)
    ap.add_argument("--stage0-term", default=None)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--soft-floor", type=float, default=0.5,
                    help="budget fraction below which SOFT patterns (flat/declining/"
                         "diverged) are only 'watch', never 'confident_bad'. Early "
                         "curves are noisy and non-monotonic; a doomed-looking dip at "
                         "20-40%% routinely recovers. Hard fails ignore this floor.")
    a = ap.parse_args()

    S = load_series(a.metrics)
    W = a.window

    def ret_key(term):
        return f"reward/{term}/episodic_return_mean" if term else None

    total = S.get("reward/total/episodic_return_mean", [])
    latest_step = max((pts[-1][0] for pts in S.values() if pts), default=None)
    n_points = len(total)

    # NaN/Inf scan across reward + loss series (the unambiguous hard-fail signal).
    nan_inf = any(
        _has_nonfinite([v for _, v in pts])
        for k, pts in S.items()
        if k.startswith("reward/") or k.startswith("train/loss")
    )

    total_vals = _finite([v for _, v in total])
    total_trend, total_rel = _trend(total, W)
    total_last = total_vals[-1] if total_vals else None
    total_max = max(total_vals) if total_vals else None
    total_start = total_vals[0] if total_vals else None
    # "flat near start" = learned essentially nothing above where it began.
    total_flat_near_start = (
        total_last is not None
        and total_start is not None
        and total_max is not None
        and (total_max - total_start) <= 0.10 * (abs(total_start) + 1.0)
    )

    ent = S.get("train/entropy", [])
    ent_last = _last(ent)
    ent_first = _finite([v for _, v in ent])[0] if _finite([v for _, v in ent]) else None
    ent_trend, _ = _trend(ent, W)
    ent_collapsed = (
        ent_last is not None and ent_first is not None
        and ent_last <= 0.05 * abs(ent_first) + 1e-6
    )

    kl = S.get("train/approx_kl", [])
    kl_vals = _finite([v for _, v in kl])
    kl_recent = kl_vals[-W:] if kl_vals else []
    kl_last = kl_vals[-1] if kl_vals else None
    kl_max_recent = max(kl_recent) if kl_recent else None
    kl_exploded = kl_max_recent is not None and kl_max_recent > 0.5

    vloss = S.get("train/loss/value", [])
    vloss_trend, vloss_rel = _trend(vloss, W)
    vloss_exploded = vloss_rel > 5.0  # >500% growth over the window

    # success signal — prefer explicit train/success_rate, else the success term's return.
    succ = {"source": None, "last": None, "success_rate_proxy": None, "trend": "unknown"}
    if S.get("train/success_rate"):
        sr = S["train/success_rate"]
        succ.update(source="train/success_rate", last=_last(sr),
                    success_rate_proxy=_last(sr), trend=_trend(sr, W)[0])
    elif a.success_term and S.get(ret_key(a.success_term)):
        sp = S[ret_key(a.success_term)]
        last = _last(sp)
        succ.update(source=ret_key(a.success_term), last=last, trend=_trend(sp, W)[0])
        if a.success_weight:
            succ["success_rate_proxy"] = (last / a.success_weight) if last is not None else None

    stage0 = None
    if a.stage0_term and S.get(ret_key(a.stage0_term)):
        s0 = S[ret_key(a.stage0_term)]
        stage0 = {"term": a.stage0_term, "last": _last(s0), "trend": _trend(s0, W)[0]}

    budget_frac = None
    if a.total_steps and latest_step is not None:
        budget_frac = round(min(1.0, latest_step / a.total_steps), 4)

    # Advisory flags (the AGENT decides; these only summarize the evidence).
    flags = []
    if nan_inf:
        flags.append("nan_inf")
    if ent_collapsed and total_flat_near_start:
        flags.append("dead_policy")  # collapsed entropy + no learning = hard fail
    if total_flat_near_start:
        flags.append("no_learning")  # total never rose >10% above its start (absolute)
    if total_trend == "declining" and total_max is not None and total_last is not None \
            and not total_flat_near_start \
            and total_last < 0.6 * total_max and total_max > 0.15 * abs(total_start or 0.0) + 1.0:
        flags.append("total_declining")  # fell to <60% of an established (non-trivial) peak
    if kl_exploded and vloss_exploded:
        flags.append("optimization_diverged")
    # Protective: real absolute progress means NOT a kill candidate. Keyed on absolute
    # gain (a noisy last-window slope on near-zero values is not "improving").
    improving = (total_trend == "rising" and not total_flat_near_start) \
        or (succ["trend"] == "rising" and (succ["last"] or 0) > 0)
    if improving:
        flags.append("improving")

    hard = any(f in flags for f in ("nan_inf", "dead_policy"))
    soft = any(f in flags for f in ("no_learning", "total_declining", "optimization_diverged"))
    # SOFT patterns are only trustworthy in the second half — early curves are noisy and
    # non-monotonic (dips at 20-40% routinely recover). Below the floor a soft flag is
    # only "watch". HARD fails are unambiguous and bypass the floor.
    past_floor = a.total_steps is not None and budget_frac is not None and budget_frac >= a.soft_floor
    if hard:
        concern = "hard_fail"
    elif soft and "improving" not in flags and past_floor:
        concern = "confident_bad"
    elif soft:
        concern = "watch"  # bad signal but too early, or contradicted by an improving trend
    else:
        concern = "none"

    print(json.dumps({
        "latest_step": latest_step,
        "budget_frac": budget_frac,
        "n_points": n_points,
        "total_return": {"last": total_last, "start": total_start, "max": total_max,
                         "trend": total_trend, "rel_change": round(total_rel, 4)},
        "success_signal": succ,
        "stage0": stage0,
        "entropy": {"last": ent_last, "trend": ent_trend, "collapsed": ent_collapsed},
        "approx_kl": {"last": kl_last, "max_recent": kl_max_recent, "exploded": kl_exploded},
        "value_loss": {"trend": vloss_trend, "exploded": vloss_exploded},
        "nan_inf": nan_inf,
        "flags": flags,
        "concern": concern,  # advisory: hard_fail | confident_bad | watch | none
    }, indent=2))


if __name__ == "__main__":
    main()
