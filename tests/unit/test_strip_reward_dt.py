"""Unit: scripts/benchmark-generator/strip_reward_dt.py — the fail-closed dt-strip.

This script edits **vendored IsaacLab source**, so it has the widest blast radius in the
plugin. Its whole safety argument is that it is structure-agnostic and fails CLOSED: it
strips the layouts it recognizes, then re-scans and refuses to claim success if any dt
scaling survived. These tests pin that argument, especially the case that matters — an
IsaacLab release whose `compute()` was restructured into a shape the stripper does not know.
"""
import importlib.util

import pytest

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "benchmark-generator" / "strip_reward_dt.py"


def _mod():
    spec = importlib.util.spec_from_file_location("strip_reward_dt", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()

STOCK = '''\
class RewardManager(ManagerBase):
    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_cfg.weight == 0.0:
                continue
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._reward_buf += value
            self._step_reward[:, self._term_names.index(name)] = value / dt
        return self._reward_buf

    def reset(self, env_ids=None):
        pass
'''


def _lines(text):
    return text.splitlines(keepends=True)


def test_finds_the_compute_body_and_stops_at_the_next_def():
    lines = _lines(STOCK)
    i, j = M.find_compute_span(lines)
    body = "".join(lines[i:j])
    assert "term_cfg.func" in body
    assert "def reset" not in body, "span ran past compute() into the next method"


def test_stock_layout_is_detected_then_fully_stripped():
    lines = _lines(STOCK)
    span = M.find_compute_span(lines)
    assert len(M.dt_scaling_lines(lines, span)) == 2, "both the value and the log line scale by dt"

    assert M.best_effort_strip(lines, span) is True
    # The POSTCONDITION, which is the actual guarantee: nothing dt-scaled survives.
    assert M.dt_scaling_lines(lines, M.find_compute_span(lines)) == []
    out = "".join(lines)
    assert "term_cfg.weight" in out and "self._reward_buf += value" in out, "stripped too much"
    assert out.count(M.MARKER) == 2


def test_running_twice_changes_nothing():
    lines = _lines(STOCK)
    M.best_effort_strip(lines, M.find_compute_span(lines))
    once = "".join(lines)
    assert M.best_effort_strip(lines, M.find_compute_span(lines)) is False
    assert "".join(lines) == once


@pytest.mark.parametrize("scaling", [
    "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * self.step_dt",
    "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * self._dt",
])
def test_dt_token_variants_are_recognized(scaling):
    src = STOCK.replace(
        "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt", scaling)
    lines = _lines(src)
    span = M.find_compute_span(lines)
    assert M.dt_scaling_lines(lines, span), f"missed dt-like token in: {scaling}"
    M.best_effort_strip(lines, span)
    assert M.dt_scaling_lines(lines, M.find_compute_span(lines)) == []


def test_unrecognized_layout_survives_the_strip_so_the_caller_can_fail_closed():
    """The case the fail-closed gate exists for: a future IsaacLab restructures compute()
    into a shape best_effort_strip does not know. It must NOT be silently half-stripped —
    the postcondition scan has to still see dt scaling so main() can refuse."""
    restructured = STOCK.replace(
        "value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt",
        "value = self._scale(term_cfg) \n            value *= dt",
    )
    lines = _lines(restructured)
    span = M.find_compute_span(lines)
    M.best_effort_strip(lines, span)
    survivors = M.dt_scaling_lines(lines, M.find_compute_span(lines))
    assert survivors, "an unrecognized `value *= dt` was not reported — the gate would pass wrongly"


def test_dt_outside_compute_is_left_alone():
    src = STOCK + '''
    def other(self, dt):
        return self.thing * dt
'''
    lines = _lines(src)
    M.best_effort_strip(lines, M.find_compute_span(lines))
    assert "self.thing * dt" in "".join(lines), "stripped dt outside compute()"


def test_inline_comment_does_not_hide_scaling():
    src = STOCK.replace(
        "* term_cfg.weight * dt", "* term_cfg.weight * dt  # scale to per-second")
    lines = _lines(src)
    assert M.dt_scaling_lines(lines, M.find_compute_span(lines))


def test_missing_compute_returns_none():
    assert M.find_compute_span(_lines("class X:\n    def other(self):\n        pass\n")) is None
