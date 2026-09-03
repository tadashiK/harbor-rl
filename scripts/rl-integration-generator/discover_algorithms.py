"""
Probe the host environment for which algorithm sources are usable.

Echoes JSON describing which sources will work without further setup, and
which would need pip installs:

    {
      "stable-baselines3": {"installed": true,  "version": "2.1.0"},
      "custom-torch":      {"installed": true,  "version": "2.1.0"},   # ships in plugin templates; checks torch
      "custom-jax":        {"installed": true,  "version": "0.4.x"},   # checks jax + flax + optax
      "custom-library":    {"installed": null}                          # depends on user input
    }

Used by rl-integration-generator to default the user's algorithm-source
choice and to know whether to inject extra pip lines into setup_uv.sh.
"""
from __future__ import annotations

import argparse
import importlib
import json


def probe(pkg: str) -> dict:
    try:
        mod = importlib.import_module(pkg)
        return {"installed": True, "version": getattr(mod, "__version__", "unknown")}
    except ImportError:
        return {"installed": False}


def main():
    # No inputs — it only probes the host. argparse is here so `--help` works and the
    # script is discoverable like every other tool in scripts/ (CLAUDE.md L4 rule 1).
    argparse.ArgumentParser(description=__doc__.strip().splitlines()[0]).parse_args()
    torch_p = probe("torch")
    jax_p = probe("jax")
    flax_p = probe("flax")
    optax_p = probe("optax")
    out = {
        "stable-baselines3": probe("stable_baselines3"),
        "custom-torch": {
            "installed": torch_p["installed"],
            "version": torch_p.get("version"),
            "package": "torch",
        },
        "custom-jax": {
            "installed": all(p["installed"] for p in (jax_p, flax_p, optax_p)),
            "version": jax_p.get("version"),
            "deps": {"jax": jax_p, "flax": flax_p, "optax": optax_p},
        },
    }
    if not out["custom-torch"]["installed"]:
        out["custom-torch"]["hint"] = "pip install torch  # custom_torch needs PyTorch"
    if not out["custom-jax"]["installed"]:
        out["custom-jax"]["hint"] = "pip install jax flax optax  # custom_jax stack"
    if not out["stable-baselines3"]["installed"]:
        out["stable-baselines3"]["hint"] = "pip install 'stable-baselines3[extra]'"
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
