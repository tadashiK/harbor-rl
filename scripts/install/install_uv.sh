#!/usr/bin/env bash
# Install uv (https://docs.astral.sh/uv) for the current user.
# No sudo required. Idempotent: skips if uv is already on PATH.
set -e

if command -v uv >/dev/null 2>&1; then
    echo "[uv] already installed: $(uv --version)"
    exit 0
fi

echo "[uv] installing from astral.sh ..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv installs to ~/.local/bin or ~/.cargo/bin; both should be on PATH after shell restart.
# Source the env file so the current shell sees it immediately.
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

if command -v uv >/dev/null 2>&1; then
    echo "[uv] installed: $(uv --version)"
else
    echo "[uv] installed but not yet on PATH. Restart your shell or run: source ~/.bashrc"
fi
