#!/bin/bash
# Common library functions for vllm-vulkan scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

# Check if running on a supported platform
is_supported_platform() {
  # Vulkan supports Linux and macOS
  [ "$(uname)" = "Linux" ] || [ "$(uname)" = "Darwin" ]
}

# Ensure uv is installed
ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

# Install dev dependencies (using CPU-only PyTorch to avoid CUDA)
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv ".venv-vllm-vulkan"
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}

