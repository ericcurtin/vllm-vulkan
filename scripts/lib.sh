#!/bin/bash
# Common library functions for vllm-vulkan scripts

error() {
  echo "Error: $*" >&2
}

success() {
  echo "OK: $*"
}

section() {
  echo "=== $* ==="
}

is_apple_silicon() {
  [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]
}

is_macos() {
  [ "$(uname -s)" = "Darwin" ]
}

ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12 --seed
  fi
  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
}

ensure_vulkan() {
  # Install the Vulkan loader/headers needed to compile the Rust extension.
  # Must be called before maturin/cargo runs (i.e. before install_dev_deps).
  if is_macos; then
    if ! brew list molten-vk &>/dev/null 2>&1; then
      section "Installing KosmicKrisp / MoltenVK (macOS Vulkan translation layer)"
      brew install molten-vk
    fi
  else
    if ! ldconfig -p 2>/dev/null | grep -q libvulkan; then
      section "Installing Vulkan loader (Linux)"
      sudo apt-get update -qq
      sudo apt-get install -y libvulkan-dev
    fi
  fi
}

setup_dev_env() {
  ensure_uv
  ensure_vulkan
  ensure_venv ".venv-vllm-vulkan"
  install_dev_deps
}

get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}
