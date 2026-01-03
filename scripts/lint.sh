#!/bin/bash

installs() {
  section "Installing lint tools"

  if is_supported_platform; then
    if ! command -v shellcheck &> /dev/null; then
      if [ "$(uname)" = "Darwin" ]; then
        brew install shellcheck
      else
        sudo apt-get install -y shellcheck || true
      fi
    fi

    if ! command -v ruff &> /dev/null; then
      if [ "$(uname)" = "Darwin" ]; then
        brew install ruff
      else
        uv tool install ruff || true
      fi
    fi
  fi
}

linters() {
  section "Running shellcheck"
  shellcheck -- *.sh scripts/*.sh

  section "Running ruff linter (with auto-fix)"
  ruff check --fix .

  section "Running ruff formatter"
  ruff format .

  section "Running mypy type checker"
  mypy vllm_vulkan
}

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  installs

  linters
}

main "$@"
