#!/bin/bash

installs() {
  section "Installing lint tools"

  if is_macos; then
    if ! command -v shellcheck &> /dev/null; then
      brew install shellcheck
    fi
    if ! command -v ruff &> /dev/null; then
      brew install ruff
    fi
  fi
}

linters() {
  section "Running shellcheck"
  shellcheck -- scripts/*.sh install.sh

  section "Running ruff linter"
  ruff check .

  section "Running ruff formatter check"
  ruff format --check .

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
