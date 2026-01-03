#!/bin/bash

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  # Only set up uv and venv, skip installing dependencies (avoids CUDA)
  ensure_uv
  ensure_venv ".venv-vllm-vulkan"

  local version
  version=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
  echo "Building version: $version"

  section "Building wheel"
  uv build

  local tag
  tag="v${version}-$(date +%Y%m%d-%H%M%S)"
  echo "Generated tag: $tag"

  local commit_sha
  commit_sha="${GITHUB_SHA:-$(git rev-parse HEAD)}"

  section "Creating GitHub release"
  gh release create "$tag" \
    --title "Release $tag" \
    --notes "Automated release for commit $commit_sha" \
    dist/*.whl
}

main "$@"
