#!/bin/bash

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  local version
  version=$(get_version)
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
