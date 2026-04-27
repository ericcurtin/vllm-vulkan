#!/bin/bash

fetch_latest_release() {
  local repo_owner="$1"
  local repo_name="$2"

  echo "Fetching latest release..." >&2

  local latest_release_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
  local release_data

  if ! release_data=$(curl -fsSL "$latest_release_url" 2>&1); then
    error "Failed to fetch release information."
    echo "Please check your internet connection and try again." >&2
    exit 1
  fi

  if [[ -z "$release_data" ]] || [[ "$release_data" == *"Not Found"* ]]; then
    error "No releases found for this repository."
    echo "Please visit https://github.com/${repo_owner}/${repo_name}/releases" >&2
    exit 1
  fi

  echo "$release_data"
}

extract_wheel_url() {
  local release_data="$1"

  python3 -c "
import sys
import json
try:
    data = json.loads('''$release_data''')
    assets = data.get('assets', [])
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl'):
            print(asset.get('browser_download_url', ''))
            break
except Exception as e:
    print('', file=sys.stderr)
"
}

download_and_install_wheel() {
  local wheel_url="$1"
  local package_name="$2"

  local wheel_name
  wheel_name=$(basename "$wheel_url")
  echo "Latest release: $wheel_name"
  success "Found latest release"

  local tmp_dir
  tmp_dir=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp_dir'" EXIT

  echo ""
  echo "Downloading wheel..."
  local wheel_path="$tmp_dir/$wheel_name"

  if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
    error "Failed to download wheel."
    exit 1
  fi

  success "Downloaded wheel"

  if ! uv pip install "$wheel_path"; then
    error "Failed to install ${package_name}."
    exit 1
  fi

  success "Installed ${package_name}"
}

install_system_vulkan_deps() {
  if is_macos; then
    section "Installing KosmicKrisp / MoltenVK (macOS Vulkan translation layer)"
    if ! brew list molten-vk &>/dev/null 2>&1; then
      if ! brew install molten-vk; then
        error "Failed to install MoltenVK via Homebrew."
        echo "Please install KosmicKrisp or MoltenVK manually and re-run." >&2
        exit 1
      fi
    fi
    success "MoltenVK/KosmicKrisp present"
  else
    section "Checking Vulkan loader (Linux)"
    if ! ldconfig -p 2>/dev/null | grep -q libvulkan || ! [ -f /usr/include/vulkan/vulkan.h ] 2>/dev/null; then
      echo "Vulkan loader not found. Install it with:"
      echo "  Debian/Ubuntu: sudo apt-get install -y libvulkan-dev"
      echo "  Fedora/RHEL:   sudo dnf install -y vulkan-loader-devel"
      echo ""
    fi
  fi
}

main() {
  set -eu -o pipefail

  local repo_owner="vllm-project"
  local repo_name="vllm-vulkan"
  local package_name="vllm-vulkan"

  local local_lib=""
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"
    local_lib="$script_dir/scripts/lib.sh"
  fi

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # shellcheck source=/dev/null
    source "$local_lib"
  else
    local lib_url="https://raw.githubusercontent.com/$repo_owner/$repo_name/main/scripts/lib.sh"
    local lib_tmp
    lib_tmp=$(mktemp)
    if ! curl -fsSL "$lib_url" -o "$lib_tmp"; then
      echo "Error: Failed to fetch lib.sh from $lib_url" >&2
      rm -f "$lib_tmp"
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$lib_tmp"
    rm -f "$lib_tmp"
  fi

  install_system_vulkan_deps

  if ! ensure_uv; then
    exit 1
  fi

  local venv="$HOME/.venv-vllm-vulkan"
  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    venv="$PWD/.venv-vllm-vulkan"
  fi

  ensure_venv "$venv"

  # Install vLLM CPU build (no CUDA dependencies)
  local vllm_v="0.19.1"
  local url_base="https://github.com/vllm-project/vllm/releases/download"
  local filename="vllm-$vllm_v.tar.gz"
  curl -OL "$url_base/v$vllm_v/$filename"
  tar xf "$filename"
  cd "vllm-$vllm_v"

  uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
  CXXFLAGS="-Wno-parentheses" uv pip install .
  cd -
  rm -rf "vllm-$vllm_v"*

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    uv pip install .
  else
    local release_data
    release_data=$(fetch_latest_release "$repo_owner" "$repo_name")

    local wheel_url
    wheel_url=$(extract_wheel_url "$release_data")

    if [[ -z "$wheel_url" ]]; then
      error "No wheel file found in the latest release."
      exit 1
    fi

    download_and_install_wheel "$wheel_url" "$package_name"
  fi

  echo ""
  success "Installation complete!"
  echo ""
  echo "Activate the virtual environment:"
  echo "  source $venv/bin/activate"
}

main "$@"
