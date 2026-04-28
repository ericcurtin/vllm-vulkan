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
  local os_tag
  local arch_tag

  case "$(uname -s)" in
    Darwin) os_tag="macosx" ;;
    *)      os_tag="linux"  ;;
  esac

  case "$(uname -m)" in
    arm64|aarch64) arch_tag="arm64\|aarch64" ;;
    *)             arch_tag="x86_64\|amd64"  ;;
  esac

  python3 -c "
import sys
import json
try:
    data = json.loads('''$release_data''')
    assets = data.get('assets', [])
    os_tag = '$os_tag'
    arch_tags = '$arch_tag'.split('\|')
    for asset in assets:
        name = asset.get('name', '')
        if name.endswith('.whl') and os_tag in name and any(a in name for a in arch_tags):
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

install_kosmickrisp() {
  local install_dir="/usr/local"
  local lib_dir="$install_dir/lib"
  local icd_dir="$install_dir/share/vulkan/icd.d"

  if [ -f "$lib_dir/libvulkan_kosmickrisp.dylib" ]; then
    success "KosmicKrisp present"
    return 0
  fi

  section "Building KosmicKrisp from source (macOS Vulkan via Mesa/Zink)"

  # Pinned upstream refs — update these to pick up new Mesa/Vulkan releases.
  local MESA_REF="vulkan-sdk-1.4.341.0"
  local VULKAN_SDK_REF="vulkan-sdk-1.4.341.0"
  local SPIRV_REF="v21.1.4"

  local MESA_REPO="https://gitlab.freedesktop.org/aitor/mesa.git"
  local VULKAN_HEADERS_REPO="https://github.com/KhronosGroup/Vulkan-Headers.git"
  local VULKAN_LOADER_REPO="https://github.com/KhronosGroup/Vulkan-Loader.git"
  local SPIRV_REPO="https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git"

  # Build dependencies
  local missing_brew=()
  for pkg in meson ninja pkg-config "llvm@21" spirv-tools libclc cmake; do
    if ! brew list "$pkg" &>/dev/null 2>&1; then
      missing_brew+=("$pkg")
    fi
  done
  if [[ ${#missing_brew[@]} -gt 0 ]]; then
    echo "Installing build dependencies: ${missing_brew[*]}"
    brew install "${missing_brew[@]}"
  fi

  local LLVM_DIR
  LLVM_DIR="$(brew --prefix llvm@21)"

  # Python build deps
  uv pip install mako packaging pyyaml

  local build_dir
  build_dir=$(mktemp -d)
  # shellcheck disable=SC2064
  trap "rm -rf '$build_dir'" EXIT

  local src_dir="$build_dir/src"
  local stage_dir="$build_dir/stage"
  local deps_dir="$build_dir/deps"
  mkdir -p "$src_dir" "$stage_dir" "$deps_dir"

  local NCPU
  NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

  # 1. Vulkan-Headers
  echo "Cloning Vulkan-Headers..."
  git clone --depth=1 --branch "$VULKAN_SDK_REF" "$VULKAN_HEADERS_REPO" "$src_dir/vulkan-headers"
  cmake -S "$src_dir/vulkan-headers" -B "$src_dir/vulkan-headers-build" \
    -DCMAKE_INSTALL_PREFIX="$deps_dir" -DCMAKE_BUILD_TYPE=Release
  cmake --install "$src_dir/vulkan-headers-build"

  # 2. Vulkan-Loader
  echo "Cloning Vulkan-Loader..."
  git clone --depth=1 --branch "$VULKAN_SDK_REF" "$VULKAN_LOADER_REPO" "$src_dir/vulkan-loader"
  cmake -S "$src_dir/vulkan-loader" -B "$src_dir/vulkan-loader-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$deps_dir" \
    -DCMAKE_INSTALL_PREFIX="$stage_dir" \
    "-DCMAKE_C_FLAGS=-I$deps_dir/include" \
    -DBUILD_TESTS=OFF
  cmake --build "$src_dir/vulkan-loader-build" "-j$NCPU"
  cmake --install "$src_dir/vulkan-loader-build"

  # 3. SPIRV-LLVM-Translator
  echo "Cloning SPIRV-LLVM-Translator..."
  git clone --depth=1 --branch "$SPIRV_REF" "$SPIRV_REPO" "$src_dir/spirv-translator"
  cmake -S "$src_dir/spirv-translator" -B "$src_dir/spirv-translator-build" \
    -DCMAKE_BUILD_TYPE=Release \
    "-DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm" \
    -DCMAKE_INSTALL_PREFIX="$deps_dir"
  cmake --build "$src_dir/spirv-translator-build" "-j$NCPU"
  cmake --install "$src_dir/spirv-translator-build"

  # 4. Mesa (KosmicKrisp Vulkan driver)
  echo "Cloning Mesa (KosmicKrisp)..."
  git clone --depth=1 --branch "$MESA_REF" "$MESA_REPO" "$src_dir/mesa" \
    || { git clone "$MESA_REPO" "$src_dir/mesa" && git -C "$src_dir/mesa" checkout "$MESA_REF"; }

  export CC="/usr/bin/clang"
  export CXX="/usr/bin/clang++"
  export PATH="$LLVM_DIR/bin:$PATH"
  export PKG_CONFIG_PATH="$deps_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

  meson setup "$src_dir/mesa" "$src_dir/mesa-build" \
    --buildtype=release \
    -Dvulkan-drivers=kosmickrisp \
    -Dgallium-drivers= \
    -Dopengl=false \
    -Dzstd=disabled \
    --prefer-static \
    -Dplatforms=macos \
    "--prefix=$stage_dir"
  ninja -C "$src_dir/mesa-build" "-j$NCPU"
  meson install -C "$src_dir/mesa-build"

  # Install into /usr/local
  sudo mkdir -p "$lib_dir" "$icd_dir"
  sudo cp "$stage_dir/lib/"*.dylib "$lib_dir/"
  sudo cp "$stage_dir/share/vulkan/icd.d/"*.json "$icd_dir/"

  success "KosmicKrisp built and installed"
}

install_system_vulkan_deps() {
  if is_macos; then
    install_kosmickrisp
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

  local repo_owner="ericcurtin"
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
