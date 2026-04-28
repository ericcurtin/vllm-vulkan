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

  rm -rf "$build_dir"
  success "KosmicKrisp built and installed"
}

ensure_submodules() {
  if [ -f ".gitmodules" ] && [ -d ".git" ]; then
    section "Initializing git submodules"
    git submodule update --init --recursive
  fi
}

ensure_vulkan() {
  # Install the Vulkan loader/headers and shader tooling needed to compile
  # ggml's Vulkan backend from source.
  # Must be called before maturin/cargo runs (i.e. before install_dev_deps).
  #
  # On macOS: install vulkan-headers + vulkan-loader via Homebrew. This is
  # fast and sufficient for compilation. The full KosmicKrisp runtime driver
  # is only needed at runtime and is installed separately by install.sh.
  # On Linux: we check for the unversioned linker stub (libvulkan.so) rather
  # than the runtime library (libvulkan.so.1), because the linker needs
  # -lvulkan which resolves via the unversioned symlink in the -dev package.
  if is_macos; then
    section "Checking Vulkan build dependencies (macOS)"
    local missing=()
    for pkg in vulkan-headers vulkan-loader shaderc spirv-headers; do
      if ! brew list "$pkg" &>/dev/null 2>&1; then
        missing+=("$pkg")
      fi
    done
    if [ "${#missing[@]}" -gt 0 ]; then
      brew install "${missing[@]}"
    fi
    success "Vulkan build dependencies present"
  else
    section "Checking Vulkan build dependencies (Linux)"
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq
      sudo apt-get install -y build-essential cmake libvulkan-dev glslc spirv-headers
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y cmake gcc-c++ make vulkan-loader-devel glslc spirv-headers
    else
      echo "Could not detect apt-get or dnf. Install these packages manually:"
      echo "  Debian/Ubuntu: sudo apt-get install -y build-essential cmake libvulkan-dev glslc spirv-headers"
      echo "  Fedora/RHEL:   sudo dnf install -y cmake gcc-c++ make vulkan-loader-devel glslc spirv-headers"
    fi
  fi
}

setup_dev_env() {
  ensure_uv
  ensure_submodules
  ensure_vulkan
  ensure_venv ".venv-vllm-vulkan"
  install_dev_deps
}

get_version() {
  uv run --no-project python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}
