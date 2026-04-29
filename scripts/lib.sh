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
  local install_dir="${KOSMICKRISP_INSTALL_DIR:-$HOME/.local}"
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

  # Install into user-writable directory (no sudo required)
  mkdir -p "$lib_dir" "$icd_dir"
  cp "$stage_dir/lib/"*.dylib "$lib_dir/"
  cp "$stage_dir/share/vulkan/icd.d/"*.json "$icd_dir/"

  rm -rf "$build_dir"
  success "KosmicKrisp built and installed to $install_dir"
}

ensure_vulkan() {
  # Install the Vulkan headers/loader and GLSL shader compiler needed to build.
  # Must be called before maturin/cargo runs (i.e. before install_dev_deps).
  #
  # On macOS: vulkan-headers + vulkan-loader + glslang (provides glslangValidator)
  # On Linux: libvulkan-dev + glslang-tools (provides glslangValidator)
  if is_macos; then
    local brew_pkgs=()
    if ! brew list vulkan-loader &>/dev/null 2>&1; then
      brew_pkgs+=(vulkan-headers vulkan-loader)
    fi
    if ! command -v glslangValidator &>/dev/null; then
      brew_pkgs+=(glslang)
    fi
    if [ ${#brew_pkgs[@]} -gt 0 ]; then
      section "Installing Vulkan tools (macOS): ${brew_pkgs[*]}"
      brew install "${brew_pkgs[@]}"
    fi
  else
    local need_vulkan=0
    local need_glslang=0

    # Ubuntu 24.04 ships libvulkan1 (runtime) by default, but the linker needs
    # libvulkan.so (the unversioned symlink) from libvulkan-dev.
    if ! find /usr/lib /usr/lib64 /usr/local/lib -name "libvulkan.so" 2>/dev/null | grep -q .; then
      need_vulkan=1
    fi

    if ! command -v glslangValidator &>/dev/null; then
      need_glslang=1
    else
      local glslang_ver_str
      glslang_ver_str="$(glslangValidator --version 2>&1 | head -1)"
      # Version string: "Glslang Version: 11:15.1.0" — extract the major number
      local glslang_major
      glslang_major="${glslang_ver_str##*:}"
      glslang_major="${glslang_major%%.*}"
      if [ "${glslang_major:-0}" -lt 16 ] 2>/dev/null; then
        need_glslang=1
      fi
    fi

    if [ "$need_vulkan" -eq 1 ] || [ "$need_glslang" -eq 1 ]; then
      section "Installing Vulkan tools (Linux)"
      sudo apt-get update -qq
      if [ "$need_vulkan" -eq 1 ]; then
        sudo apt-get install -y libvulkan-dev
      fi

      if [ "$need_glslang" -eq 1 ]; then
        sudo apt-get install -y glslang-tools
      fi
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
  uv run --no-project python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}
