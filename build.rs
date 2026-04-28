// SPDX-License-Identifier: Apache-2.0
//! Build script for vllm-vulkan.
//!
//! Platform routing:
//!   macOS (aarch64 / x86_64) — link against libvulkan.dylib installed by
//!     KosmicKrisp (Mesa/Zink software Vulkan driver for macOS).
//!     KosmicKrisp provides a standard libvulkan.dylib loader.
//!
//!   Linux (x86_64 / aarch64) — link against the system libvulkan.so loader
//!     installed via libvulkan-dev (Debian/Ubuntu) or vulkan-loader-devel
//!     (Fedora/RHEL).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ggml");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    build_and_link_ggml(&target_os);

    match target_os.as_str() {
        "macos" => link_macos(),
        "linux" => link_linux(),
        other => {
            println!(
                "cargo:warning=vllm-vulkan: unsupported target OS '{other}'. \
                 Only macOS (via KosmicKrisp) and Linux are supported."
            );
        }
    }
}

// ─── ggml ───────────────────────────────────────────────────────────────────

fn build_and_link_ggml(target_os: &str) {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let ggml_dir = manifest_dir.join("ggml");
    if !ggml_dir.join("CMakeLists.txt").exists() {
        panic!("ggml submodule is missing. Run: git submodule update --init --recursive");
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("ggml-build");
    std::fs::create_dir_all(&build_dir).expect("failed to create ggml build directory");

    let mut configure = std::process::Command::new("cmake");
    configure
        .arg("-S")
        .arg(&ggml_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_VULKAN=ON")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_METAL=OFF")
        .arg("-DGGML_BLAS=OFF")
        .arg("-DGGML_ACCELERATE=OFF")
        .arg("-DGGML_OPENMP=OFF")
        .arg("-DGGML_NATIVE=OFF");

    if target_os == "macos" {
        add_macos_vulkan_cmake_hints(&mut configure);
        add_macos_homebrew_include_hints(&mut configure);
    }

    if let Some(glslc) = find_executable("glslc").or_else(|| {
        find_file(&[
            "/opt/homebrew/opt/shaderc/bin/glslc",
            "/usr/local/opt/shaderc/bin/glslc",
        ])
    }) {
        configure.arg(format!("-DVulkan_GLSLC_EXECUTABLE={}", glslc.display()));
    }

    run_command(configure, "configure ggml with Vulkan");

    let mut build = std::process::Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release")
        .arg("--target")
        .arg("ggml")
        .arg("--parallel");
    run_command(build, "build ggml Vulkan backend");

    for dir in [
        build_dir.join("src"),
        build_dir.join("src/ggml-cpu"),
        build_dir.join("src/ggml-vulkan"),
    ] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Keep this order: ggml references backend registration symbols, and the
    // backend libraries reference ggml-base symbols.
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-vulkan");
    println!("cargo:rustc-link-lib=static=ggml-base");

    match target_os {
        "macos" => println!("cargo:rustc-link-lib=c++"),
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
        }
        _ => {}
    }
}

fn add_macos_vulkan_cmake_hints(configure: &mut std::process::Command) {
    for prefix in ["/opt/homebrew", "/usr/local"] {
        let prefix = std::path::Path::new(prefix);
        let include_dir = prefix.join("include");
        let library = prefix.join("lib/libvulkan.dylib");
        if include_dir.join("vulkan/vulkan.h").exists() && library.exists() {
            configure
                .arg(format!("-DVulkan_INCLUDE_DIR={}", include_dir.display()))
                .arg(format!("-DVulkan_LIBRARY={}", library.display()));
            return;
        }
    }
}

fn add_macos_homebrew_include_hints(configure: &mut std::process::Command) {
    for include_dir in ["/opt/homebrew/include", "/usr/local/include"] {
        let include_dir = std::path::Path::new(include_dir);
        if include_dir.join("spirv/unified1/spirv.hpp").exists() {
            configure
                .arg(format!("-DCMAKE_C_FLAGS=-I{}", include_dir.display()))
                .arg(format!("-DCMAKE_CXX_FLAGS=-I{}", include_dir.display()));
            return;
        }
    }
}

fn run_command(mut command: std::process::Command, label: &str) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to {label}: {err}"));
    if !status.success() {
        panic!("{label} failed with status {status}");
    }
}

fn find_executable(name: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn find_file(paths: &[&str]) -> Option<std::path::PathBuf> {
    paths
        .iter()
        .map(std::path::PathBuf::from)
        .find(|candidate| candidate.is_file())
}

// ─── macOS ───────────────────────────────────────────────────────────────────

fn link_macos() {
    // Probe standard locations for libvulkan.dylib:
    //   /opt/homebrew/lib — Homebrew vulkan-loader (CI / dev builds)
    //   /usr/local/lib    — KosmicKrisp end-user install (install.sh)
    let search_paths = ["/opt/homebrew/lib", "/usr/local/lib"];

    let mut linked = false;
    for lib_dir in &search_paths {
        if std::path::Path::new(lib_dir)
            .join("libvulkan.dylib")
            .exists()
        {
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rustc-link-lib=dylib=vulkan");
            linked = true;
            break;
        }
    }

    if !linked {
        println!(
            "cargo:warning=libvulkan.dylib not found. \
             Install KosmicKrisp: curl -fsSL https://raw.githubusercontent.com/ericcurtin/vllm-vulkan/main/install.sh | bash"
        );
        // Attempt to link anyway; the linker error will be descriptive.
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    // Metal and related frameworks required by the Vulkan loader on macOS.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}

// ─── Linux ───────────────────────────────────────────────────────────────────

fn link_linux() {
    // Standard Vulkan loader; provided by libvulkan-dev / vulkan-loader.
    //
    // Probe for the unversioned linker stub (libvulkan.so), not the runtime
    // library (libvulkan.so.1), because -lvulkan resolves via the unversioned
    // symlink which is only present in the -dev package.
    let search_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ];

    let found = search_paths
        .iter()
        .any(|dir| std::path::Path::new(dir).join("libvulkan.so").exists());

    if !found {
        println!(
            "cargo:warning=libvulkan.so not found. \
             Install libvulkan-dev: sudo apt-get install -y libvulkan-dev"
        );
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}
