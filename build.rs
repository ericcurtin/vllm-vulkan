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

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

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

// ─── macOS ───────────────────────────────────────────────────────────────────

fn link_macos() {
    // Probe standard locations for libvulkan.dylib:
    //   /opt/homebrew/lib — Homebrew vulkan-loader (CI / dev builds)
    //   /usr/local/lib    — KosmicKrisp end-user install (install.sh)
    let search_paths = ["/opt/homebrew/lib", "/usr/local/lib"];

    let mut linked = false;
    for lib_dir in &search_paths {
        if std::path::Path::new(lib_dir).join("libvulkan.dylib").exists() {
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

    let found = search_paths.iter().any(|dir| {
        std::path::Path::new(dir).join("libvulkan.so").exists()
    });

    if !found {
        println!(
            "cargo:warning=libvulkan.so not found. \
             Install libvulkan-dev: sudo apt-get install -y libvulkan-dev"
        );
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}
